/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package dualpods

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"

	"github.com/llm-d-incubation/llm-d-fast-model-actuation/pkg/api"
	stubapi "github.com/llm-d-incubation/llm-d-fast-model-actuation/pkg/spi"
)

func directEngineConfig(pod *corev1.Pod) (string, time.Duration, error) {
	engine := pod.Annotations[api.EngineAnnotationName]
	if engine == "" {
		engine = "vllm"
	}
	if engine != "vllm" && engine != "sglang" {
		return "", 0, fmt.Errorf("unsupported %s: %q", api.EngineAnnotationName, engine)
	}
	timeout := 5 * time.Minute
	if engine == "sglang" {
		if _, launcher := pod.Annotations[api.InferenceServerConfigAnnotationName]; launcher {
			return "", 0, fmt.Errorf("SGLang is supported only by Direct Providers, not Launcher")
		}
		if value := pod.Annotations[api.EngineOperationTimeoutAnnotationName]; value != "" {
			var err error
			timeout, err = time.ParseDuration(value)
			if err != nil || timeout <= 0 || timeout > time.Hour {
				return "", 0, fmt.Errorf("%s must be a duration in (0, 1h]", api.EngineOperationTimeoutAnnotationName)
			}
		}
	}
	return engine, timeout, nil
}

func isSGLang(pod *corev1.Pod) bool {
	return pod.Annotations[api.EngineAnnotationName] == "sglang"
}

type sglangPhase string

const (
	sglangStarting  sglangPhase = "starting"
	sglangAwake     sglangPhase = "awake"
	sglangReleased  sglangPhase = "released"
	sglangOperating sglangPhase = "operating"
	sglangUnknown   sglangPhase = "unknown"
)

// This is controller knowledge, never an engine-authoritative status. Keeping it
// per Provider UID (not requester) permits warm reuse. Losing it deliberately
// requires cold recovery: neither a sleeping label nor /health proves state.
type sglangState struct {
	phase       sglangPhase
	incarnation string
	name        string
	observed    bool
	stopped     bool
}

func (nd *nodeData) rememberNewSGLang(pod *corev1.Pod) {
	if nd.SGLang == nil {
		nd.SGLang = make(map[types.UID]*sglangState)
	}
	nd.SGLang[pod.UID] = &sglangState{phase: sglangStarting, name: pod.Name}
}

func (nd *nodeData) sglangKnowledge(pod *corev1.Pod) *sglangState {
	if nd.SGLang == nil {
		nd.SGLang = make(map[types.UID]*sglangState)
	}
	state := nd.SGLang[pod.UID]
	if state == nil {
		state = &sglangState{phase: sglangUnknown, name: pod.Name}
		nd.SGLang[pod.UID] = state
	}
	state.observed = true
	cs := getContainerStatus(pod, api.InferenceServerContainerName)
	if cs == nil || cs.State.Running == nil || cs.ContainerID == "" {
		if state.phase != sglangStarting {
			state.phase = sglangUnknown
		}
		return state
	}
	identity := fmt.Sprintf("%s/%d/%s", cs.ContainerID, cs.RestartCount, cs.State.Running.StartedAt.UTC().Format(time.RFC3339Nano))
	if state.phase == sglangStarting && cs.RestartCount == 0 {
		state.incarnation, state.phase = identity, sglangAwake
	} else if state.incarnation != identity {
		state.phase = sglangUnknown
	}
	return state
}

func (ctl *controller) setSGLangSleeping(ctx context.Context, pod *corev1.Pod, port int32, sleeping bool) error {
	node, err := getProviderNodeName(pod)
	if err != nil {
		return err
	}
	state := ctl.getNodeData(node).sglangKnowledge(pod)
	want, path := sglangAwake, "/resume_memory_occupation"
	if sleeping {
		want, path = sglangReleased, "/release_memory_occupation"
	}
	if state.phase == want {
		return nil
	}
	if state.phase != sglangAwake && state.phase != sglangReleased {
		return fmt.Errorf("cannot change SGLang provider %s in state %s", pod.Name, state.phase)
	}
	_, timeout, err := directEngineConfig(pod)
	if err != nil {
		return err
	}
	if pod.Status.PodIP == "" {
		return fmt.Errorf("SGLang provider %s has no IP", pod.Name)
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	logger := klog.FromContext(ctx).WithValues("engine", "sglang", "provider", pod.Name, "providerUID", pod.UID, "incarnation", state.incarnation)
	ctx = klog.NewContext(ctx, logger)
	state.phase = sglangOperating
	// Use the operation deadline, not the shared vLLM client's 30s limit.
	// Never follow redirects for non-idempotent lifecycle operations.
	client := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	url := fmt.Sprintf("http://%s:%d%s", pod.Status.PodIP, port, path)
	body := struct {
		Tags []string `json:"tags"`
	}{[]string{"kv_cache", "cuda_graph", "weights"}}
	_, err = doHTTPWithClient(ctx, client, "sglang"+path, http.MethodPost, url, nil,
		ctl.httpLatencySecsHistograms.MustCurryWith(prometheus.Labels{"isc_name": ""}), body, httpResultConsumer{octets: io.Discard})
	if err != nil {
		// Even a non-2xx can follow partial changes. Never retry it blindly.
		state.phase = sglangUnknown
		return fmt.Errorf("SGLang provider %s operation outcome unknown: %w", pod.Name, err)
	}
	state.phase = want
	logger.V(2).Info("Recorded SGLang operation completion", "state", want)
	return nil
}

// Unbound SGLang providers also need reconciliation (crash, eviction, GC).
type sglangNodeItem struct{}

func (sglangNodeItem) process(ctx context.Context, ctl *controller, nd *nodeData) processResult {
	err := ctl.guardSGLangProviders(ctx, nd)
	return processResult{err: err, retry: err != nil}
}

func (ctl *controller) enqueueSGLangProvider(pod *corev1.Pod) {
	if !isSGLang(pod) || pod.Annotations[nominalHashAnnotationKey] == "" {
		return
	}
	node, err := getProviderNodeName(pod)
	if err != nil {
		ctl.enqueueLogger.Error(err, "Cannot enqueue SGLang provider", "provider", pod.Name)
		return
	}
	ctl.getNodeData(node).add(sglangNodeItem{})
	ctl.Queue.Add(nodeItem{node})
}

// Unknown processes fence actuation on their node, including other engines.
// This conservative node-wide fence avoids inferring safety from GPU usage or
// a possibly stale sleeping label. A retained finalizer lets us wait for kubelet
// terminal status; Delete success or informer disappearance alone is not proof.
func (ctl *controller) guardSGLangProviders(ctx context.Context, nd *nodeData) error {
	pods, err := ctl.podLister.Pods(ctl.namespace).List(labels.Everything())
	if err != nil {
		return err
	}
	seen := make(map[types.UID]bool)
	var pending []error
	for _, pod := range pods {
		if !isSGLang(pod) || pod.Annotations[nominalHashAnnotationKey] == "" {
			continue
		}
		node, err := getProviderNodeName(pod)
		if err != nil || node != nd.NodeName {
			continue
		}
		seen[pod.UID] = true
		state := nd.sglangKnowledge(pod)
		abandonedStart := false
		if state.phase == sglangStarting {
			parts := strings.Fields(pod.Annotations[requesterAnnotationKey])
			if len(parts) == 2 {
				req, err := ctl.podLister.Pods(ctl.namespace).Get(parts[1])
				if err != nil && !apierrors.IsNotFound(err) {
					return err
				}
				abandonedStart = apierrors.IsNotFound(err) || (err == nil && (string(req.UID) != parts[0] || req.DeletionTimestamp != nil))
			}
		}
		// A starting Pod deleted before becoming ready must be stopped too.
		if state.phase == sglangUnknown || state.phase == sglangOperating || pod.DeletionTimestamp != nil || abandonedStart ||
			(state.phase == sglangStarting && (pod.Status.Phase == corev1.PodFailed || pod.Status.Phase == corev1.PodSucceeded)) {
			state.phase = sglangUnknown
			pending = append(pending, ctl.retireSGLang(ctx, pod, state))
		}
	}
	for uid, state := range nd.SGLang {
		if !seen[uid] {
			if !state.observed {
				pending = append(pending, fmt.Errorf("waiting to observe newly created SGLang provider %s (%s)", state.name, uid))
				continue
			}
			if !state.stopped {
				pending = append(pending, fmt.Errorf("SGLang provider %s (%s) disappeared without termination confirmation; node fenced", state.name, uid))
				continue
			}
			delete(nd.SGLang, uid)
		}
	}
	return errors.Join(pending...)
}

func sglangStopped(pod *corev1.Pod) bool {
	// A never-scheduled Pod has no kubelet to report termination. Observing its
	// deletion timestamp is sufficient: the API rejects binding a deleting Pod.
	if pod.Spec.NodeName == "" && pod.DeletionTimestamp != nil {
		return true
	}
	if pod.Status.Phase != corev1.PodSucceeded && pod.Status.Phase != corev1.PodFailed {
		return false
	}
	for _, container := range pod.Spec.Containers {
		cs := getContainerStatus(pod, container.Name)
		if cs == nil || cs.State.Terminated == nil {
			return false
		}
	}
	for _, cs := range pod.Status.InitContainerStatuses {
		if cs.State.Terminated == nil {
			return false
		}
	}
	return true
}

func (ctl *controller) retireSGLang(ctx context.Context, pod *corev1.Pod, state *sglangState) error {
	logger := klog.FromContext(ctx).WithValues("engine", "sglang", "provider", pod.Name, "providerUID", pod.UID)
	ctx = klog.NewContext(ctx, logger)
	logger.V(2).Info("Cold recovery of uncertain or departing SGLang provider", "state", state.phase)
	// Recreate the requester too: its proxy intentionally cannot change target.
	parts := strings.Fields(pod.Annotations[requesterAnnotationKey])
	if len(parts) == 2 {
		req, err := ctl.podLister.Pods(ctl.namespace).Get(parts[1])
		if err != nil && !apierrors.IsNotFound(err) {
			return err
		}
		if err == nil && string(req.UID) == parts[0] && req.DeletionTimestamp == nil {
			if !slices.Contains(req.Finalizers, requesterFinalizer) {
				if _, err = ctl.addRequesterFinalizer(ctx, req, pod.Name, ""); err != nil {
					return err
				}
			}
			if req.Status.PodIP != "" {
				port := req.Annotations[api.AdminPortAnnotationName]
				if port == "" {
					port = api.AdminPortDefaultValue
				}
				url := fmt.Sprintf("http://%s:%s%s", req.Status.PodIP, port, stubapi.BecomeUnreadyPath)
				// If the proxy is unreachable, deleting its Pod still removes it from
				// Service endpoints. Do not let that failure prevent termination.
				_, _ = doHTTP(ctx, "sglang_unready", http.MethodPost, url, nil, ctl.httpLatencySecsHistograms.MustCurryWith(prometheus.Labels{"isc_name": ""}), nil, httpResultConsumer{})
			}
			if err := ctl.deleteSGLangPod(ctx, req); err != nil {
				return err
			}
		}
	}
	if pod.DeletionTimestamp == nil {
		if err := ctl.deleteSGLangPod(ctx, pod); err != nil {
			return err
		}
	}
	if sglangStopped(pod) {
		state.stopped = true
		if _, err := ctl.removeProviderFinalizer(ctx, pod); err != nil {
			return err
		}
	}
	return fmt.Errorf("waiting for SGLang provider %s termination and removal", pod.Name)
}

func (ctl *controller) deleteSGLangPod(ctx context.Context, pod *corev1.Pod) error {
	start := time.Now()
	err := ctl.coreclient.Pods(ctl.namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{Preconditions: &metav1.Preconditions{UID: &pod.UID}})
	klog.FromContext(ctx).V(2).Info("Requested Pod deletion for SGLang cold recovery", "pod", pod.Name, "uid", pod.UID, "k8sCallStartTime", start.Format(time.RFC3339Nano), "err", err)
	if apierrors.IsNotFound(err) {
		return nil
	}
	return err
}
