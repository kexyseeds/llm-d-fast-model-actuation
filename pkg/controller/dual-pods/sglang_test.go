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
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes/fake"
	corelisters "k8s.io/client-go/listers/core/v1"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	kubemetrics "k8s.io/component-base/metrics"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/yaml"

	"github.com/llm-d-incubation/llm-d-fast-model-actuation/pkg/api"
	stubapi "github.com/llm-d-incubation/llm-d-fast-model-actuation/pkg/spi"
)

func testSGLangPod() *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "provider", Namespace: "test", UID: "provider-uid", Finalizers: []string{providerFinalizer}, Annotations: map[string]string{
			api.EngineAnnotationName: "sglang", nominalHashAnnotationKey: "hash", requesterAnnotationKey: "req-uid requester",
		}},
		Spec:   corev1.PodSpec{NodeName: "node", NodeSelector: map[string]string{"kubernetes.io/hostname": "node"}, RestartPolicy: corev1.RestartPolicyNever, Containers: []corev1.Container{{Name: api.InferenceServerContainerName, ReadinessProbe: &corev1.Probe{ProbeHandler: corev1.ProbeHandler{HTTPGet: &corev1.HTTPGetAction{Port: intstr.FromString("http")}}}, Ports: []corev1.ContainerPort{{Name: "http", ContainerPort: 8000}}}}},
		Status: corev1.PodStatus{PodIP: "127.0.0.1", Phase: corev1.PodRunning, ContainerStatuses: []corev1.ContainerStatus{{Name: api.InferenceServerContainerName, ContainerID: "containerd://first", State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{StartedAt: metav1.Now()}}}}},
	}
}

func testEngineController(t *testing.T, pods ...*corev1.Pod) (*controller, *fake.Clientset, cache.Indexer) {
	t.Helper()
	informer := cache.NewSharedIndexInformer(&cache.ListWatch{}, &corev1.Pod{}, 0, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc, requesterIndexName: requesterIndexFunc, nominalHashIndexName: nominalHashIndexFunc})
	index := informer.GetIndexer()
	objects := make([]runtime.Object, 0, len(pods))
	for _, pod := range pods {
		objects = append(objects, pod.DeepCopy())
		if err := index.Add(pod.DeepCopy()); err != nil {
			t.Fatal(err)
		}
	}
	client := fake.NewClientset(objects...)
	ctl := &controller{namespace: "test", podInformer: informer, coreclient: client.CoreV1(), podLister: corelisters.NewPodLister(index), nodeNameToData: make(map[string]*nodeData),
		httpLatencySecsHistograms: prometheus.NewHistogramVec(prometheus.HistogramOpts{Name: "test_engine_http"}, []string{"purpose", "method", "isc_name", "status_code"})}
	return ctl, client, index
}

func testServerAddress(t *testing.T, url string) (string, int32) {
	t.Helper()
	host, port, err := net.SplitHostPort(url[len("http://"):])
	if err != nil {
		t.Fatal(err)
	}
	n, err := strconv.Atoi(port)
	if err != nil {
		t.Fatal(err)
	}
	return host, int32(n)
}

func TestSGLangLifecycleAcrossRequesters(t *testing.T) {
	var paths []string
	var mu sync.Mutex
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Tags []string `json:"tags"`
		}
		if r.Method != http.MethodPost || r.Header.Get("Content-Type") != "application/json" {
			t.Error("incorrect request method or content type")
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if !slices.Equal(body.Tags, []string{"kv_cache", "cuda_graph", "weights"}) {
			t.Errorf("tags = %v", body.Tags)
		}
		mu.Lock()
		paths = append(paths, r.URL.Path)
		mu.Unlock()
		fmt.Fprint(w, "null")
	}))
	defer server.Close()
	pod := testSGLangPod()
	ip, port := testServerAddress(t, server.URL)
	pod.Status.PodIP = ip
	pod.Spec.Containers[0].Ports[0].ContainerPort = port
	ctl, client, _ := testEngineController(t, pod)
	nd := ctl.getNodeData("node")
	nd.rememberNewSGLang(pod)
	ctx := context.Background()
	// A freshly created process is awake; neither query nor wake calls resume.
	for range 2 {
		if asleep, err := ctl.querySleeping(ctx, "", pod, port); err != nil || asleep {
			t.Fatalf("fresh state = %v, %v", asleep, err)
		}
		if err := ctl.wakeUp(ctx, &serverData{}, nil, pod, port, "new", nil); err != nil {
			t.Fatal(err)
		}
	}
	oldRequesterState := &serverData{ProvidingPodName: pod.Name}
	for range 2 {
		if err := ctl.ensureUnbound(ctx, oldRequesterState, "", nd, pod, false); err != nil {
			t.Fatal(err)
		}
	}
	unbound, err := client.CoreV1().Pods("test").Get(ctx, pod.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if unbound.Annotations[requesterAnnotationKey] != "" || unbound.Labels[api.SleepingLabelName] != "true" || !slices.Contains(unbound.Finalizers, providerFinalizer) {
		t.Fatalf("unsafe unbind: %+v", unbound.ObjectMeta)
	}
	// Discarding requester state must retain Provider knowledge.
	newState := &serverData{}
	req := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "replacement", Namespace: "test", UID: "replacement-uid"}, Spec: corev1.PodSpec{NodeName: "node"}}
	if _, err := client.CoreV1().Pods("test").Create(ctx, req, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	if out := ctl.bind(ctx, newState, req, unbound, nil, false); out.err != nil {
		t.Fatal(out.err)
	}
	for range 2 {
		if err := ctl.wakeUp(ctx, newState, req, pod, port, "retry", nil); err != nil {
			t.Fatal(err)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if !slices.Equal(paths, []string{"/release_memory_occupation", "/resume_memory_occupation"}) {
		t.Fatalf("unexpected lifecycle calls: %v", paths)
	}
}

func TestSGLangUncertainOperationIsNotRetried(t *testing.T) {
	for _, sleeping := range []bool{true, false} {
		for _, mode := range []string{"http-error", "lost-response", "timeout", "truncated-body", "redirect"} {
			t.Run(fmt.Sprintf("%s/sleep=%v", mode, sleeping), func(t *testing.T) {
				var mu sync.Mutex
				calls := 0
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					mu.Lock()
					calls++
					mu.Unlock()
					if !sleeping && r.URL.Path == "/release_memory_occupation" {
						fmt.Fprint(w, "null")
						return
					}
					switch mode {
					case "http-error":
						http.Error(w, "busy or partially failed", 500)
					case "lost-response":
						conn, _, _ := w.(http.Hijacker).Hijack()
						_ = conn.Close()
					case "timeout":
						time.Sleep(200 * time.Millisecond)
					case "truncated-body":
						w.Header().Set("Content-Length", "10")
						fmt.Fprint(w, "x")
					case "redirect":
						w.Header().Set("Location", "/other")
						w.WriteHeader(http.StatusTemporaryRedirect)
					}
				}))
				defer server.Close()
				pod := testSGLangPod()
				ip, port := testServerAddress(t, server.URL)
				pod.Status.PodIP = ip
				pod.Annotations[api.EngineOperationTimeoutAnnotationName] = "100ms"
				ctl, _, _ := testEngineController(t, pod)
				ctl.getNodeData("node").rememberNewSGLang(pod)
				if !sleeping {
					if err := ctl.setSGLangSleeping(context.Background(), pod, port, true); err != nil {
						t.Fatal(err)
					}
				}
				for range 2 {
					if err := ctl.setSGLangSleeping(context.Background(), pod, port, sleeping); err == nil {
						t.Fatal("uncertain operation accepted")
					}
					if err := ctl.setSGLangSleeping(context.Background(), pod, port, !sleeping); err == nil {
						t.Fatal("uncertain process resumed")
					}
				}
				mu.Lock()
				defer mu.Unlock()
				wantCalls := 1
				if !sleeping {
					wantCalls = 2
				}
				if calls != wantCalls {
					t.Fatalf("calls = %d, want %d", calls, wantCalls)
				}
			})
		}
	}
}

func TestSGLangOperationDeadline(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { time.Sleep(60 * time.Millisecond); fmt.Fprint(w, "null") }))
	defer server.Close()
	pod := testSGLangPod()
	ip, port := testServerAddress(t, server.URL)
	pod.Status.PodIP = ip
	pod.Annotations[api.EngineOperationTimeoutAnnotationName] = "1s"
	ctl, _, _ := testEngineController(t, pod)
	ctl.getNodeData("node").rememberNewSGLang(pod)
	// Prove the shared client deadline is not accidentally applied to SGLang.
	original := myHTTPClient
	myHTTPClient = &http.Client{Timeout: time.Millisecond}
	defer func() { myHTTPClient = original }()
	if err := ctl.setSGLangSleeping(context.Background(), pod, port, true); err != nil {
		t.Fatal(err)
	}
}

func TestSGLangRecoveryWaitsForTermination(t *testing.T) {
	for _, reason := range []string{"controller-restart", "container-restart", "pod-replacement"} {
		t.Run(reason, func(t *testing.T) {
			pod := testSGLangPod()
			ctl, client, index := testEngineController(t, pod)
			nd := ctl.getNodeData("node")
			if reason != "controller-restart" {
				nd.rememberNewSGLang(pod)
				nd.sglangKnowledge(pod)
				if reason == "container-restart" {
					pod.Status.ContainerStatuses[0].RestartCount++
					pod.Status.ContainerStatuses[0].ContainerID = "containerd://second"
				} else {
					pod.UID = "new-uid"
				}
				if err := index.Update(pod); err != nil {
					t.Fatal(err)
				}
			}
			// API delete acceptance is insufficient; keep finalizer while running.
			if err := ctl.guardSGLangProviders(context.Background(), nd); err == nil {
				t.Fatal("unknown instance not fenced")
			}
			pod.DeletionTimestamp = ptr.To(metav1.Now())
			if err := index.Update(pod); err != nil {
				t.Fatal(err)
			}
			if err := ctl.guardSGLangProviders(context.Background(), nd); err == nil {
				t.Fatal("terminating instance not fenced")
			}
			if nd.SGLang[pod.UID].stopped {
				t.Fatal("running process marked stopped")
			}
			for _, action := range client.Actions() {
				if update, ok := action.(k8stesting.UpdateAction); ok {
					updated := update.GetObject().(*corev1.Pod)
					if updated.Name == pod.Name && !slices.Contains(updated.Finalizers, providerFinalizer) {
						t.Fatal("provider finalizer removed before termination")
					}
				}
			}
			pod.Status.Phase = corev1.PodFailed
			pod.Status.ContainerStatuses[0].State = corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{ExitCode: 1}}
			if err := index.Update(pod); err != nil {
				t.Fatal(err)
			}
			// Fake client deletes immediately, so reinstall the finalizer-bearing
			// object to model Kubernetes retaining it until kubelet termination.
			if _, err := ctl.coreclient.Pods("test").Create(context.Background(), pod, metav1.CreateOptions{}); err != nil {
				t.Fatal(err)
			}
			if err := ctl.guardSGLangProviders(context.Background(), nd); err == nil {
				t.Fatal("must wait for informer removal")
			}
			if !nd.SGLang[pod.UID].stopped {
				t.Fatal("terminal process not confirmed")
			}
			terminal, err := client.CoreV1().Pods("test").Get(context.Background(), pod.Name, metav1.GetOptions{})
			if err != nil || slices.Contains(terminal.Finalizers, providerFinalizer) {
				t.Fatalf("terminal provider finalizer not removed: %v", err)
			}
			if err := index.Delete(pod); err != nil {
				t.Fatal(err)
			}
			err = ctl.guardSGLangProviders(context.Background(), nd)
			if reason == "pod-replacement" {
				if err == nil {
					t.Fatal("old UID disappeared without proof: should remain fenced")
				}
			} else if err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestSGLangExample(t *testing.T) {
	content, err := os.ReadFile("../../../docs/examples/sglang-direct-provider.yaml")
	if err != nil {
		t.Fatal(err)
	}
	documents := strings.SplitN(string(content), "\n---\n", 2)
	if len(documents) != 2 {
		t.Fatal("expected ReplicaSet and Service")
	}
	var example appsv1.ReplicaSet
	if err := yaml.UnmarshalStrict([]byte(documents[0]), &example); err != nil {
		t.Fatal(err)
	}
	var service corev1.Service
	if err := yaml.UnmarshalStrict([]byte(documents[1]), &service); err != nil {
		t.Fatal(err)
	}
	req := &corev1.Pod{TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Pod"}, ObjectMeta: example.Spec.Template.ObjectMeta, Spec: example.Spec.Template.Spec}
	req.Spec.NodeName = "node"
	data := &serverData{GPUIndicesStr: ptr.To("0"), GPUIDsStr: ptr.To("GPU-1")}
	pod, _, err := data.getNominalServerProvidingPod(context.Background(), req, req.Annotations[api.ServerPatchAnnotationName], api.ProviderData{NodeName: "node"})
	if err != nil {
		t.Fatal(err)
	}
	if !isSGLang(pod) || pod.Labels["app"] == "sglang-requester" || data.ServerPort != 8000 {
		t.Fatalf("example Provider incorrectly configured: %+v", pod)
	}
	if service.Spec.Selector["app"] != req.Labels["app"] || service.Spec.Ports[0].TargetPort.IntVal != 8082 {
		t.Fatal("Service does not target requester proxy")
	}
}

func TestSGLangCancelledBeforeScheduling(t *testing.T) {
	pod := testSGLangPod()
	pod.Spec.NodeName = ""
	pod.Status = corev1.PodStatus{Phase: corev1.PodPending}
	pod.DeletionTimestamp = ptr.To(metav1.Now())
	ctl, client, index := testEngineController(t, pod)
	nd := ctl.getNodeData("node")
	if err := ctl.guardSGLangProviders(context.Background(), nd); err == nil {
		t.Fatal("must wait for removal")
	}
	updated, err := client.CoreV1().Pods("test").Get(context.Background(), pod.Name, metav1.GetOptions{})
	if err != nil || slices.Contains(updated.Finalizers, providerFinalizer) {
		t.Fatalf("cancelled startup finalizer stuck: %v", err)
	}
	if err := index.Delete(pod); err != nil {
		t.Fatal(err)
	}
	if err := ctl.guardSGLangProviders(context.Background(), nd); err != nil {
		t.Fatal(err)
	}
}

func TestDirectEngineConfigurationAndHash(t *testing.T) {
	ctx := context.Background()
	build := func(engine string) (*corev1.Pod, string, error) {
		req := testSGLangPod()
		req.Annotations = map[string]string{api.EngineAnnotationName: engine}
		d := &serverData{GPUIndicesStr: ptr.To("0"), GPUIDsStr: ptr.To("GPU-1")}
		return d.getNominalServerProvidingPod(ctx, req, "{}", api.ProviderData{NodeName: "node"})
	}
	_, defaultHash, err := build("")
	if err != nil {
		t.Fatal(err)
	}
	vllm, vllmHash, err := build("vllm")
	if err != nil {
		t.Fatal(err)
	}
	sglang, sglangHash, err := build("sglang")
	if err != nil {
		t.Fatal(err)
	}
	if defaultHash != vllmHash || vllmHash == sglangHash {
		t.Fatal("engine reuse hash is incorrect")
	}
	if vllm.Annotations[api.EngineAnnotationName] != "vllm" || !isSGLang(sglang) || sglang.Spec.RestartPolicy != corev1.RestartPolicyNever {
		t.Fatal("engine metadata/restart policy missing")
	}
	if _, _, err := build("invalid"); err == nil {
		t.Fatal("invalid engine accepted")
	}
	pod := testSGLangPod()
	pod.Annotations[api.InferenceServerConfigAnnotationName] = "launcher"
	if _, _, err := directEngineConfig(pod); err == nil {
		t.Fatal("SGLang launcher accepted")
	}
	delete(pod.Annotations, api.InferenceServerConfigAnnotationName)
	for _, value := range []string{"0s", "-1s", "2h", "garbage"} {
		pod.Annotations[api.EngineOperationTimeoutAnnotationName] = value
		if _, _, err := directEngineConfig(pod); err == nil {
			t.Fatalf("invalid duration %q accepted", value)
		}
	}
}

func TestDefaultVLLMLifecycle(t *testing.T) {
	var paths []string
	var mu sync.Mutex
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		paths = append(paths, r.URL.Path)
		if r.URL.Path == "/is_sleeping" {
			fmt.Fprint(w, `{"is_sleeping":true}`)
		}
	}))
	defer server.Close()
	pod := testSGLangPod()
	delete(pod.Annotations, api.EngineAnnotationName)
	ip, port := testServerAddress(t, server.URL)
	pod.Status.PodIP = ip
	pod.Spec.Containers[0].Ports[0].ContainerPort = port
	ctl, _, _ := testEngineController(t, pod)
	data := &serverData{}
	if asleep, err := ctl.querySleeping(context.Background(), "", pod, port); err != nil || !asleep {
		t.Fatalf("vLLM query: %v %v", asleep, err)
	}
	if err := ctl.wakeUp(context.Background(), data, &corev1.Pod{}, pod, port, "test", nil); err != nil {
		t.Fatal(err)
	}
	if err := ctl.ensureUnbound(context.Background(), data, "", ctl.getNodeData("node"), pod, false); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if !slices.Equal(paths, []string{"/is_sleeping", "/wake_up", "/sleep"}) {
		t.Fatal(paths)
	}
}

func TestSGLangBoundReconcileAndColdRecovery(t *testing.T) {
	ctx := context.Background()
	var calls []string
	var target stubapi.ProxyTargetConfig
	var mu sync.Mutex
	spi := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		calls = append(calls, r.URL.Path)
		if r.URL.Path == stubapi.ProxyConfigPath {
			if err := json.NewDecoder(r.Body).Decode(&target); err != nil {
				t.Error(err)
			}
		}
	}))
	defer spi.Close()
	ip, port := testServerAddress(t, spi.URL)
	pod := testSGLangPod()
	pod.Status.Conditions = []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}}
	req := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "requester", Namespace: "test", UID: "req-uid", Finalizers: []string{requesterFinalizer}, Annotations: map[string]string{api.EngineAnnotationName: "sglang", api.AdminPortAnnotationName: strconv.Itoa(int(port)), api.ServerPatchAnnotationName: "{}"}}, Spec: corev1.PodSpec{NodeName: "node"}, Status: corev1.PodStatus{PodIP: ip}}
	ctl, client, index := testEngineController(t, pod, req)
	nodes := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	if err := nodes.Add(&corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node"}}); err != nil {
		t.Fatal(err)
	}
	ctl.nodeLister = corelisters.NewNodeLister(nodes)
	nd := ctl.getNodeData("node")
	nd.rememberNewSGLang(pod)
	data := ctl.getServerData(nd, req.Name, req.UID)
	data.GPUIDsStr, data.GPUIndicesStr = ptr.To("GPU-1"), ptr.To("0")
	data.GPUIDs = []string{"GPU-1"}
	item := infSvrItem{UID: req.UID, RequesterName: req.Name}
	for range 2 {
		if out := item.process(ctx, ctl, nd); out.err != nil {
			t.Fatal(out.err)
		}
	}
	mu.Lock()
	initialCalls, initialTarget := slices.Clone(calls), target
	mu.Unlock()
	if !slices.Equal(initialCalls, []string{stubapi.ProxyConfigPath, stubapi.BecomeReadyPath}) {
		t.Fatalf("fresh reconcile SPI calls: %v", calls)
	}
	if initialTarget.Address != pod.Status.PodIP || initialTarget.Port != 8000 {
		t.Fatalf("wrong proxy target: %+v", initialTarget)
	}
	// Losing controller knowledge must revoke readiness and delete both Pods;
	// no replacement is created until the old instance is confirmed stopped.
	nd.SGLang = nil
	for range 2 {
		if out := item.process(ctx, ctl, nd); out.err == nil || !out.retry {
			t.Fatalf("unknown instance not blocked: %+v", out)
		}
		// Model the API's terminating state arriving in the informer.
		pod.DeletionTimestamp, req.DeletionTimestamp = ptr.To(metav1.Now()), ptr.To(metav1.Now())
		if err := index.Update(pod); err != nil {
			t.Fatal(err)
		}
		if err := index.Update(req); err != nil {
			t.Fatal(err)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if calls[len(calls)-1] != stubapi.BecomeUnreadyPath {
		t.Fatalf("readiness not revoked: %v", calls)
	}
	for _, action := range client.Actions() {
		if action.GetVerb() == "create" {
			t.Fatalf("created replacement before termination: %v", action)
		}
	}
}

func TestSGLangNodeFenceBlocksNewRequester(t *testing.T) {
	// This fixture constructs controllers without NewController, which normally
	// initializes the metric reached when a running requester first becomes ready.
	originalMetric := actuationSecsHistograms
	actuationSecsHistograms = kubemetrics.NewHistogramVec(&kubemetrics.HistogramOpts{
		Name: "test_actuation_seconds", Help: "Test requester actuation latency",
	}, []string{"exported_namespace", "path", "instancesDeleted", "isc_name"})
	t.Cleanup(func() { actuationSecsHistograms = originalMetric })
	for _, mode := range []string{"create", "resume"} {
		t.Run(mode, func(t *testing.T) {
			ctx := context.Background()
			var mu sync.Mutex
			var calls []string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				mu.Lock()
				calls = append(calls, r.Method+" "+r.URL.Path)
				mu.Unlock()
				fmt.Fprint(w, "null")
			}))
			defer server.Close()
			ip, port := testServerAddress(t, server.URL)
			callSnapshot := func() []string {
				mu.Lock()
				defer mu.Unlock()
				return slices.Clone(calls)
			}

			old := testSGLangPod()
			old.Name, old.UID = "provider-a", "provider-a-uid"
			old.Annotations[requesterAnnotationKey] = "old-requester-uid old-requester"
			req := testSGLangPod()
			req.Name, req.UID = "requester-b", "requester-b-uid"
			req.Finalizers = nil
			req.Annotations = map[string]string{
				api.EngineAnnotationName: "sglang", api.ServerPatchAnnotationName: "{}",
				api.AdminPortAnnotationName: strconv.Itoa(int(port)),
			}
			req.Status.PodIP = ip
			req.Spec.Containers[0].Ports[0].ContainerPort = port
			pods := []*corev1.Pod{old, req}
			var provider *corev1.Pod
			if mode == "resume" {
				provider = testSGLangPod()
				provider.Name, provider.UID = "provider-b", "provider-b-uid"
				provider.Annotations[requesterAnnotationKey] = string(req.UID) + " " + req.Name
				provider.Status.PodIP = ip
				provider.Spec.Containers[0].Ports[0].ContainerPort = port
				provider.Status.Conditions = []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}}
				pods = append(pods, provider)
			}
			ctl, client, index := testEngineController(t, pods...)
			nodes := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
			if err := nodes.Add(&corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node"}}); err != nil {
				t.Fatal(err)
			}
			ctl.nodeLister = corelisters.NewNodeLister(nodes)
			nd := ctl.getNodeData("node")
			data := ctl.getServerData(nd, req.Name, req.UID)
			data.GPUIDs, data.GPUIDsStr, data.GPUIndicesStr = []string{"GPU-1"}, ptr.To("GPU-1"), ptr.To("0")
			item := infSvrItem{UID: req.UID, RequesterName: req.Name}
			if provider != nil {
				nd.rememberNewSGLang(provider)
				if err := ctl.setSGLangSleeping(ctx, provider, port, true); err != nil {
					t.Fatal(err)
				}
			}
			baseline := callSnapshot()
			resource := corev1.SchemeGroupVersion.WithResource("pods")
			// The fake API normally deletes immediately and does not generate names.
			// Model grace/finalizer retention and API-assigned identity explicitly;
			// informer observations remain separate from API state.
			client.PrependReactor("delete", "pods", func(action k8stesting.Action) (bool, runtime.Object, error) {
				if action.(k8stesting.DeleteAction).GetName() != old.Name {
					return false, nil, nil
				}
				object, err := client.Tracker().Get(resource, old.Namespace, old.Name)
				if err != nil {
					return true, nil, err
				}
				terminating := object.(*corev1.Pod).DeepCopy()
				terminating.DeletionTimestamp = ptr.To(metav1.Now())
				return true, nil, client.Tracker().Update(resource, terminating, old.Namespace)
			})
			client.PrependReactor("create", "pods", func(action k8stesting.Action) (bool, runtime.Object, error) {
				created := action.(k8stesting.CreateAction).GetObject().(*corev1.Pod).DeepCopy()
				if created.Name == "" {
					created.Name, created.UID = "provider-b", "provider-b-uid"
				}
				if err := client.Tracker().Create(resource, created, created.Namespace); err != nil {
					return true, nil, err
				}
				return true, created, nil
			})
			createdCount := func() int {
				count := 0
				for _, action := range client.Actions() {
					if action.GetVerb() == "create" && action.GetResource().Resource == "pods" {
						count++
					}
				}
				return count
			}
			checkBlocked := func(stage string) {
				t.Helper()
				out := item.process(ctx, ctl, nd)
				actual := callSnapshot()
				t.Logf("stage=%s old_uid=%s phase=%s state=%+v requester_uid=%s node=%s gpus=%v creates=%d http_delta=%v retry=%v err=%v",
					stage, old.UID, old.Status.Phase, old.Status.ContainerStatuses[0].State,
					req.UID, req.Spec.NodeName, data.GPUIDs, createdCount(), actual[len(baseline):], out.retry, out.err)
				if out.err == nil || !out.retry {
					t.Fatalf("new requester not fenced at %s: %+v", stage, out)
				}
				if createdCount() != 0 || !slices.Equal(actual, baseline) {
					t.Fatalf("actuation before old provider removal at %s: creates=%d calls=%v", stage, createdCount(), actual)
				}
			}

			checkBlocked("unknown-running")
			var err error
			old, err = client.CoreV1().Pods("test").Get(ctx, old.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if err := index.Update(old.DeepCopy()); err != nil {
				t.Fatal(err)
			}
			checkBlocked("deleting-running")
			retained, err := client.CoreV1().Pods("test").Get(ctx, old.Name, metav1.GetOptions{})
			if err != nil || !slices.Contains(retained.Finalizers, providerFinalizer) || nd.SGLang[old.UID].stopped {
				t.Fatalf("running provider lost termination fence: pod=%+v err=%v", retained, err)
			}
			old.Status.Phase = corev1.PodSucceeded
			old.Status.ContainerStatuses[0].State = corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{ExitCode: 0}}
			if _, err := client.CoreV1().Pods("test").Update(ctx, old, metav1.UpdateOptions{}); err != nil {
				t.Fatal(err)
			}
			if err := index.Update(old.DeepCopy()); err != nil {
				t.Fatal(err)
			}
			checkBlocked("terminal-in-informer")
			retained, err = client.CoreV1().Pods("test").Get(ctx, old.Name, metav1.GetOptions{})
			if err != nil || slices.Contains(retained.Finalizers, providerFinalizer) || !nd.SGLang[old.UID].stopped {
				t.Fatalf("termination not confirmed before removal: pod=%+v err=%v", retained, err)
			}
			if err := client.Tracker().Delete(resource, old.Namespace, old.Name); err != nil {
				t.Fatal(err)
			}
			if err := index.Delete(old); err != nil {
				t.Fatal(err)
			}
			if out := item.process(ctx, ctl, nd); out.err != nil {
				t.Fatalf("actuation not allowed after removal: %+v", out)
			}
			if mode == "create" {
				provider, err = client.CoreV1().Pods("test").Get(ctx, "provider-b", metav1.GetOptions{})
				if err != nil {
					t.Fatal(err)
				}
				provider.Status = req.Status
				provider.Status.Conditions = []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}}
				if _, err := client.CoreV1().Pods("test").Update(ctx, provider, metav1.UpdateOptions{}); err != nil {
					t.Fatal(err)
				}
			}
			// Reflect normal API updates before repeat reconciliation, including
			// the created Provider or the resumed Provider's sleeping label.
			current, err := client.CoreV1().Pods("test").List(ctx, metav1.ListOptions{})
			if err != nil {
				t.Fatal(err)
			}
			for i := range current.Items {
				if err := index.Update(current.Items[i].DeepCopy()); err != nil {
					t.Fatal(err)
				}
			}
			for range 2 {
				if out := item.process(ctx, ctl, nd); out.err != nil {
					t.Fatal(out.err)
				}
			}
			want := []string{"PUT " + stubapi.ProxyConfigPath, "POST " + stubapi.BecomeReadyPath}
			wantCreates := 1
			if mode == "resume" {
				want = append([]string{"POST /resume_memory_occupation"}, want...)
				wantCreates = 0
			}
			actual := callSnapshot()[len(baseline):]
			t.Logf("stage=removed-and-reconciled requester_uid=%s provider_uid=%s creates=%d http_delta=%v", req.UID, provider.UID, createdCount(), actual)
			if createdCount() != wantCreates || !slices.Equal(actual, want) {
				t.Fatalf("wrong positive control or duplicate actuation: creates=%d want=%d calls=%v want=%v", createdCount(), wantCreates, actual, want)
			}
		})
	}
}
