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

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

const lifecycleBody = `{"tags":["kv_cache","cuda_graph","weights"]}`

func newTestServer(t *testing.T) (*httptest.Server, *fixture, *bytes.Buffer) {
	t.Helper()
	logs := &bytes.Buffer{}
	f := newFixture("model-a", "uid-test", logs)
	ts := httptest.NewServer(f.mux())
	t.Cleanup(ts.Close)
	return ts, f, logs
}

func doPost(t *testing.T, url, body string) (*http.Response, string) {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		return nil, err.Error()
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read response body: %v", err)
	}
	return resp, string(b)
}

func doGet(t *testing.T, url string) (*http.Response, string) {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		return nil, err.Error()
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read response body: %v", err)
	}
	return resp, string(b)
}

func getRecords(t *testing.T, url string) []recordEvent {
	t.Helper()
	resp, body := doGet(t, url+"/test/records")
	if resp == nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /test/records failed: resp=%v body=%q", resp, body)
	}
	var out struct {
		Events []recordEvent `json:"events"`
	}
	if err := json.Unmarshal([]byte(body), &out); err != nil {
		t.Fatalf("decode records %q: %v", body, err)
	}
	return out.Events
}

func assertEvent(t *testing.T, ev recordEvent, seq int, op, result string) {
	t.Helper()
	if ev.Seq != seq || ev.Op != op || ev.Result != result {
		t.Errorf("event %d = {seq:%d op:%s result:%s}, want {seq:%d op:%s result:%s}",
			seq, ev.Seq, ev.Op, ev.Result, seq, op, result)
	}
	for i, tag := range ev.Tags {
		if tag != wantTags[i] {
			t.Errorf("event %d tag[%d] = %q, want %q", seq, i, tag, wantTags[i])
		}
	}
	if ev.Model != "model-a" || ev.PodUID != "uid-test" || ev.ProcessID == 0 {
		t.Errorf("event %d identity = %+v", seq, ev)
	}
}

func TestHealthAndInferAwake(t *testing.T) {
	ts, _, _ := newTestServer(t)
	resp, body := doGet(t, ts.URL+"/health")
	if resp == nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("health: resp=%v body=%q", resp, body)
	}
	resp, body = doGet(t, ts.URL+"/test/infer")
	if resp == nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("infer awake: resp=%v body=%q", resp, body)
	}
	var infer map[string]any
	if err := json.Unmarshal([]byte(body), &infer); err != nil {
		t.Fatalf("decode infer %q: %v", body, err)
	}
	if infer["model"] != "model-a" || infer["pod_uid"] != "uid-test" {
		t.Errorf("infer identity = %v", infer)
	}
	if pid, ok := infer["process_id"].(float64); !ok || pid == 0 {
		t.Errorf("infer process_id missing: %v", infer)
	}
	if len(getRecords(t, ts.URL)) != 0 {
		t.Errorf("records not empty at startup")
	}
}

func TestLifecycleReleaseResumeRecords(t *testing.T) {
	ts, _, _ := newTestServer(t)
	resp, body := doPost(t, ts.URL+"/release_memory_occupation", lifecycleBody)
	if resp == nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("release: resp=%v body=%q", resp, body)
	}
	if body != "null" {
		t.Errorf("release body = %q, want null", body)
	}
	resp, body = doGet(t, ts.URL+"/test/infer")
	if resp == nil || resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("infer released must not return business result: resp=%v body=%q", resp, body)
	}
	resp, body = doPost(t, ts.URL+"/resume_memory_occupation", lifecycleBody)
	if resp == nil || resp.StatusCode != http.StatusOK || body != "null" {
		t.Fatalf("resume: resp=%v body=%q", resp, body)
	}
	resp, _ = doGet(t, ts.URL+"/test/infer")
	if resp == nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("infer after resume: resp=%v", resp)
	}
	events := getRecords(t, ts.URL)
	if len(events) != 4 {
		t.Fatalf("records = %+v, want 4 events", events)
	}
	assertEvent(t, events[0], 1, "release", "received")
	assertEvent(t, events[1], 2, "release", "completed")
	assertEvent(t, events[2], 3, "resume", "received")
	assertEvent(t, events[3], 4, "resume", "completed")
}

func TestDuplicateResumeRejected(t *testing.T) {
	ts, _, _ := newTestServer(t)
	resp, body := doPost(t, ts.URL+"/resume_memory_occupation", lifecycleBody)
	if resp == nil || resp.StatusCode != http.StatusConflict {
		t.Fatalf("duplicate resume must fail: resp=%v body=%q", resp, body)
	}
	if len(getRecords(t, ts.URL)) != 0 {
		t.Errorf("rejected call must not produce record events")
	}
	resp, _ = doGet(t, ts.URL+"/test/infer")
	if resp == nil || resp.StatusCode != http.StatusOK {
		t.Errorf("state must stay awake after rejection")
	}
}

func TestStrictTagsRejected(t *testing.T) {
	ts, _, _ := newTestServer(t)
	badBodies := []string{
		`{"tags":["kv_cache"]}`,
		`{"tags":["weights","cuda_graph","kv_cache"]}`,
		`{"tags":["kv_cache","cuda_graph","weights"],"extra":1}`,
		`{"other":["kv_cache","cuda_graph","weights"]}`,
		`not json`,
	}
	for _, body := range badBodies {
		resp, _ := doPost(t, ts.URL+"/release_memory_occupation", body)
		if resp == nil || resp.StatusCode != http.StatusBadRequest {
			t.Errorf("body %q: want 400, got resp=%v", body, resp)
		}
	}
	resp, _ := doGet(t, ts.URL+"/test/infer")
	if resp == nil || resp.StatusCode != http.StatusOK {
		t.Errorf("state must stay awake after rejected releases")
	}
	if len(getRecords(t, ts.URL)) != 0 {
		t.Errorf("rejected calls must not produce record events")
	}
}

func TestFaultApplyThenDisconnectOnce(t *testing.T) {
	ts, f, logs := newTestServer(t)
	resp, body := doPost(t, ts.URL+"/test/fault", `{"next_release":"apply_then_disconnect","termination_delay_ms":30000}`)
	if resp == nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("fault: resp=%v body=%q", resp, body)
	}
	var got faultConfig
	if err := json.Unmarshal([]byte(body), &got); err != nil {
		t.Fatalf("decode fault response %q: %v", body, err)
	}
	if got.NextRelease != faultApplyThenDisconnect || got.TerminationDelayMS != 30000 {
		t.Fatalf("fault response = %+v", got)
	}

	// The faulted release applies but the HTTP response is lost: the client
	// must observe a transport error, not a status code.
	_, err := http.Post(ts.URL+"/release_memory_occupation", "application/json", strings.NewReader(lifecycleBody))
	if err == nil {
		t.Fatalf("faulted release must lose the response, got success")
	}
	events := getRecords(t, ts.URL)
	if len(events) != 3 {
		t.Fatalf("records = %+v, want received/completed/response_lost", events)
	}
	assertEvent(t, events[0], 1, "release", "received")
	assertEvent(t, events[1], 2, "release", "completed")
	assertEvent(t, events[2], 3, "release", "response_lost")
	if !strings.Contains(logs.String(), "response_lost") {
		t.Errorf("response_lost missing from JSON logs")
	}
	f.mu.Lock()
	state := f.state
	f.mu.Unlock()
	if state != stateReleased {
		t.Errorf("faulted release must still apply, state = %s", state)
	}

	// One-shot: the next operation responds normally.
	resp, body = doPost(t, ts.URL+"/resume_memory_occupation", lifecycleBody)
	if resp == nil || resp.StatusCode != http.StatusOK || body != "null" {
		t.Fatalf("resume after fault must respond normally: resp=%v body=%q", resp, body)
	}
}

func TestSigtermDelayLogsAndWaits(t *testing.T) {
	logs := &bytes.Buffer{}
	f := newFixture("model-b", "uid-b", logs)
	f.mu.Lock()
	f.sigtermD = 100 * time.Millisecond
	f.mu.Unlock()
	start := time.Now()
	f.handleShutdown()
	elapsed := time.Since(start)
	if elapsed < 100*time.Millisecond {
		t.Errorf("shutdown took %v, want >= 100ms", elapsed)
	}
	lines := strings.Split(strings.TrimSpace(logs.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("shutdown logs = %q, want sigterm + exit lines", logs.String())
	}
	var sigterm, exit map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &sigterm); err != nil {
		t.Fatalf("decode %q: %v", lines[0], err)
	}
	if err := json.Unmarshal([]byte(lines[1]), &exit); err != nil {
		t.Fatalf("decode %q: %v", lines[1], err)
	}
	if sigterm["event"] != "sigterm" || fmt.Sprint(sigterm["termination_delay_ms"]) != "100" {
		t.Errorf("sigterm line = %v", sigterm)
	}
	if exit["event"] != "exit" {
		t.Errorf("exit line = %v", exit)
	}
	if exit["model"] != "model-b" || exit["pod_uid"] != "uid-b" {
		t.Errorf("exit line identity = %v", exit)
	}
}
