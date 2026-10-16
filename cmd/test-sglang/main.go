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

// test-sglang is a test fixture that mimics the SGLang upstream lifecycle
// contract used by the dual-pods controller: POST /release_memory_occupation
// and POST /resume_memory_occupation with a strict tags body. It exists only
// for control-flow testing on fake GPUs; success here says nothing about real
// SGLang behavior. The /test/* endpoints are for the human tester and must
// never be used by the controller.
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"slices"
	"strconv"
	"sync"
	"syscall"
	"time"
)

const (
	stateAwake    = "awake"
	stateReleased = "released"

	faultApplyThenDisconnect = "apply_then_disconnect"
)

// wantTags is the exact body the controller sends; the fixture rejects
// anything else so a contract drift cannot pass silently.
var wantTags = []string{"kv_cache", "cuda_graph", "weights"}

type recordEvent struct {
	Seq       int      `json:"seq"`
	At        string   `json:"at"`
	Op        string   `json:"op"`
	Tags      []string `json:"tags"`
	Model     string   `json:"model"`
	PodUID    string   `json:"pod_uid"`
	ProcessID int      `json:"process_id"`
	Result    string   `json:"result"`
}

type faultConfig struct {
	NextRelease        string `json:"next_release"`
	TerminationDelayMS int64  `json:"termination_delay_ms"`
}

type fixture struct {
	mu       sync.Mutex
	model    string
	podUID   string
	pid      int
	state    string
	events   []recordEvent
	nextRele string
	sigtermD time.Duration

	logMu  sync.Mutex
	logOut io.Writer
}

func newFixture(model, podUID string, logOut io.Writer) *fixture {
	return &fixture{
		model:  model,
		podUID: podUID,
		pid:    os.Getpid(),
		state:  stateAwake,
		logOut: logOut,
	}
}

func nowUTC() string {
	return time.Now().UTC().Format(time.RFC3339Nano)
}

func (f *fixture) log(fields map[string]any) {
	line := map[string]any{
		"at":         nowUTC(),
		"model":      f.model,
		"pod_uid":    f.podUID,
		"process_id": f.pid,
	}
	for k, v := range fields {
		line[k] = v
	}
	b, err := json.Marshal(line)
	if err != nil {
		b = []byte(fmt.Sprintf(`{"at":%q,"event":"log_error","detail":%q}`, nowUTC(), err.Error()))
	}
	f.logMu.Lock()
	defer f.logMu.Unlock()
	_, _ = f.logOut.Write(append(b, '\n'))
}

// appendEventLocked records a lifecycle event and mirrors it to the JSON log.
// Callers must hold f.mu.
func (f *fixture) appendEventLocked(op, result string) {
	f.events = append(f.events, recordEvent{
		Seq:       len(f.events) + 1,
		At:        nowUTC(),
		Op:        op,
		Tags:      slices.Clone(wantTags),
		Model:     f.model,
		PodUID:    f.podUID,
		ProcessID: f.pid,
		Result:    result,
	})
}

func (f *fixture) rejected(op, detail string) {
	f.log(map[string]any{"event": "rejected", "op": op, "detail": detail})
}

func (f *fixture) mux() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprintln(w, "OK")
	})
	mux.HandleFunc("POST /release_memory_occupation", f.lifecycle("release"))
	mux.HandleFunc("POST /resume_memory_occupation", f.lifecycle("resume"))
	mux.HandleFunc("GET /test/infer", f.handleInfer)
	mux.HandleFunc("GET /test/records", f.handleRecords)
	mux.HandleFunc("POST /test/fault", f.handleFault)
	return mux
}

func decodeTags(r *http.Request) ([]string, error) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 8<<10))
	if err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.DisallowUnknownFields()
	var req struct {
		Tags []string `json:"tags"`
	}
	if err := dec.Decode(&req); err != nil {
		return nil, fmt.Errorf("decode body: %w", err)
	}
	if dec.More() {
		return nil, errors.New("trailing data after JSON object")
	}
	return req.Tags, nil
}

func (f *fixture) lifecycle(op string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		tags, err := decodeTags(r)
		if err != nil || !slices.Equal(tags, wantTags) {
			f.rejected(op, fmt.Sprintf("body must be exactly {\"tags\":%q}; got err=%v tags=%q", wantTags, err, tags))
			http.Error(w, fmt.Sprintf("strict body {\"tags\":%q} required", wantTags), http.StatusBadRequest)
			return
		}
		f.mu.Lock()
		want := map[string]string{"release": stateAwake, "resume": stateReleased}[op]
		if f.state != want {
			state := f.state
			f.mu.Unlock()
			f.rejected(op, fmt.Sprintf("state is %s; %s requires %s", state, op, want))
			http.Error(w, fmt.Sprintf("state is %s; %s requires %s", state, op, want), http.StatusConflict)
			return
		}
		f.appendEventLocked(op, "received")
		if op == "release" {
			f.state = stateReleased
		} else {
			f.state = stateAwake
		}
		f.appendEventLocked(op, "completed")
		disconnect := op == "release" && f.nextRele == faultApplyThenDisconnect
		if disconnect {
			f.nextRele = ""
			f.appendEventLocked(op, "response_lost")
		}
		f.mu.Unlock()
		f.log(map[string]any{"event": "call", "op": op, "result": "received", "tags": wantTags})
		f.log(map[string]any{"event": "call", "op": op, "result": "completed"})
		if disconnect {
			f.log(map[string]any{"event": "call", "op": op, "result": "response_lost"})
			// Apply-then-disconnect: the operation took effect, but the HTTP
			// response is deliberately lost. ErrAbortHandler makes net/http
			// close the connection without writing anything.
			panic(http.ErrAbortHandler)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("null"))
	}
}

func (f *fixture) handleInfer(w http.ResponseWriter, _ *http.Request) {
	f.mu.Lock()
	state := f.state
	f.mu.Unlock()
	f.log(map[string]any{"event": "infer", "state": state})
	w.Header().Set("Content-Type", "application/json")
	if state == stateAwake {
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model":      f.model,
			"pod_uid":    f.podUID,
			"process_id": f.pid,
		})
		return
	}
	w.WriteHeader(http.StatusServiceUnavailable)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error":      "memory released; no business result",
		"model":      f.model,
		"pod_uid":    f.podUID,
		"process_id": f.pid,
	})
}

func (f *fixture) handleRecords(w http.ResponseWriter, _ *http.Request) {
	f.mu.Lock()
	events := slices.Clone(f.events)
	f.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(map[string]any{"events": events})
}

func (f *fixture) handleFault(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 8<<10))
	if err != nil {
		http.Error(w, "read body: "+err.Error(), http.StatusBadRequest)
		return
	}
	var req faultConfig
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil || dec.More() {
		http.Error(w, `body must be {"next_release":"apply_then_disconnect","termination_delay_ms":30000}`, http.StatusBadRequest)
		return
	}
	if req.NextRelease != "" && req.NextRelease != faultApplyThenDisconnect {
		http.Error(w, "unsupported next_release: "+req.NextRelease, http.StatusBadRequest)
		return
	}
	if req.TerminationDelayMS < 0 {
		http.Error(w, "termination_delay_ms must be >= 0", http.StatusBadRequest)
		return
	}
	f.mu.Lock()
	if req.NextRelease != "" {
		f.nextRele = req.NextRelease
	}
	if req.TerminationDelayMS > 0 {
		f.sigtermD = time.Duration(req.TerminationDelayMS) * time.Millisecond
	}
	effective := faultConfig{NextRelease: f.nextRele, TerminationDelayMS: int64(f.sigtermD.Milliseconds())}
	f.mu.Unlock()
	f.log(map[string]any{"event": "fault_configured", "next_release": effective.NextRelease, "termination_delay_ms": effective.TerminationDelayMS})
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(effective)
}

// handleShutdown delays exit after SIGTERM so the deletion window keeps the
// container Running and observable; HTTP keeps serving during the delay.
func (f *fixture) handleShutdown() {
	f.mu.Lock()
	delay := f.sigtermD
	f.mu.Unlock()
	f.log(map[string]any{"event": "sigterm", "termination_delay_ms": int64(delay.Milliseconds())})
	if delay > 0 {
		time.Sleep(delay)
	}
	f.log(map[string]any{"event": "exit"})
}

func main() {
	port := flag.Int("port", 8000, "port at which to listen for HTTP connections")
	flag.Parse()

	f := newFixture(os.Getenv("FMA_TEST_MODEL"), os.Getenv("FMA_TEST_POD_UID"), os.Stdout)
	addr := net.JoinHostPort("", strconv.Itoa(*port))
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		f.log(map[string]any{"event": "listen_error", "detail": err.Error()})
		os.Exit(1)
	}
	f.log(map[string]any{"event": "startup", "addr": listener.Addr().String()})

	server := &http.Server{Handler: f.mux()}
	serveErr := make(chan error, 1)
	go func() { serveErr <- server.Serve(listener) }()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGTERM, os.Interrupt)
	select {
	case sig := <-stop:
		f.log(map[string]any{"event": "signal", "signal": sig.String()})
		f.handleShutdown()
		os.Exit(0)
	case err := <-serveErr:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			f.log(map[string]any{"event": "serve_error", "detail": err.Error()})
			log.Fatal(err)
		}
	}
}
