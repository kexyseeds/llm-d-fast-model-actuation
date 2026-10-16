# SGLang Direct Providers

The Direct Provider controller can manage upstream SGLang without an engine
patch. Set `dual-pods.llm-d.ai/engine: sglang` on the requester Pod template.
Omitting the annotation selects the existing vLLM protocol. Unsupported values
and SGLang with `inference-server-config` (Launcher mode) are rejected.

The initial contract targets **SGLang v0.5.21**, source commit
`e00930c5489053f26d86b179cee0d087f846acbb`. HTTP/controller tests use a simulated
engine. Real GPU memory release, model output after restore, CUDA Graph behavior,
and FMA/SGLang end-to-end operation still require GPU integration validation.

## Configuration

Use [the example ReplicaSet and Service](examples/sglang-direct-provider.yaml)
with the usual [Direct Provider controller setup](../cmd/requester/README.md),
including the GPU map and GPU device access for providers. Replace the requester
image placeholder with an image built from this FMA checkout. The SGLang image
is version pinned; resolve and pin its digest for the target GPU architecture
before a reproducible deployment. Image availability and GPU compatibility are
not established by the Go tests.

The example enables `--enable-memory-saver`, `--enable-weights-cpu-backup`, and
`SGLANG_MEMORY_SAVER_CUDA_GRAPH=1` for the relevant Full CUDA Graph backend.
Ensure the chosen backend honors these settings. HTTP success alone does not
prove memory saver is active or that enough HBM was released. Budget host RAM
for CPU weight backup, plus runtime overhead; the example's memory and CPU
limits are illustrative and must be sized for your model and GPU environment.

The controller copies engine identity and operation timeout to the Provider.
Both participate in SGLang reuse matching. Configure them on the requester,
not inside `server-patch`; the controller owns their Provider values. Replace
requesters to change their configuration rather than mutating a bound Pod.

`dual-pods.llm-d.ai/engine-operation-timeout` is a Go duration, defaults to `5m`,
and accepts values greater than zero and at most `1h`. It bounds each SGLang
release/resume request independently of the existing vLLM client's 30-second
timeout. Expiry means **unknown outcome**, not cancellation of engine work.

Service labels in the example select only requesters. The server patch removes
the requester label from Providers. Traffic follows Service → requester TCP
proxy → SGLang. Never select sleeping Providers directly with that Service.

## Lifecycle and recovery

| Action | SGLang request |
| --- | --- |
| Fresh process | No resume; use normal Provider readiness |
| Release on unbind | `POST /release_memory_occupation` |
| Resume on reuse | `POST /resume_memory_occupation` |

Both requests send `{"tags":["kv_cache","cuda_graph","weights"]}`. FMA records
completion only after a successful HTTP response, including consuming its body.
It does not call `/is_sleeping`, `/suspend`, `/resume`, or `/is_suspended` on
SGLang. This is not a checkpoint implementation or an L1/L2 policy API.

State belongs to the Provider UID and container ID/start time/restart count,
and survives deletion of the old requester in controller memory. Repeated
reconciliation does not repeat a confirmed operation. Sleeping labels are
informational, **not an authoritative recovery journal**. FMA must be the sole
lifecycle operator; do not call engine release/resume manually or supervise and
restart the engine inside its container independently of Kubernetes.

For SGLang, the controller forces Provider `restartPolicy: Never`: restarting a
released process could allocate memory that now belongs to another model.
Engine process exit is handled by cold recovery. Normal release/reuse keeps
the same Provider and process.

Timeouts, connection errors, non-success HTTP responses, changed incarnation,
and loss of controller knowledge trigger conservative cold recovery. This also
means a controller restart cold-rebuilds its SGLang Providers, including sleepers.
The controller makes a bound requester unready and deletes it and its Provider.
The requester must be owned by a ReplicaSet/Deployment (as in the example) to
be recreated automatically; its proxy cannot be retargeted in place.

An uncertain or terminating SGLang Provider **blocks new actuation across its
node**, including vLLM actuation. This first implementation deliberately uses a
node-wide fence. Existing healthy serving instances on that node are not shut
down by this fence. A Provider finalizer is retained even while sleeping. The
controller waits for terminal Pod phase and terminated container statuses before
removing it, then waits for the informer to observe removal. Delete acceptance
alone is insufficient. A never-scheduled Provider can be removed once its
deletion timestamp is observed, since Kubernetes rejects binding a deleting
Pod and no kubelet could have started it. A lost node, missing termination evidence, or forced
removal of the finalizer can leave actuation blocked; restore node evidence or
perform infrastructure-level fencing before operator recovery. Do not strip
these finalizers to force GPU handoff.

This adapter reuses FMA's existing scale-down policy. It does not drain requests,
pause generation, abort/retract work, migrate requests, or preserve streaming
connections. SGLang release requires an idle engine. Start validation with idle
instances. Busy/partial failures are treated as unknown; they can cause a cold
rebuild and interrupted requests. Readiness revocation removes normal Service
routing eligibility, but does not promise to drain existing TCP connections.

## Basic verification

```sh
go test ./pkg/controller/dual-pods -count=1
go test ./pkg/...
go build ./cmd/...
```

These tests exercise real HTTP requests and fake Kubernetes objects: engine
selection, reuse separation, fresh startup, release/reuse across requesters,
duplicate reconciliation, deadlines and lost responses, proxy/readiness
configuration, termination fencing, and unchanged vLLM endpoints. They do not
create a kind cluster or validate GPU memory, Kubernetes scheduling, or an actual
SGLang server.
