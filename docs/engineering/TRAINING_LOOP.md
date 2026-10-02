# Native training loop contracts

What the native runtime (`retro_training.cpp` and the fork's `llama-context.cpp`
and `ggml-opt.cpp`) guarantees around a training call, and what it does not.

## Cancellation

The epoch callback keeps the upstream void signature. When the caller's
progress callback returns false, the runtime calls `llama_opt_request_stop`.
The loop checks the request after each micro-batch callback, before the next
compute:

- the micro-batch that just finished stays applied;
- an incomplete accumulation period is discarded, so no partial gradient
  reaches a later call;
- remaining micro-batches and the evaluation tail are skipped;
- the metrics describe the work that ran, with `epoch_complete = false`;
- the duty-cycle limiter drops its debt rather than sleeping on the way out.

The callback scope (the thread-local progress pointer) is restored on every
exit, including exceptions.

## Failures

Preparation, allocation and compute errors reach the FFI boundary as errors
carrying the backend's diagnostic. A failed computation publishes no result and
no success callback, and the scheduler cursor returns to the last completed
step.

Two classes of failure, two outcomes:

- **Before compute** (batch or memory initialization, output reserve, graph
  build, allocation, unsupported fused-CE head): the step is abandoned at its
  update boundary - allocation, graph metadata, partial gradients and sequence
  state are dropped - and the trainer stays usable. Steps completed earlier in
  the call remain applied.
- **Backend compute** (`ggml_backend_sched_graph_compute` fails): the backend
  may already have written part of the parameters, and that cannot be rolled
  back. `ggml_opt_backend_failed` records it, and the trainer refuses every
  later call. Recreate it and restore a known checkpoint to recover.

## Shapes and counters

Dataset shapes, byte sizes and scheduler horizons are checked before the
allocation or native state change they feed. Values crossing signed native
counters (batch sizes, token counts, the optimizer iteration) are bounded
explicitly; an arithmetic overflow is an overflow error, not a truncation.

## Host buffers

Weighted and packed datasets borrow the caller's buffers for the synchronous
call; top-k batches materialize only the scalar label column the callbacks
read. The row plan computes active spans and accumulation padding once, and
throughput counts the physical tokens actually executed.

CE and sparse-label host arrays keep their capacity in the context. The packed
batch keeps one token/membership geometry and is replaced when either changes.
Dense labels are fully zeroed on every pass: a reused device allocation may
hold another tensor's data. Duplicate top-k IDs at one position contribute the
sum of their weights, on the dense and the fused paths alike.

## Packed graph topology is rebuilt on every step

`opt_packed_graph_storage` and `opt_packed_compute_ctx` hold metadata storage
that persists across steps; they are not a topology cache. Reusing the packed
topology is unsafe as things stand: `ggml_backend_sched` rewrites node sources
to scheduler-owned copy tensors, and changing the allocation can invalidate
those copies while the graph still points at them.

A safe cache needs:

- an immutable model/optimizer topology plus a disposable scheduler clone;
- a remap, at every allocation, of each graph input, CE target/weight, output,
  view, gradient and optimizer slot - externally owned parameters, optimizer
  slots and accumulation storage kept as they are, and no scheduler copy
  address ever carried into the immutable graph;
- a key covering token geometry, memberships/attention structure, target
  count, CE options, checkpoint settings, trainable identity and optimizer
  layout;
- invalidation on scoring, generation, preflight, trainable or layout changes,
  and errors.

One entry is enough to start with. It is worth enabling only once its lifetimes
and failure paths are covered on CPU and on each GPU backend, and the rebuild
cost it removes has been measured.
