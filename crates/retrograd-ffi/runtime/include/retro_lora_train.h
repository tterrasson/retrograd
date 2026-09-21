#ifndef RETRO_LORA_TRAIN_H
#define RETRO_LORA_TRAIN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct retro_trainer retro_trainer;

// Device selection for the training backend.
typedef enum retro_device {
    // Use a GPU backend (Metal, Vulkan, or CUDA) when one is compiled in and
    // present, otherwise fall back to CPU.
    RETRO_DEVICE_AUTO = 0,
    // Force CPU-only execution.
    RETRO_DEVICE_CPU = 1,
    // Force GPU offload; fails at trainer creation if no GPU backend is available.
    RETRO_DEVICE_GPU = 2,
} retro_device;

typedef enum retro_kv_dtype {
    RETRO_KV_DTYPE_F32 = 0,
    RETRO_KV_DTYPE_F16 = 1,
} retro_kv_dtype;

// Which parameters a run trains. Mirrors retrograd_core::TrainablePolicy, and
// is read before a single tensor is named: LORA leaves the CPU weights mapped
// read-only, every other value loads them into owned writable buffers because a
// base update would otherwise fault on a PROT_READ page.
typedef enum retro_trainable {
    RETRO_TRAINABLE_LORA    = 0,
    RETRO_TRAINABLE_FULL    = 1,
    RETRO_TRAINABLE_PARTIAL = 2,
    RETRO_TRAINABLE_HYBRID  = 3,
} retro_trainable;

// The optimizers this runtime can build an update step for. The values are
// ggml_opt_optimizer_type's, which is what retro_train_config.optimizer is
// passed through to.
typedef enum retro_optimizer {
    RETRO_OPTIMIZER_ADAMW = 0,
    RETRO_OPTIMIZER_SGD   = 1,
    RETRO_OPTIMIZER_MUON  = 2,
    RETRO_OPTIMIZER_GEFEN = 3,
} retro_optimizer;

// Which fixed-block state a Gefen run keeps. Not a coefficient: it selects a
// slot table, so it moves the layout version a checkpoint records rather than
// parameterizing one layout.
typedef enum retro_gefen_variant {
    RETRO_GEFEN_SHARED_V    = 0,
    RETRO_GEFEN_QUANTIZED_M = 1,
} retro_gefen_variant;

typedef enum retro_checkpoint_dtype {
    // F32 inserts no casts at all, so the recompute stays bit-exact.
    RETRO_CHECKPOINT_DTYPE_F32  = 0,
    RETRO_CHECKPOINT_DTYPE_F16  = 1,
    RETRO_CHECKPOINT_DTYPE_BF16 = 2,
} retro_checkpoint_dtype;

// Identifies a single training-backward op for the correctness probe API.
typedef enum retro_probe_op {
    RETRO_PROBE_OP_SILU_BACK = 1,
    // RMS-norm backward: src0 = dy (grad), src1 = x (forward input), param0 = eps.
    RETRO_PROBE_OP_RMS_NORM_BACK = 2,
    // out-prod: dst[i0,i1] = sum_k src0[i0,k] * src1[i1,k] (per batch); output
    // shape differs from the inputs, so size `dst` accordingly.
    RETRO_PROBE_OP_OUT_PROD = 3,
    // soft-max backward: src0 = dy (grad of softmax output), src1 = y (softmax
    // output), param0 = scale, param1 = max_bias.
    RETRO_PROBE_OP_SOFT_MAX_BACK = 4,
    // cross-entropy loss forward: src0 = logits, src1 = labels (same shape);
    // output is a single scalar (dst_len >= 1). A positive param1 pins the
    // active-row normalization count (see ggml_opt_set_loss_active_rows).
    RETRO_PROBE_OP_CROSS_ENTROPY_LOSS = 5,
    // cross-entropy loss backward: src0 = grad of the loss (scalar, ne_src0 =
    // [1,1,1,1]), src1 = logits, src2 = labels; output has the logits shape.
    // A positive param1 pins the active-row normalization count.
    RETRO_PROBE_OP_CROSS_ENTROPY_LOSS_BACK = 6,
    // get-rows backward: src0 = grad rows [n_embd, n_rows], src1 = row indices
    // as floats (each cast to int32 internally, ne_src1 = [n_rows,1,1,1]),
    // src2 = shape template [n_embd, n_vocab] (data unused); output has the
    // src2 shape. Duplicate indices accumulate.
    RETRO_PROBE_OP_GET_ROWS_BACK = 7,
    // out-prod with a Q8_0-quantized src0 (the activation-gradient case where
    // src0 is a quantized model weight): src0 arrives as F32, is quantized to
    // Q8_0 internally (ne_src0[0] must be a multiple of 32), then
    // dst[i0,i1] = sum_k dequant(src0)[i0,k] * src1[i1,k] as for OUT_PROD.
    RETRO_PROBE_OP_OUT_PROD_Q8_0 = 8,
    // Same activation-gradient contraction with src0 quantized to Q5_0.
    RETRO_PROBE_OP_OUT_PROD_Q5_0 = 9,
    // SSM convolution backward: src0=sx [ncs,d_inner,n_s], src1=kernel
    // [d_conv,d_inner], src2=dy [d_inner,n_t,n_s].
    RETRO_PROBE_OP_SSM_CONV_BACK = 10,
    // SSM scan backward test harness. ne_src0=[d_state,head_dim,n_head,n_slots]
    // and ne_src1=[n_group,n_tokens,n_seqs,A.ne0]; src0 is the concatenation
    // s,x,dt,A,B,C,ids-as-f32,ds. src1 is ignored but must be non-null.
    RETRO_PROBE_OP_SSM_SCAN_BACK = 11,
    // K-quantized OUT_PROD variants. src0 arrives as F32 and is quantized by
    // the probe; ne_src0[0] must be a multiple of the 256-value K block.
    RETRO_PROBE_OP_OUT_PROD_Q2_K = 12,
    RETRO_PROBE_OP_OUT_PROD_Q3_K = 13,
    RETRO_PROBE_OP_OUT_PROD_Q4_K = 14,
    RETRO_PROBE_OP_OUT_PROD_Q5_K = 15,
    RETRO_PROBE_OP_OUT_PROD_Q6_K = 16,
    // Additional 32-value block formats supported by the GPU OUT_PROD kernels.
    RETRO_PROBE_OP_OUT_PROD_Q4_0 = 17,
    RETRO_PROBE_OP_OUT_PROD_Q4_1 = 18,
    RETRO_PROBE_OP_OUT_PROD_Q5_1 = 19,
    // One AdamW step with an F16 parameter and F32 gradient/moments. src0 is
    // the initial weight, src1 its gradient, param0=learning rate, param1=wd.
    RETRO_PROBE_OP_OPT_STEP_ADAMW_F16 = 20,
    // Packed differentiable Flash Attention probe. ne_src0 encodes
    // {head_k, head_v, query_rows, kv_rows}; ne_src1 encodes
    // {query_heads, kv_heads, batches, causal_mask}. src0 packs Q/K/V/dO.
    RETRO_PROBE_OP_FLASH_ATTN_BACK = 21,
    // L2-norm backward: src0 = dy (grad), src1 = x (forward input), param0 = eps.
    RETRO_PROBE_OP_L2_NORM_BACK = 22,
    // Gated delta net (Qwen3-Next / KDA) backward probe. ne_src0 =
    // {S_v, H, n_tokens, n_seqs}; ne_src1 = {K, kda_flag, 1, 1}. src0 packs
    // q,k,v,g,beta,state,grad (see ggml_gated_delta_net_back); no head/batch
    // broadcast between q/k and v is exercised (q/k head count == H).
    RETRO_PROBE_OP_GATED_DELTA_NET_BACK = 23,
    // Recurrent-state rollback snapshot gather. src0 = conv_input
    // [kernel_m1 + n_seq_tokens, n_channels, n_seqs], param0 = kernel_m1,
    // param1 = K. src1 is ignored but must be non-null.
    RETRO_PROBE_OP_CONV_RS_GATHER = 24,
    // Repeat backward: src0 = the broadcast gradient, src1 = the shape template
    // it reduces onto (its data is unused). Every src1 dimension must divide the
    // matching src0 one.
    RETRO_PROBE_OP_REPEAT_BACK = 25,
    // Out-prod with a non-F32 src0, the type given by param0 (a ggml_type id, cast
    // from float). Supersedes the RETRO_PROBE_OP_OUT_PROD_<T> ids, which stay
    // for compatibility: one op parameterized by type instead of one id per type,
    // so a type added to GGML_RETRO_DEQUANT_TYPES needs no new ABI. The caller
    // still passes F32 data; it is converted to the requested type internally, so
    // both backends decode identical bytes. ne_src0[0] must be a multiple of the
    // type's block size. Enumerate the valid types with retro_dequant_types().
    RETRO_PROBE_OP_OUT_PROD_QUANT = 26,
    // Inclusive prefix sum along ne0: src0 = x, output has the src0 shape.
    // src1 is ignored but must be non-null (the probe ABI always takes two
    // inputs). The second op with a RIR variant.
    RETRO_PROBE_OP_CUMSUM = 27,
    // The AdamW step with a BF16 parameter; same ABI as the F16 probe.
    RETRO_PROBE_OP_OPT_STEP_ADAMW_BF16 = 28,
    // The SGD step on the two half-precision storages: src0 the initial
    // weight, src1 its gradient, param0 = learning rate, param1 = weight
    // decay, src2 = {gradient scale, rounding seed} as for AdamW. One id per
    // (optimizer, storage): four separate kernels, so a test naming the wrong
    // one fails rather than measuring another.
    RETRO_PROBE_OP_OPT_STEP_SGD_F16 = 29,
    RETRO_PROBE_OP_OPT_STEP_SGD_BF16 = 30,
} retro_probe_op;

// Which kernel implementation a probe asks for / reports.
typedef enum retro_kernel_impl {
    // Whatever the configured RIR mode selects; a contract rejection falls
    // back to the native kernel before launch and is reported, not hidden.
    RETRO_KERNEL_IMPL_AUTO   = 0,
    // Force the native kernel for this run, whatever the configured mode is.
    RETRO_KERNEL_IMPL_NATIVE = 1,
    // Require the RIR variant. Fails if it does not run - a probe that fell
    // back would report a green "RIR" test that in fact exercised the native
    // kernel.
    RETRO_KERNEL_IMPL_RIR    = 2,
} retro_kernel_impl;

// Rejection taxonomy mirror of ggml_rir_reject. Kept as a separate enum so the
// FFI does not force the fork's header on its consumers; the values are ABI and
// a conformance test pins them to the ggml side.
typedef enum retro_kernel_reject {
    RETRO_KERNEL_MATCHED              = 0,
    RETRO_KERNEL_REJECT_WRONG_OP      = 1,
    RETRO_KERNEL_REJECT_DTYPE         = 2,
    RETRO_KERNEL_REJECT_RANK          = 3,
    RETRO_KERNEL_REJECT_SHAPE         = 4,
    RETRO_KERNEL_REJECT_STRIDE        = 5,
    RETRO_KERNEL_REJECT_QUANT_BLOCK   = 6,
    RETRO_KERNEL_REJECT_INTEGER_RANGE = 7,
    RETRO_KERNEL_REJECT_MISSING_FEATURE = 8,
    RETRO_KERNEL_REJECT_PIPELINE      = 9,
    RETRO_KERNEL_REJECT_POLICY_NATIVE = 10,
    RETRO_KERNEL_REJECT_DEVICE_GRID   = 11,
    RETRO_KERNEL_REJECT_DEVICE_ALIGNMENT = 12,
    // A node naming an op-family member for which no registered kernel writes.
    // This contract reason was appended rather than inserted:
    // 2-7 and 13 are the contract, 8-12 are the device, and these numbers are
    // ABI on both sides of this header.
    RETRO_KERNEL_REJECT_OP_VARIANT    = 13,
} retro_kernel_reject;

// Number of buckets in `retro_rir_counters.reject_by_reason`, including
// `MATCHED`. Must equal `GGML_RIR_REJECT_COUNT`; the C implementation asserts
// this before copying the counters.
#define RETRO_KERNEL_REJECT_COUNT 14

// What the probe actually did. `struct_size` must be set by the caller to
// sizeof(retro_kernel_run_info) before the call: it is how this struct grows
// without breaking a separately compiled caller.
typedef struct retro_kernel_run_info {
    uint32_t struct_size;
    int32_t  requested_impl;  // retro_kernel_impl, echoed back
    int32_t  executed_impl;   // retro_kernel_impl: NATIVE or RIR, never AUTO
    // retro_kernel_reject. MATCHED with executed_impl == NATIVE means the
    // contract matched but the mode did not dispatch (observe), or the op has
    // no registered variant at all.
    int32_t  reject_reason;
    char     variant[64];     // variant_id that ran, empty when native
} retro_kernel_run_info;

// retro_probe_op_run with an explicit implementation choice and a report of
// what ran. Arguments up to `dst_len` are identical to retro_probe_op_run, which
// keeps its exact behaviour (== AUTO with no reporting) so existing oracles do
// not move. `info` may be NULL.
// Requesting RIR fails cleanly when the build has no RIR variant for the op, or
// when the process was not started with a RETRO_RIR_MODE that builds the
// pipelines: the policy has to be known before the backend context exists, so
// this entry point cannot turn RIR on retroactively.
int retro_probe_op_run_ex(
    int32_t op,
    int32_t use_gpu,
    const int64_t * ne_src0,
    const float * src0,
    const int64_t * ne_src1,
    const float * src1,
    const int64_t * ne_src2,
    const float * src2,
    float param0,
    float param1,
    float * dst,
    size_t dst_len,
    int32_t implementation,
    retro_kernel_run_info * info);

// retro delta: the two fixed-block Gefen ops driven directly, with no model and
// no training graph.
//
// A Gefen step reached through a run is a step whose inputs were produced by a
// backward pass: a gradient nobody chose, a state nobody wrote, and one
// arithmetic path per fixture. This drives both phases over state the caller
// supplies, which is what makes the algorithm's edge cases - a zero block, a
// block of one repeated magnitude, every one of the 256 index codes, a partial
// trailing block, a decay-only step - reachable at all, and what lets the same
// inputs be run on two backends and compared.
//
// Every buffer is host memory in the caller's layout; the probe uploads it,
// computes `n_steps` full updates on the selected device, and reads the state
// back into the same buffers. `weights`, `moment`/`indices`, `scales` and `v`
// are therefore in/out.
//
// Which pointers are required depends on the variant:
//   shared_v     moment != NULL, indices/scales/codebook == NULL
//   quantized_m  indices/scales/codebook != NULL, moment == NULL
// `stats` is phase A's answer for the last step, or NULL when the caller does
// not want it.
typedef struct retro_gefen_probe {
    uint32_t struct_size;
    int32_t  use_gpu;
    int32_t  variant;     // ggml_opt_gefen_variant
    int32_t  block_size;
    int64_t  n_elements;
    int64_t  n_blocks;    // ceil(n_elements / block_size)
    int32_t  levels;      // codebook entries, 0 under shared_v
    int32_t  n_steps;     // full updates over the same gradient, >= 1

    float         * weights;  // [n_elements], in/out
    const float   * grad;     // [n_elements]
    float         * moment;   // [n_elements] F32, shared_v only, in/out
    uint8_t       * indices;  // [n_elements] bytes, quantized_m only, in/out
    float         * scales;   // [n_blocks], quantized_m only, in/out
    float         * v;        // [n_blocks], in/out
    const float   * codebook; // [levels], quantized_m only
    const float   * pars;     // 8: alpha, beta1, beta2, eps, wd, beta1h, beta2h, grad_scale
    float         * stats;    // [2*n_blocks], out, may be NULL
} retro_gefen_probe;

// Returns 0, or non-zero with retro_last_error() set. A device that declines
// either phase is one of the failures, and its message says so: the two are
// admitted together, so a partial answer is never returned.
int retro_probe_gefen_run(retro_gefen_probe * probe);

// Whether `device` (0 CPU, 1 GPU) declares both Gefen phases for this variant
// and block size. Writes 1 or 0 into `out_supported`. The predicate is the
// backend's own, so it is the same answer a run's preflight gets.
int retro_probe_gefen_supported(
    int32_t use_gpu,
    int32_t variant,
    int32_t block_size,
    int32_t * out_supported);

// Process-wide RIR dispatch counters. Monotonic since
// process start, across every backend. `struct_size` must be set by the caller.
typedef struct retro_rir_counters {
    uint32_t struct_size;
    int32_t  mode;                // ggml_rir_mode: 0 off, 1 observe, 2 prefer, 3 require
    uint64_t ops_seen;
    uint64_t rir_eligible;
    uint64_t rir_dispatched;
    uint64_t native_dispatched;
    uint64_t fallback_contract;
    uint64_t fallback_feature;
    uint64_t fallback_pipeline;
    // Same rejections bucketed by retro_kernel_reject: the aggregate counters say
    // how much fell back, this says what to widen first on a real graph.
    uint64_t reject_by_reason[RETRO_KERNEL_REJECT_COUNT];
} retro_rir_counters;

// Fills `out` with a snapshot. Returns 0, or non-zero when `out` is NULL or its
// struct_size does not match this build.
int retro_rir_counters_get(retro_rir_counters * out);

// One line per compiled RIR variant:
//   "variant\tkernel\tggml_op\tvariant_id\tbackend\tpriority"
// plus one line per op/backend pair the registry locks to the native kernel:
//   "native-only\tggml_op\tbackend"
// so a CUDA build reports its reserved kernels instead of staying silent about
// them. Same buffer contract as retro_backend_list.
int retro_rir_variant_report(
    char * buffer,
    size_t n_buffer,
    size_t * out_n_bytes);

// The census of the graphs computed so far, empty unless RETRO_RIR_CENSUS=1 was
// set before the first graph. One row per (ggml_op, backend) *whatever* RIR
// knows about it, which is what makes it able to name the op to write next:
//   "census\tggml_op\tbackend\tnodes\tbytes\telements\tregistered|uncovered"
//   "census-shape\tggml_op\tbackend\ttype\tne0,ne1,ne2,ne3\tnodes"
//   "census-shape-overflow\tggml_op\tbackend\tnodes"
// Bytes, not time: nothing here executed a kernel - see ggml-rir.h. Same buffer
// contract as retro_backend_list.
int retro_rir_census_report(
    char * buffer,
    size_t n_buffer,
    size_t * out_n_bytes);

// Runs the registry's selection rule on synthetic variant tables. Returns 0 when
// every case holds, or a bitmask of failing cases. Needs no device.
uint32_t retro_rir_selection_selftest(void);

// Writes the ggml_type ids of GGML_RETRO_DEQUANT_TYPES -- the types the training
// ops can decode in place -- into `out_types`, and returns how many there are.
// Writes nothing and returns the count when `out_types` is NULL or `cap` is too
// small. Lets a test enumerate the supported types instead of restating them, so
// adding a row to the table extends the tests automatically.
size_t retro_dequant_types(int32_t * out_types, size_t cap);

// ggml_type_name for a ggml_type id, for test output. Returns "?" if unknown.
const char * retro_ggml_type_name(int32_t type);

// Block geometry of a ggml_type: blck_size (logical elements per block) and
// type_size (physical bytes per block, i.e. nb[0]). Returns 0, or -1 on an
// unknown type. The RIR quantized-format table restates both numbers so the
// generated header can stay legal MSL; this is what keeps the restatement
// honest.
int retro_quant_traits(int32_t type, int64_t * out_block_elements, size_t * out_block_bytes);

// Quantizes `n` F32 values with ggml's reference quantizer and decodes them
// back with ggml's own to_float, returning **both** the quantized bytes and the
// decoded values. A parity test feeds those same bytes to the RIR oracle: the
// comparison is then between two decoders, never between two quantizations of
// the same F32.
// `n` must be a multiple of the type's block size, and `out_bytes_cap` at least
// (n / blck_size) * type_size. Returns 0, or -1 if any of that does not hold.
int retro_quant_roundtrip(
    int32_t         type,
    int64_t         n,
    const float   * src,
    uint8_t       * out_bytes,
    size_t          out_bytes_cap,
    float         * out_dequant);

// See retro_fused_sparse_ce_probe's w_type.
#define RETRO_FUSED_CE_W_TYPE_GGML_BASE 1000

// retro delta (plan DISTILL D6.5): ceiling on the number of sparse targets a
// position may carry. Mirrors GGML_FUSED_SPARSE_CE_K_MAX, which bounds the
// scratch the Vulkan and Metal kernels reserve per position; kept here so a
// caller of this header does not have to include ggml.h to validate a sidecar.
#define RETRO_FUSED_CE_K_MAX 32

// Runs a single ggml training-backward op in isolation on the chosen backend and
// copies the result into `dst`. Used by targeted GPU-vs-CPU correctness tests:
// feed identical random inputs, run once with use_gpu=0 (CPU reference) and once
// with use_gpu=1 (the registered GPU backend), and compare. Tensors are F32
// (except the op-specific integer inputs noted on the enum); shapes are 4-element
// arrays. `ne_src2`/`src2`
// are only read by ops that take a third input and may be NULL otherwise.
// `param0`/`param1` are op-specific scalars. Fails (non-zero) if the op is
// unknown, a required input is missing, the backend is unavailable, or `dst_len`
// is smaller than the output element count.
int retro_probe_op_run(
    int32_t op,
    int32_t use_gpu,
    const int64_t * ne_src0,
    const float * src0,
    const int64_t * ne_src1,
    const float * src1,
    const int64_t * ne_src2,
    const float * src2,
    float param0,
    float param1,
    float * dst,
    size_t dst_len);

// Chunked/fused vocabulary cross-entropy parity probe. For one set of inputs,
// computes both the current full-vocabulary path (mul_mat -> weighted
// cross-entropy -> autodiff back to the hidden states, using the real fork ggml
// ops) and a fused/tiled CPU reference that never materializes the full
// [n_vocab, n_tokens] logits. Lets a test assert exact loss + grad_h parity for a
// chosen tile count before the fused operator is wired into the training graph.
// Layout is ggml column-major (ne0 fastest):
//   h: [n_embd, n_tokens]   final hidden states
//   w: [n_embd, n_vocab]    frozen projection head; logits = mul_mat(w, h)
//   targets: [n_tokens] int32     target vocab id per token, < 0 marks a masked
//                                    (inactive) token
//   weights: [n_tokens] f32       GRPO coefficient placed at the target id
//   bias: [n_vocab] f32, or NULL for no bias - a fixed (non-trainable)
//              per-vocab additive term added to every logit before the
//              softmax, matching an ADD(mul_mat(w, h), bias) output head
//              (e.g. gemma4's suppressed-token logits bias). Never receives
//              a gradient.
// seq_chunk caps how many tokens the fused operators process at once (0 = all);
// offload_h additionally places grad_h on top of h, reproducing the in-place
// allocation the training graph gets from ggml-alloc when the flag is set. Both
// are pure memory knobs: the returned loss and grad_h must not move with them.
// w_type selects the projection-head storage exercised by the fused path:
// 0 = F32, 1 = Q8_0, 2 = Q6_K, 3 = Q4_0, 4 = Q4_K, 5 = Q5_K, 6 = F16 (the
// full path always runs on the dequantized head so it stays the exact oracle).
// A value >= RETRO_FUSED_CE_W_TYPE_GGML_BASE instead carries
// (w_type - RETRO_FUSED_CE_W_TYPE_GGML_BASE) as a raw ggml_type id, so a test can
// sweep retro_dequant_types() without a new wire value per type.
// grad_loss is the upstream gradient seeded into the
// scalar loss. Outputs are caller-allocated: *_loss_* are scalars; *_grad_h_*
// are [n_embd, n_tokens]. Fails (non-zero) on non-positive dims, null buffers,
// an out-of-range target, an unknown w_type, or a quant block-size mismatch.
int retro_fused_sparse_ce_probe(
    int32_t n_embd,
    int32_t n_tokens,
    int32_t n_vocab,
    // retro delta (plan DISTILL D6.5): sparse targets per position. 0 and 1 are
    // the same one-target probe; above that `targets` and `weights` are
    // [n_topk, n_tokens] and the full path builds the dense label row from the
    // n_topk entries, which is what makes it the oracle for the fused operator.
    int32_t n_topk,
    int32_t n_tiles,
    int32_t seq_chunk,
    int32_t offload_h,
    int32_t w_type,
    int32_t use_gpu,
    const float * h,
    const float * w,
    const int32_t * targets,
    const float * weights,
    const float * bias,
    float grad_loss,
    float * out_loss_full,
    float * out_loss_fused,
    float * out_grad_h_full,
    float * out_grad_h_fused);

// Compares the production vectorized token log-softmax (`vectorized=true`)
// with its scalar reference without loading a model. This is a numerical
// correctness probe; `out_logprob` receives one temperature-1 logprob.
int retro_probe_token_logprob(
    const float * logits,
    size_t n_vocab,
    int32_t token,
    bool vectorized,
    float * out_logprob);

// Lists the ggml backend devices registered in this build (one per line:
// "type\tname\tdescription"). Useful to confirm a GPU backend is present.
// Writes a NUL-terminated string when buffer is large enough; always sets
// *out_n_bytes to the length excluding the NUL. Returns -2 if the buffer is too
// small.
int retro_backend_list(
    char * buffer,
    size_t n_buffer,
    size_t * out_n_bytes);

// Performs a real initialization of the first registered GPU backend. Unlike
// retro_backend_list, this creates the backend/device context and synchronizes
// it, so callers can distinguish a registered backend from a usable runtime
// queue. Returns -1 when no usable GPU device is available.
int retro_gpu_runtime_probe(void);

// Reads the memory budget of the first GPU device registered in this build
// (`ggml_backend_dev_memory`). Both outputs are in bytes and may be NULL.
// Returns 0 on success, -1 when the build has no GPU device, in which case the
// outputs are left untouched. Host RSS does not account for discrete VRAM, so
// this is the only way to observe the device peak of a training run.
int retro_device_memory(
    size_t * out_free,
    size_t * out_total);

// Host<->device transfer rates, measured on this build's first GPU device.
// This is the bandwidth gate for activation offloading (
// The contract): the plan refuses the offload path if its traffic would add more than 30%
// of the compute it is meant to hide behind, and that ratio cannot
// be estimated without knowing what a copy of this size actually costs on this
// machine. Measured rather than assumed, because the number that decides it is a
// property of the link, not of the design.
// Two rows, because the difference between them *is* the finding: `pinned_*` uses
// the backend's own host buffer type (page-locked where the backend has one) and
// `pageable_*` uses ordinary malloc'd memory. A backend that reports no host
// buffer type fills the pinned row with the pageable one and sets
// `pinned_is_pageable`, so a caller never reads a fallback as a measurement of
// page-locked memory.
// A unified-memory device (Metal, an iGPU) legitimately reports rates far above
// any bus figure and near-identical rows: there is no bus. That is a result,
// it says an offload ring has nothing to move - not a broken measurement, and
// the rate is what says it. `device_buffer_is_host` is *not* that signal: Metal
// has unified memory and still answers false, because its default buffer type is
// a Metal buffer and not host memory. Read the flag as what it is.
typedef struct retro_transfer_rates {
    // Bytes per transfer and how many round trips were timed, echoed back so a
    // rate is never read without the size it was measured at.
    uint64_t bytes_per_transfer;
    uint64_t iterations;
    double pinned_h2d_bytes_per_second;
    double pinned_d2h_bytes_per_second;
    double pageable_h2d_bytes_per_second;
    double pageable_d2h_bytes_per_second;
    // True when this build's device exposes no dedicated host buffer type, so the
    // pinned rates are the pageable rates repeated.
    bool pinned_is_pageable;
    // True when the device's own default buffer type *is* host memory, i.e. a
    // backend with no separate allocation at all. False on Metal despite its
    // unified memory: a Metal buffer is not a host buffer. This says how the
    // backend allocates, never how fast the link is - that is the rates' job.
    bool device_buffer_is_host;
} retro_transfer_rates;

// Times `iterations` host->device and device->host copies of `bytes` each, twice:
// once out of a pinned host buffer and once out of a pageable one. Every copy is
// followed by a device synchronization, so the rates are end-to-end and include
// no overlap - the pessimistic direction, which is the correct one for a budget.
// A warm-up round trip is discarded before timing.
// Returns 0 on success, -1 when the build has no GPU device or the allocation
// fails, in which case `out_rates` is left untouched.
int retro_transfer_probe(
    size_t bytes,
    uint32_t iterations,
    retro_transfer_rates * out_rates);

// Capacity of the fixed-size name fields in retro_model_info. Architecture
// names and ggml type names are short and bounded by llama.cpp's own tables;
// a fixed array keeps the struct a plain value the caller can stack-allocate
// instead of forcing a second two-call string round trip for two labels.
#define RETRO_MODEL_INFO_NAME_MAX 64

// Geometry of a GGUF model, everything a memory estimator needs *before* a
// training context exists. Read by opening the model with no context and no
// LoRA adapter (retro_model_info), which is orders of magnitude cheaper than
// retro_trainer_new and is what makes it usable as a planning input.
// KV-cache sizing uses `n_embd_k_gqa` / `n_embd_v_gqa`, not
// n_head_kv * n_embd_head_*: those two already fold in per-layer variation and
// MLA, and are the widths llama.cpp actually allocates per layer and per token.
// The `*_max` semantics matter for models whose layers disagree (sliding-window
// or hybrid attention): a maximum over-estimates, which is the direction the
// resolver is required to err in.
typedef struct retro_model_info {
    uint32_t n_layer;
    uint32_t n_embd;
    // Maximum feed-forward width across layers. Publishing it avoids having
    // every planner reverse-engineer architecture-specific widths from n_params.
    uint32_t n_ff;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
    // Maximum per-layer KV width in elements.
    uint32_t n_embd_k_gqa;
    uint32_t n_embd_v_gqa;
    // Recurrent/state widths, non-zero only for recurrent or hybrid models.
    // Their memory does not scale with n_ctx, unlike the KV cache.
    uint32_t n_embd_r;
    uint32_t n_embd_s;
    uint32_t n_vocab;
    uint32_t n_ctx_train;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint64_t n_params;
    // Sum of the model's tensor bytes as loaded (llama_model_size), which is
    // what the weights actually occupy - not the file length.
    uint64_t model_size_bytes;
    // Length of the GGUF file on disk, for reference and for cache keys.
    uint64_t file_size_bytes;
    // Bytes held by tensors of `dominant_weight_type`, i.e. how much of the
    // model that quantization actually accounts for.
    uint64_t dominant_weight_bytes;
    // Whether the output projection reuses the token-embedding tensor. When it
    // does, the vocabulary head costs no extra weights.
    bool tied_embeddings;
    bool is_recurrent;
    bool has_encoder;
    char architecture[RETRO_MODEL_INFO_NAME_MAX];
    // ggml name of the type accounting for the most weight bytes ("Q4_K",
    // "F16",...). The scratch cost of dequantization depends on it.
    char dominant_weight_type[RETRO_MODEL_INFO_NAME_MAX];
} retro_model_info;

// Model/device capabilities consumed by the planner. Unlike the human report,
// this is a versioned data contract and must never be parsed from text.
typedef struct retro_model_capabilities {
    uint32_t struct_size;
    bool shared_prefix_packed_training;
    bool fused_sparse_cross_entropy;
    bool differentiable_flash_attention;
} retro_model_capabilities;

#define RETRO_PREFLIGHT_FINGERPRINT_MAX 65

// Machine-readable summary of the exact graph built by training preflight.
// Detailed human diagnostics remain available separately; callers must not
// parse them to make planning decisions.
typedef struct retro_preflight_summary {
    uint32_t struct_size;
    int32_t missing_gradient_rules;
    uint64_t active_device_fallback_nodes;
    char graph_fingerprint[RETRO_PREFLIGHT_FINGERPRINT_MAX];
} retro_preflight_summary;

// Fills `out_info` from the GGUF at `model_path`, without creating a training
// context, a KV cache or a LoRA adapter. `device` follows retro_device; it only
// selects which backend the tensor metadata is read through, and no weights are
// offloaded. Returns 0 on success, -1 with retro_last_error set otherwise.
// Named `retro_read_model_info` rather than `retro_model_info` because C puts
// typedef names and function names in the same namespace, and the struct owns
// that name.
int retro_read_model_info(
    const char * model_path,
    int32_t device,
    retro_model_info * out_info);

// --- tensor inventory --------------------------------------------------------
// The per-tensor counterpart of retro_model_info, which carries aggregate
// geometry and a dominant dtype and therefore cannot resolve a tensor pattern
// or an exact mixed-dtype cost. Selection, optimizer allocation, checkpointing
// and the planner all resolve against this list.
//
// Versioned because it is a data contract and not a diagnostic: a reader that
// keys on a field has to be able to tell a v1 inventory from a later one rather
// than silently miss a tensor family.
#define RETRO_TENSOR_INVENTORY_VERSION 1

// Long enough for every GGUF tensor name llama.cpp loads (`blk.<N>.<stem>.
// <suffix>`, plus adapter suffixes); a name that would not fit is reported as
// an error rather than silently truncated, since a truncated name is a name
// that selects the wrong tensor.
#define RETRO_TENSOR_NAME_MAX 128

typedef struct retro_tensor_desc {
    char name[RETRO_TENSOR_NAME_MAX];
    // ggml dimension order, ne[0] fastest-varying; trailing dimensions are 1.
    int64_t ne[4];
    uint64_t n_elements;
    uint64_t n_bytes;
    // Opaque identity of the allocation backing this tensor: two names for one
    // allocation carry the same value, which is what lets a resolver refuse to
    // issue two optimizer updates against one buffer. Comparable only within
    // one inventory - it is not stable across processes and must never reach a
    // manifest or a signature.
    uint64_t storage_id;
    // ggml type name ("F32", "Q4_K", ...), the same spelling the backend report
    // uses. A name rather than the enum value so the contract does not move
    // when ggml renumbers its table.
    char type_name[RETRO_MODEL_INFO_NAME_MAX];
} retro_tensor_desc;

// Enumerates the tensors of the GGUF at `model_path` as the loader sees them,
// with no context, no KV cache, no adapter and no device offload: the model is
// mapped, so no weight is copied and nothing is allocated per tensor.
//
// It is the loader's view rather than the raw file's on purpose - that is where
// llama.cpp's own aliases have been resolved, so `storage_id` answers "is this
// the same buffer" and a tied head is visible as the absence of its tensor
// rather than as a name the file happens not to spell.
//
// Two-call contract, like every sized reader here: pass `out_tensors = NULL`
// and `n_max = 0` to learn the count, then call again with a buffer of at least
// that many entries. A buffer shorter than the count fails with -2 and writes
// nothing. `out_version` and `out_count` are required in both calls.
int retro_read_tensor_inventory(
    const char * model_path,
    uint32_t * out_version,
    retro_tensor_desc * out_tensors,
    size_t n_max,
    size_t * out_count);

// Structured counterpart of the byte fields in retro_trainer_backend_report.
// The text report stays the human-facing diagnostic; this is what code reads,
// so no caller has to parse a report to know where memory went. Both are
// rendered from the same computation, so they cannot disagree.
// The `*_compute_bytes` fields only become non-zero once a graph has been
// reserved (i.e. after the preflight); before that they read zero, which is a
// fact about the allocation and not a missing measurement.
typedef struct retro_memory_report {
    // Static allocations, summed from the ggml backend buffers.
    uint64_t model_weight_bytes;
    uint64_t optimizer_kv_bytes;
    uint64_t optimizer_compute_bytes;
    uint64_t generation_kv_bytes;
    uint64_t generation_compute_bytes;
    // False when the run shares the optimizer context for generation, in which
    // case both generation_* fields are zero rather than unknown.
    bool has_generation_context;
    // The resolved trainable set, whatever it is made of: LoRA factors today,
    // selected base tensors once full/partial training lands. Named for the
    // role rather than for one policy so a caller reading a hybrid run does not
    // have to know which field to trust.
    uint64_t trainable_parameter_bytes;
    uint64_t trainable_gradient_bytes;
    // Persistent optimizer state of that set: AdamW's two moments today, and
    // whatever the selected optimizer's descriptor allocates later. Zero for an
    // optimizer that keeps none (SGD), which is a fact and not a missing value.
    uint64_t optimizer_state_bytes;
    // Whether `trainable_parameter_bytes` is *already* inside
    // `model_weight_bytes`. Adapter factors are allocated on top of the loaded
    // model and are not; base tensors selected out of it are. The device/host
    // rollup below adds the parameter bytes only when this is false. The
    // gradient and the optimizer state are always additional.
    bool trainable_parameters_are_model_subset;
    // Which budget the trainable state draws on, per family rather than as one
    // boolean: a hybrid run can hold its adapter on the device and its selected
    // base tensors on the host, and one flag cannot describe both.
    // `base_trainable_on_host` is meaningless while no base tensor is trainable;
    // `trainable_parameters_are_model_subset` is what says whether it applies.
    bool adapter_on_host;
    bool base_trainable_on_host;
    // These allocations, rolled up into the two budgets they draw on.
    uint64_t device_bytes;
    uint64_t host_bytes;
    // Measured device memory (llama_opt_memory), as opposed to the summed
    // buffer totals. They differ by exactly the backends' own scratch and
    // the graph allocator's transient reserve - the gap every VRAM lever lands
    // in. `device_memory_samples == 0` means unavailable, not measured zero,
    // and all measured fields are then zero.
    uint64_t device_total_bytes;
    uint64_t device_used_bytes;
    uint64_t device_peak_used_bytes;
    uint64_t backend_scratch_bytes;
    uint64_t backend_scratch_peak_bytes;
    uint64_t device_memory_samples;
    // Retained activation checkpoints from the last backward graph. All fields
    // are zero when checkpointing is off or no backward graph exists; use
    // `checkpoint_count == 0` to distinguish that state.
    // `checkpoint_retained_bytes` is what the checkpoints hold *as held*: a run
    // with checkpoint_dtype=f16 reports the 16-bit copies, not the F32 tensors the
    // forward built. It is the numerator of the offload trigger, whose
    // denominator is `device_peak_used_bytes`.
    uint64_t checkpoint_count;
    uint64_t checkpoint_retained_bytes;
    uint64_t checkpoint_live_peak_bytes;
    uint64_t checkpoint_live_peak_count;
    // Bytes held by checkpoints alive across at least half the backward graph;
    // these are the best candidates for activation offloading.
    uint64_t checkpoint_long_lived_bytes;
    uint64_t checkpoint_long_lived_count;
    // Lifetime spans in graph-node positions, not seconds. Nodes differ by orders
    // of magnitude in cost, so this ranks checkpoints against each other and says
    // nothing about durations.
    uint64_t checkpoint_graph_nodes;
    uint64_t checkpoint_max_span_nodes;
    uint64_t checkpoint_total_span_nodes;
} retro_memory_report;

// Structured memory breakdown of this trainer. Safe to call at any point after
// retro_trainer_new; the compute and measured fields fill in as the run
// progresses. Returns 0 on success.
int retro_trainer_memory_report(
    retro_trainer * trainer,
    retro_memory_report * out_report);

typedef enum retro_lora_dtype {
    RETRO_LORA_DTYPE_F32 = 0,
    RETRO_LORA_DTYPE_F16 = 1,
} retro_lora_dtype;

typedef struct retro_lora_config {
    uint32_t rank;
    float alpha;
    float dropout;
    uint32_t seed;
    // NULL/zero selects the architecture's default Retrograd LoRA profile.
    // A non-empty list overrides that profile with explicit tensor patterns.
    const char ** target_patterns;
    size_t n_target_patterns;
    // Storage dtype for A/B. Gradients and AdamW moments always remain F32.
    int32_t dtype;
} retro_lora_config;

typedef struct retro_train_config {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_ubatch;
    // Maximum number of independent sequences packed by the optimizer.
    uint32_t n_seq_max;
    // Maximum number of rollout sequences decoded together. This controls the
    // dedicated generation context's KV capacity independently of optimizer packing.
    uint32_t generation_concurrency;
    // Opt-in fast sampling context: the dedicated generation context is built
    // with an F16 KV cache and flash-attention instead of mirroring the
    // optimizer context's exact F32/no-FA settings. Halves KV memory and speeds
    // up decoding, at the cost of ulp-level differences between the sampling
    // distribution and the trained policy. Never affects the optimizer context.
    bool fast_generation_context;
    // Requested differentiable training KV storage. F16 is accepted only when
    // the active device reports the differentiable Flash Attention forward and
    // backward ops for the model's head geometry (see cap_flash_attn_back);
    // otherwise it falls back to F32. backend_report exposes the effective
    // choice and status.
    int32_t kv_dtype;
    // CPU worker threads. Zero selects the performance-core/hardware default;
    // RETRO_THREADS, when set, takes precedence at runtime.
    uint32_t threads;
    uint32_t epochs;
    float learning_rate;
    float weight_decay;
    // Global L2 gradient-norm ceiling applied before every optimizer step.
    float max_grad_norm;
    // 0=constant, 1=linear warmup/decay, 2=cosine warmup/decay.
    int32_t lr_scheduler;
    uint64_t warmup_steps;
    bool verbose;
    // One of retro_device; selects CPU vs GPU execution.
    int32_t device;
    // Fuse the projection and cross-entropy in the packed optimizer step so the
    // full [n_vocab, n_tokens] logits are never materialized. Off by default;
    // only the packed sequence path is affected.
    bool chunked_cross_entropy;
    // Vocabulary tile count C for the fused cross-entropy (>= 1). Peak logits
    // footprint is ~ n_vocab / C. Ignored unless chunked_cross_entropy is set.
    uint32_t chunked_ce_tiles;
    // Flattened (batch x seq) token chunk size for the fused cross-entropy. 0
    // processes all tokens at once (unchanged); > 0 bounds the tiled logits
    // intermediate to this many tokens, capping peak footprint independently of
    // the sequence length. Ignored unless chunked_cross_entropy is set.
    uint32_t chunked_ce_seq_chunk;
    // Let fused cross-entropy backward write grad_h over the hidden states instead
    // of a second [n_embd, n_tokens] buffer. This is numerically inert and is
    // ignored unless chunked_ce_seq_chunk > 0 bounds the staging buffer.
    bool chunked_ce_offload_logsoftmax;
    // Recompute forward activations in the packed-sequence backward graph.
    // Selected layer outputs remain live as segment boundaries. The
    // legacy micro-batch accumulation path is intentionally unchanged.
    bool gradient_checkpointing;
    // Retain one transformer-layer output every N layers (>= 1).
    uint32_t checkpoint_every_n_layers;
    // One of retro_checkpoint_dtype: precision the retained checkpoints are held
    // in across the backward. F16 halves that term but breaks recompute
    // bit-parity, so it is opt-in. Ignored unless gradient_checkpointing is set.
    int32_t checkpoint_dtype;
    // Turn any training-graph op the active GPU declines into a hard error at
    // preflight instead of a silent CPU fallback. Off by default, because a
    // fallback is correct -- it only costs a scheduler split and a device<->host
    // round trip per node. Set it when that cost is the thing being measured or
    // guarded against: on Metal, for instance, chunked_cross_entropy sends both
    // FUSED_SPARSE_CE nodes to the CPU with nothing in the log to say so.
    bool require_gpu_resident;
    // Batch/ubatch geometry of the dedicated generation context. Zero derives
    // min(n_ctx, 512), raised to generation_concurrency so a decode wave fits in
    // one launch. Larger values use more compute memory but can improve prefill.
    uint32_t generation_batch;
    // Permute SFT training rows at the start of every epoch. Evaluation rows in
    // a split dataset keep their order so eval loss remains comparable. Ignored
    // outside the SFT path.
    bool shuffle_dataset;
    // One of retro_optimizer: which update step the optimizer graph builds.
    // Placed here rather than at the end because the three bytes after
    // shuffle_dataset were padding anyway, so no later field moved.
    int32_t optimizer;
    // Seed the permutation. Each epoch's order is a pure function of
    // (shuffle_seed, epoch), so checkpoint resume reproduces an uninterrupted run.
    uint64_t shuffle_seed;
    // One of retro_trainable: which family of parameters this run trains. The
    // runtime reads it at model load, to decide between a read-only mapping and
    // owned writable buffers, and again when it installs the optimizer's
    // parameter filter. The resolved tensor names themselves arrive separately,
    // through retro_trainer_set_trainable_base().
    int32_t trainable;
    // Muon, read only when optimizer == RETRO_OPTIMIZER_MUON. The fallback rate
    // is AdamW's, for the parameters Muon's eligibility rule leaves it; an
    // orthogonalized update and an AdamW one are not in the same units, so it
    // is declared rather than taken from learning_rate. The schedule scales
    // both. Zero selects the declared defaults.
    float muon_momentum;
    float muon_ns_epsilon;
    float muon_fallback_learning_rate;
    // Newton-Schulz iterations. Structural: it decides how many nodes the
    // update graph has. Zero selects the frozen v1 count.
    uint32_t muon_ns_steps;
    bool muon_nesterov;
    // Gefen, read only when optimizer == RETRO_OPTIMIZER_GEFEN. One of
    // retro_gefen_variant; the block size is structural and zero selects the
    // frozen v1 value.
    int32_t gefen_variant;
    uint32_t gefen_block_size;
    float gefen_beta1;
    float gefen_beta2;
    float gefen_eps;
} retro_train_config;

typedef struct retro_train_metrics {
    uint32_t epoch;
    bool epoch_complete;
    uint64_t global_step;
    float train_loss;
    float eval_loss;
    float tokens_per_second;
    float learning_rate;
} retro_train_metrics;

// Monotonic wall-clock counters for the optimizer path. They are accumulated
// by the runtime so callers can snapshot before and after a logical update.
// `graph_build_seconds` covers graph construction, `allocation_seconds`
// covers backend scheduling/allocation, and `execution_seconds` covers the
// forward + backward + optimizer graph evaluation.
typedef struct retro_optimizer_timing {
    double graph_build_seconds;
    double allocation_seconds;
    double execution_seconds;
} retro_optimizer_timing;

// Device-memory counters for the optimizer path, in bytes. Mirrors
// llama_opt_memory; see retro_trainer_optimizer_memory for the semantics and for
// why the buffer-level breakdown is not enough.
typedef struct retro_optimizer_memory {
    uint64_t device_used_bytes;
    uint64_t device_total_bytes;
    uint64_t device_peak_used_bytes;
    uint64_t scratch_bytes;
    uint64_t scratch_peak_bytes;
    uint64_t n_samples;
} retro_optimizer_memory;

// Monotonic counters for the shared-prefix behavior scorer
// (retro_trainer_score_token_suffix_batch). They exist to make the two cliffs
// of that path observable rather than inferred from a wall clock:
//   - `prefix_decodes` greater than `calls` means the shared prefix was rebuilt
//     because a branch could not be evicted from the cache, i.e. the scorer
//     degraded to one prompt prefill per completion (`prefix_reprefills`
//     counts the rebuilds, `branch_evictions_refused` the refusals).
//   - `device_logprob_positions` less than `scored_positions` means the target
//     log-probability was reduced on the host, one n_vocab row at a time,
//     instead of being gathered on the device.
typedef struct retro_scoring_stats {
    uint64_t calls;
    uint64_t prefix_decodes;
    uint64_t prefix_reprefills;
    uint64_t branch_evictions_refused;
    uint64_t scored_positions;
    uint64_t device_logprob_positions;
} retro_scoring_stats;

// Monotonic counters of the generation context's prefix reuse, accumulated for
// the trainer's lifetime so a caller can attribute one update by differencing
// two snapshots.
// `prompt_tokens` is what the caller asked to be resident before sampling;
// `prefilled_tokens` is what was actually decoded to get there. Their ratio is
// the whole point of the cache: a single-turn run keeps them equal, and a
// multi-turn one drives the second toward the length of one turn rather than
// of the whole trajectory. `evictions` counts slots dropped because the context
// had fewer sequences than the rollout had live trajectories - a cache that is
// working but too small, which reads very differently from one that is missing.
typedef struct retro_generation_stats {
    uint64_t calls;
    uint64_t sequences;
    uint64_t prompt_tokens;
    uint64_t prefilled_tokens;
    uint64_t reused_tokens;
    uint64_t hits;
    uint64_t evictions;
} retro_generation_stats;

// Wall-clock accounting of the GPU duty-cycle limiter, in seconds, accumulated
// for the trainer's lifetime. See retro_trainer_set_max_gpu_duty_cycle.
// `requested_fraction` is what the caller asked for, `active` whether the
// limiter engaged; the pair distinguishes "not requested" from "requested on a
// CPU backend", which no single field can.
// `compute_seconds` covers only the accounted windows and `idle_seconds` only
// the deliberate sleeps, so `compute / (compute + idle)` converges to the
// requested fraction whenever the limiter is working and mostly echoes the
// setting back. `wall_seconds` runs from the moment the limiter was enabled, so
// `compute / wall` is the share of the *run* and the difference between the two
// ratios is the unaccounted host time - data loading, judging, tokenization,
// checkpoint I/O. A run whose first ratio is 0.50 and second 0.20 is not
// misconfigured, it is CPU-bound, and no duty cycle will free the compute its
// operator was hoping to release.
// All three are zero while the limiter is inactive.
typedef struct retro_duty_cycle_stats {
    float requested_fraction;
    bool active;
    double compute_seconds;
    double idle_seconds;
    double wall_seconds;
} retro_duty_cycle_stats;

// One step of a retro_probe_duty_cycle script. `micros` is simulated
// microseconds; it is ignored by the kinds that do not span time.
typedef struct retro_duty_cycle_event {
    int32_t kind; // one of RETRO_DUTY_CYCLE_EVENT_*
    uint64_t micros;
} retro_duty_cycle_event;

enum retro_duty_cycle_event_kind {
    // `micros` of synchronized GPU work, accounted at its completion.
    RETRO_DUTY_CYCLE_EVENT_WORK = 0,
    // One idle boundary: repay what is owed, up to the internal sleep cap.
    RETRO_DUTY_CYCLE_EVENT_IDLE = 1,
    // `micros` of host time the limiter did not choose, between two accounted
    // windows - a progress callback, a run-control pause.
    RETRO_DUTY_CYCLE_EVENT_HOST = 2,
    // An explicit window reset.
    RETRO_DUTY_CYCLE_EVENT_RESET = 3,
};

// Outcome of a retro_probe_duty_cycle replay.
typedef struct retro_duty_cycle_probe {
    retro_duty_cycle_stats stats;
    // Fake-clock reads and fake-sleeper calls. A disabled replay must leave
    // both at zero: that is what "no overhead when the limiter is off" means at
    // this level, and it is not observable from the seconds alone.
    uint64_t clock_reads;
    uint64_t sleep_calls;
} retro_duty_cycle_probe;

// Replays `events` through the duty-cycle controller on a fake clock, without a
// model, a device or real time. This is the arithmetic probe for the debt rule,
// the coalescing floor, the sleep cap and the stale-window rule; it drives the
// same code the trainer does.
// `fraction` follows the retro_trainer_set_max_gpu_duty_cycle contract, and the
// replay behaves as if the backend were a GPU. Returns -1 on an invalid
// fraction, a null argument or an unknown event kind.
int retro_probe_duty_cycle(
    float fraction,
    const retro_duty_cycle_event * events,
    size_t n_events,
    retro_duty_cycle_probe * out_probe);

// Aggregate forward-only evaluation result. `negative_log_likelihood` is the
// sum over every non-ignored SFT label; divide it by `supervised_tokens` for
// mean cross-entropy loss.
typedef struct retro_eval_metrics {
    double negative_log_likelihood;
    uint64_t supervised_tokens;
} retro_eval_metrics;

// Fixed-size SFT rows. `labels` mirrors `tokens`; a label value of -1 is
// ignored by the loss. Both buffers contain n_rows*n_ctx int32 values.
typedef struct retro_sft_dataset {
    const int32_t * tokens;
    const int32_t * labels;
    size_t n_rows;
    uint32_t n_ctx;
} retro_sft_dataset;

// Sampling parameters for rollout generation. Temperature and top_p shape the
// sampling distribution only; reported logprobs always describe the raw
// (temperature 1) model distribution, which is the policy PPO optimizes.
typedef struct retro_sampling_params {
    float temperature;       // must be finite and > 0
    float top_p;             // must be in (0, 1]
    uint32_t max_new_tokens; // must be > 0
    uint32_t seed;           // seeds the sampler; same seed => same rollout
} retro_sampling_params;

// One teacher-forced completion. `n_prompt` is the first completion target
// index, exactly as for retro_trainer_score_token_suffix().  A batch is
// required to share the same prompt tokens so the runtime can decode that
// prefix once and branch its KV state to every completion.
typedef struct retro_token_suffix_sequence {
    const int32_t * tokens;
    size_t n_tokens;
    size_t n_prompt;
} retro_token_suffix_sequence;

// One independently sampled prompt for continuous generation. Rows may have
// different prompt lengths and sampling seeds; output order follows this
// array exactly.
typedef struct retro_generation_sequence {
    const int32_t * prompt_tokens;
    size_t n_prompt;
    retro_sampling_params sampling;
} retro_generation_sequence;

// Fixed-size weighted training rows. `labels` mirrors `tokens` (-1 = ignored);
// `weights` scales each label position's cross-entropy contribution and 0
// masks the position. With detached per-token coefficients the resulting
// gradient, weight * (softmax - one_hot), is exactly the clipped PPO
// policy-gradient, so this is the runtime's differentiable RL objective.
//
// retro delta (plan DISTILL D6.5): `n_topk` targets per position instead of one.
// `labels` and `weights` are then n_rows * n_ctx * n_topk values, entry j of
// position p of row r at ((r*n_ctx + p)*n_topk + j); the weights are the
// teacher's renormalized probabilities for that position. n_topk = 0 and
// n_topk = 1 are the same one-target layout, and produce the same run bit for
// bit. n_topk must not exceed RETRO_FUSED_CE_K_MAX.
typedef struct retro_weighted_dataset {
    const int32_t * tokens;  // n_rows * n_ctx values
    const int32_t * labels;  // n_rows * n_ctx * max(n_topk,1) values, -1 = ignored
    const float   * weights; // n_rows * n_ctx * max(n_topk,1) values, finite; 0 = ignored
    size_t n_rows;
    uint32_t n_ctx;
    uint32_t n_topk; // retro delta (DISTILL D6.5): 0/1 = one target per position
} retro_weighted_dataset;

// One physical optimizer micro-batch containing several teacher-forced
// sequences. Sequence membership uses CSR: token i belongs to the non-empty
// slice seq_ids[seq_offsets[i]..seq_offsets[i + 1]]. A prompt token can thus
// belong to every completion branch in its group while unrelated groups stay
// causally isolated. Positions are explicit because branches reuse positions.
typedef struct retro_packed_sequence_batch {
    const int32_t * tokens;     // n_tokens values
    const int32_t * labels;     // n_tokens * max(n_topk,1) values, -1 = ignored
    const float   * weights;    // n_tokens * max(n_topk,1) values, finite; 0 = ignored
    const int32_t * positions;  // n_tokens values
    const size_t  * seq_offsets;// n_tokens + 1 values; first=0, last=n_seq_ids
    const int32_t * seq_ids;    // n_seq_ids values
    size_t n_tokens;
    size_t n_seq_ids;
    uint32_t n_sequences;
    // retro delta (plan DISTILL D6.5): same layout and same meaning as in
    // retro_weighted_dataset. 0/1 = one target per position.
    uint32_t n_topk;
} retro_packed_sequence_batch;

// Return false to stop after the current safe training boundary. Optimizer
// step callbacks can request a stop; the active epoch finishes before return.
typedef bool (*retro_train_progress_callback)(
    uint32_t epoch,
    const retro_train_metrics * metrics,
    void * user_data);

retro_trainer * retro_trainer_new(
    const char * model_path,
    const retro_train_config * train_config);

// Process-wide runtime policy, fixed *before* any backend context exists.
// Versioned by struct_size rather than by adding fields
// to retro_train_config, whose size is part of the existing ABI: a caller
// compiled against an older header keeps working because the runtime reads only
// the prefix that caller's struct_size covers.
// This is what makes the RIR policy a public option instead of RETRO_RIR_MODE.
typedef struct retro_runtime_config {
    uint32_t struct_size;  // filled by the caller: the evolution key
    // The retro_rir_mode enum. Takes precedence over RETRO_RIR_MODE.
    int32_t  rir_mode;
} retro_runtime_config;

// Mirror of ggml_rir_mode. `off` is the default, so RIR stays opt-in.
typedef enum retro_rir_mode {
    RETRO_RIR_MODE_OFF     = 0,
    RETRO_RIR_MODE_OBSERVE = 1,  // decide and count, always run native
    RETRO_RIR_MODE_PREFER  = 2,  // run RIR when the contract matches
    RETRO_RIR_MODE_REQUIRE = 3,  // like prefer, and the preflight fails if it did not apply
} retro_rir_mode;

// retro_trainer_new plus the runtime policy. `runtime_config` may be NULL, in
// which case this is exactly retro_trainer_new.
// Fails, rather than silently ignoring the request, when a policy differs from
// one already in force: the backends have then already built (or not built) the
// pipelines it asks for, and a mode that disagrees with what was compiled is the
// one way retro_kernel_run_info.executed_impl could become untrustworthy.
// Requesting the policy already in force always succeeds.
retro_trainer * retro_trainer_new_ex(
    const char * model_path,
    const retro_train_config * train_config,
    const retro_runtime_config * runtime_config);

// The policy actually in force, and whether it can still be changed. A caller
// that passed a runtime_config can check what it got rather than assume.
int retro_runtime_config_effective(retro_runtime_config * out, bool * out_latched);

// Applies the policy on its own, without creating a trainer. Same rules as
// retro_trainer_new_ex: idempotent for the value already in force, an error for
// a different one once the policy has been read. Exists so a host can set the
// policy at startup, and so the refusal is testable without a model.
int retro_runtime_config_apply(const retro_runtime_config * config);

int retro_trainer_create_lora(
    retro_trainer * trainer,
    const retro_lora_config * lora_config);

int retro_trainer_load_lora(
    retro_trainer * trainer,
    const char * adapter_path);

// Declares the resolved base trainable set, by canonical tensor name.
//
// The selection itself is resolved outside the runtime, from the GGUF's tensor
// inventory; this call is where the answer arrives, and it is validated against
// the loaded model: an unknown name is an error, not an empty selection.
// The names are copied, so the caller's buffers need only outlive the call.
//
// Must be called before the optimizer context exists, and only for a run whose
// retro_train_config.trainable is not RETRO_TRAINABLE_LORA. Passing zero names
// to a LoRA run is a no-op; passing any is an error.
int retro_trainer_set_trainable_base(
    retro_trainer * trainer,
    const char * const * names,
    size_t n_names);

// Declares which optimizer owns each marked parameter, by canonical name.
// A parameter no row names is owned by the run's own optimizer, so an empty
// table is the single-optimizer run.
//
// The names are copied; `optimizers[i]` must be one of retro_optimizer that
// this build can build a step for. Call before the optimizer context exists;
// a row naming a parameter this run does not train is refused once the marked
// set is known.
int retro_trainer_set_optimizer_assignment(
    retro_trainer * trainer,
    const char * const * names,
    const int32_t * optimizers,
    size_t n_rows);

// Tokenizes `text` with the model vocabulary (BOS/special tokens added and
// parsed). On success writes up to n_tokens_max ids and sets *out_n_tokens to
// the count. If the buffer is too small, sets *out_n_tokens to the required
// count and returns -2; a NULL/zero buffer only reports that size and returns 0.
int retro_trainer_tokenize_text(
    retro_trainer * trainer,
    const char * text,
    int32_t * tokens,
    size_t n_tokens_max,
    size_t * out_n_tokens);

// Same as retro_trainer_tokenize_text, but adds no BOS/EOS of its own: for a
// fragment that continues a token stream someone else opened. Special tokens
// spelled out in the text are still parsed. A multi-turn prompt is assembled
// out of these - the framing a chat template puts around the turns the policy
// sampled - because re-tokenizing a sampled turn does not in general give back
// the tokens that were sampled.
int retro_trainer_tokenize_fragment(
    retro_trainer * trainer,
    const char * text,
    int32_t * tokens,
    size_t n_tokens_max,
    size_t * out_n_tokens);

int retro_trainer_eos_token(retro_trainer * trainer, int32_t * out_token);
int retro_trainer_vocab_size(retro_trainer * trainer, uint32_t * out_n_vocab);
// True when the token ends generation for this model. Generation stops on any
// end-of-generation token, not only the canonical EOS, so callers that must
// tell a natural stop from a budget cutoff have to ask the vocabulary.
int retro_trainer_is_eog_token(retro_trainer * trainer, int32_t token, bool * out_is_eog);
int retro_trainer_context_size(retro_trainer * trainer, uint32_t * out_n_ctx);

// Format messages using the GGUF's `tokenizer.chat_template`. The output uses
// the two-call buffer convention used by the report APIs.
int retro_trainer_format_chat(
    retro_trainer * trainer,
    const char * const * roles,
    const char * const * contents,
    size_t n_messages,
    bool add_assistant,
    char * buffer,
    size_t n_buffer,
    size_t * out_n_bytes);

// Same renderer, with the messages given as a JSON array so a message can carry
// the structured fields a chat template expects - `tool_call_id`, `name`, and
// `tool_calls` - with the tool catalog passed as `tools` in OpenAI function
// shape. `tools_json` may be NULL. Uses the same two-call buffer contract.
int retro_trainer_format_chat_messages(
    retro_trainer * trainer,
    const char * messages_json,
    const char * tools_json,
    bool add_assistant,
    char * buffer,
    size_t n_buffer,
    size_t * out_n_bytes);

// Sets JSON chat-template variables and invalidates the tool-support probe.
// NULL, an empty string or "{}" clears them. Rejects non-objects and reserved
// keys: messages, tools, bos_token, eos_token, add_generation_prompt.
int retro_trainer_set_chat_template_variables(
    retro_trainer * trainer,
    const char * variables_json);

// Whether this model's chat template renders a tool catalog of its own. Probed
// once by rendering with a sentinel tool: a template that mentions `tools` and
// then drops it reports false, since what matters is only whether the catalog
// reaches the text. Callers use it to choose between native tool rendering and
// describing the tools in the system prompt.
int retro_trainer_chat_template_supports_tools(
    retro_trainer * trainer,
    bool * out_supports);

// Derives a parser for this model's own tool-call format from its chat template
// and writes it serialized using the renderer buffer contract. This is the
// reading counterpart to
// retro_trainer_format_chat_messages does for writing: a template that renders
// a catalog teaches the model a call format, and that format is the template's,
// not a convention the caller may assume.
// `tools_json` is the OpenAI-shaped catalog, or NULL for "no tools"; the
// generated grammar may name the functions, so the parser is only valid for the
// catalog it was built from. Returns RETRO_CHAT_PARSER_UNAVAILABLE when a valid
// template yields no parser - the only result callers may answer with a
// prompt-described fallback. Invalid input and runtime failures return -1.
#define RETRO_CHAT_PARSER_UNAVAILABLE (-3)
int retro_trainer_tool_call_parser(
    retro_trainer * trainer,
    const char * tools_json,
    char * buffer,
    size_t n_buffer,
    size_t * out_n_bytes);

// Model-free counterparts of retro_trainer_format_chat and
// retro_trainer_format_chat_messages, using a chat template supplied as Jinja
// source. They open no model and allocate no context, which
// is what lets the agreement between what a template renders and what its
// derived parser reads be tested on template fixtures alone. `{{ bos_token }}`
// and `{{ eos_token }}` render empty, there being no vocabulary.
int retro_chat_template_render(
    const char * template_src,
    const char * messages_json,
    const char * tools_json,
    bool add_assistant,
    char * buffer,
    size_t n_buffer,
    size_t * out_n_bytes);

int retro_chat_template_tool_call_parser(
    const char * template_src,
    const char * tools_json,
    char * buffer,
    size_t n_buffer,
    size_t * out_n_bytes);

// Parses one assistant output with a parser produced by the parser functions.
// Takes no retro_trainer and opens no model: the blob is self-sufficient, so this may be
// called from any thread while the trainer is busy generating.
// Writes a JSON object shaped like:
//   {"content": "...", "reasoning_content": "...",
//    "tool_calls": [{"id": "...", "name": "...", "arguments": "{...}"}]}
// `arguments` is the raw string llama.cpp produced, left for the caller to
// deserialize so a malformed one stays reportable rather than fatal.
int retro_chat_parse_assistant(
    const char * parser_blob,
    size_t n_parser_blob,
    const char * text,
    char * buffer,
    size_t n_buffer,
    size_t * out_n_bytes);

int retro_trainer_describe_lora(
    retro_trainer * trainer,
    char * buffer,
    size_t n_buffer,
    size_t * out_n_bytes);

// Lists every tensor a LoRA adapter can attach to, one name per line, sorted.
// Layer indices are kept (unlike the wildcard families the capability report
// suggests), so a caller can resolve a concrete target instead of assuming a
// layout: on a hybrid architecture such as lfm2 or falcon-h1, blk.0 is not
// necessarily an attention block. Safe to call before a LoRA adapter exists.
// Same buffer contract as retro_trainer_describe_lora().
int retro_trainer_lora_candidate_targets(
    retro_trainer * trainer,
    char * buffer,
    size_t n_buffer,
    size_t * out_n_bytes);

// Reports the backend placement for this trainer: the selected/effective device,
// whether GPU offload is active, and how many model and LoRA tensors live on each
// backend buffer type. Same buffer contract as retro_trainer_describe_lora().
int retro_trainer_backend_report(
    retro_trainer * trainer,
    char * buffer,
    size_t n_buffer,
    size_t * out_n_bytes);

// Reports the loaded model architecture, tensor types, target-profile status,
// requested backend, and the state of training-graph compatibility checks.
// This is safe to call before a LoRA adapter is created. Same buffer contract
// as retro_trainer_describe_lora().
int retro_trainer_capability_report(
    retro_trainer * trainer,
    char * buffer,
    size_t n_buffer,
    size_t * out_n_bytes);

int retro_trainer_model_capabilities(
    retro_trainer * trainer,
    retro_model_capabilities * out_capabilities);

// Builds the exact training graph (forward, backward, optimizer step) for a
// representative micro-batch without running it and reports, per registered
// backend device, every op the device cannot execute plus every op that has
// no gradient rule at all. Requires a freshly created LoRA adapter (the
// trainable parameters define the backward graph). The same preflight runs
// automatically before the first optimizer step and turns what would be a
// process abort into a precise error. Same buffer contract as
// retro_trainer_describe_lora().
int retro_trainer_train_preflight(
    retro_trainer * trainer,
    char * buffer,
    size_t n_buffer,
    size_t * out_n_bytes);

int retro_trainer_preflight_summary(
    retro_trainer * trainer,
    retro_preflight_summary * out_summary);

int retro_trainer_train_tokens(
    retro_trainer * trainer,
    const int32_t * tokens,
    size_t n_tokens,
    retro_train_metrics * out_metrics);

int retro_trainer_train_sft(
    retro_trainer * trainer,
    const retro_sft_dataset * train,
    const retro_sft_dataset * eval,
    retro_train_metrics * out_metrics,
    retro_train_progress_callback progress_callback,
    void * progress_user_data);

// Evaluates fixed-size SFT rows without creating an optimizer or requiring a
// LoRA adapter. Labels equal to -1 are ignored. The currently active adapter,
// when any, is applied by the inference context.
int retro_trainer_eval_sft(
    retro_trainer * trainer,
    const retro_sft_dataset * data,
    retro_eval_metrics * out_metrics);

// Samples `n_sequences` completions from one prompt with the current model +
// LoRA weights. The prompt is decoded once, its memory is copied to one llama
// sequence per member, and all live members advance in one decode batch per
// token step. Each member owns its sampling params and RNG seed.
// Outputs are row-major with a stride of n_out_max. Generation stops per row at
// max_new_tokens, at an end-of-generation token (included in the output), or
// when the per-sequence context fills. `out_logprobs` may be NULL; otherwise it
// receives temperature-1 policy logprobs with the same layout. `out_n_tokens`
// must hold n_sequences counts.
int retro_trainer_generate_batch(
    retro_trainer * trainer,
    const int32_t * prompt_tokens,
    size_t n_prompt,
    const retro_sampling_params * sampling,
    size_t n_sequences,
    int32_t * out_tokens,
    float * out_logprobs,
    size_t n_out_max,
    size_t * out_n_tokens);

// Continuous counterpart of retro_trainer_generate_batch() for heterogeneous
// prompts. Every input produces one completion; output is row-major with the
// supplied stride and is deterministically split by the caller when capacity
// is exceeded.
int retro_trainer_generate_continuous_batch(
    retro_trainer * trainer,
    const retro_generation_sequence * sequences,
    size_t n_sequences,
    int32_t * out_tokens,
    float * out_logprobs,
    size_t n_out_max,
    size_t * out_n_tokens);

// Teacher-forced scoring with the current model + LoRA weights: writes
// log p(tokens[i+1] | tokens[0..=i]) into out_logprobs[i] for i in
// [0, n_tokens-1). Forward only; the optimizer state is untouched.
int retro_trainer_score_tokens(
    retro_trainer * trainer,
    const int32_t * tokens,
    size_t n_tokens,
    float * out_logprobs);

// Teacher-forced scoring for completion targets only. `n_prompt` is the index
// of the first completion token. Prefix states are decoded without retaining
// logits and output is bounded to n_tokens - n_prompt entries.
int retro_trainer_score_token_suffix(
    retro_trainer * trainer,
    const int32_t * tokens,
    size_t n_tokens,
    size_t n_prompt,
    float * out_logprobs);

// Teacher-forced truncated distribution over completion targets: for each of
// the n_tokens - n_prompt target positions, the `k` most probable next tokens
// and their temperature-1 log-probabilities, in decreasing probability order
// with ties broken by ascending id. Output is row-major, `k` entries per
// position, into both arrays; `n_out_max` is the capacity of each.
//
// This is the one scoring entry point that returns more than the value of a
// token the caller already had. It reads the same teacher-forced forward pass
// retro_trainer_score_token_suffix() runs, on the host: the reduction needs the
// whole logits row, so the in-graph target gather does not apply. For k = 1 the
// returned id is the row's argmax and its log-probability is what
// retro_trainer_score_token_suffix() returns when handed that same id.
int retro_trainer_top_logprobs_suffix(
    retro_trainer * trainer,
    const int32_t * tokens,
    size_t n_tokens,
    size_t n_prompt,
    size_t k,
    int32_t * out_ids,
    float * out_logprobs,
    size_t n_out_max);

// Scores several completions of one shared prompt. Results are row-major with
// `out_stride` entries per input row; only the first `out_n_logprobs[row]`
// values of each row are written. Input and output row order are identical.
int retro_trainer_score_token_suffix_batch(
    retro_trainer * trainer,
    const retro_token_suffix_sequence * sequences,
    size_t n_sequences,
    float * out_logprobs,
    size_t out_stride,
    size_t * out_n_logprobs);

// Enables or disables the trainer's LoRA on the inference context. This is
// the primitive behind fixed-reference (base model) scoring; callers must
// restore the adapter afterwards.
int retro_trainer_set_lora_enabled(retro_trainer * trainer, bool enabled);

// Reports the model's output hidden-state width: the number of features per
// position written by retro_trainer_hidden_states().
int retro_trainer_hidden_size(retro_trainer * trainer, uint32_t * out_n_embd);

// Teacher-forced feature extraction with the current model + LoRA weights:
// writes the final-layer hidden state of every position into out_features
// (n_tokens rows of hidden-size floats, row-major). Forward only; this feeds
// the value head (critic) that estimates per-position returns.
int retro_trainer_hidden_states(
    retro_trainer * trainer,
    const int32_t * tokens,
    size_t n_tokens,
    float * out_features,
    size_t n_features_max);

// Combined teacher-forced scoring and critic feature extraction for completion
// positions. Writes n_tokens - n_prompt log-probabilities and the same number
// of final-layer hidden-state rows. The decoded states are exactly
// tokens[n_prompt-1..n_tokens-1), so both outputs come from one forward pass.
int retro_trainer_score_token_suffix_and_hidden_states(
    retro_trainer * trainer,
    const int32_t * tokens,
    size_t n_tokens,
    size_t n_prompt,
    float * out_logprobs,
    float * out_features,
    size_t n_features_max);

// Renders tokens back to text. unparse_special controls whether control tokens
// (template delimiters such as a model's own tool-call markers) are written
// out as their surface text or dropped: false for text meant to be read by a
// person, true for text a tool-call parser must see intact. Same two-call
// buffer contract as retro_trainer_describe_lora().
int retro_trainer_detokenize(
    retro_trainer * trainer,
    const int32_t * tokens,
    size_t n_tokens,
    bool unparse_special,
    char * buffer,
    size_t n_buffer,
    size_t * out_n_bytes);

// Runs one weighted optimization epoch over the rows (one AdamW step per
// batch). scheduler_total_steps pins the LR-schedule horizon across repeated
// calls; 0 extends the horizon to just the steps in this call. The scheduler
// step accumulates across calls so PPO can drive many short epochs through
// one trainer. The base model must stay frozen (error otherwise); unlike SFT,
// an all-clipped batch may legitimately leave the LoRA unchanged, so that is
// not an error here.
int retro_trainer_train_weighted(
    retro_trainer * trainer,
    const retro_weighted_dataset * data,
    uint64_t scheduler_total_steps,
    retro_train_metrics * out_metrics,
    retro_train_progress_callback progress_callback,
    void * progress_user_data);

// Runs one optimizer step over a multi-branch teacher-forced batch whose
// differentiable prompt prefix is evaluated once. Intended for GRPO groups;
// falls under the same scheduler/callback contract as train_weighted.
int retro_trainer_train_packed_sequences(
    retro_trainer * trainer,
    const retro_packed_sequence_batch * data,
    uint64_t scheduler_total_steps,
    uint32_t accumulation_steps,
    retro_train_metrics * out_metrics,
    retro_train_progress_callback progress_callback,
    void * progress_user_data);

// Returns monotonic optimizer timing counters accumulated since trainer
// creation. Snapshot this before and after an update to attribute its cost.
int retro_trainer_optimizer_timing(
    retro_trainer * trainer,
    retro_optimizer_timing * out_timing);

// Returns the device-memory high-water of the optimizer path, sampled by the
// runtime inside each step (after graph allocation and after evaluation). This is
// the only accounting that sees the backends' own scratch: the CUDA pool and the
// Vulkan prealloc_* buffers belong to no ggml_backend_buffer, so the byte
// breakdown in backend_report cannot see them, and a host-side sampler between
// steps cannot either because the peak lives inside a single evaluation.
// `device_*` come from the backend's device budget and are therefore device-wide
// (other processes included): compare deltas, not absolute values.
// `scratch_*` are this process's backends and are attributable. Every field is
// zero without an active non-CPU device; `n_samples == 0` distinguishes
// "unavailable" from "measured zero".
int retro_trainer_optimizer_memory(
    retro_trainer * trainer,
    retro_optimizer_memory * out_memory);

// Returns monotonic behavior-scoring counters accumulated since trainer
// creation. Snapshot before and after an update to attribute it, the same way
// retro_trainer_optimizer_timing attributes the optimizer.
int retro_trainer_scoring_stats(
    retro_trainer * trainer,
    retro_scoring_stats * out_stats);

// Bounds the fraction of wall time this trainer spends with GPU work submitted
// and in flight, so another workload can use the device in between. `fraction`
// must be finite and in (0, 1]; `1.0f` means no limit and clears every window
// and counter the limiter holds.
//
// This releases *compute* time, not device memory: weights, KV caches, retained
// activations and optimizer state stay allocated while the trainer sleeps.
//
// It is deliberately not a field of retro_train_config. That structure has a
// published layout guarded by exact size and offset tests, and an execution
// policy that can be changed between two operations does not belong in the
// document that describes the training problem.
//
// Safe to call between trainer operations, never during one. On a CPU backend
// the value is retained but the limiter stays inactive - retro_trainer_report
// and retro_trainer_duty_cycle_stats both say so rather than promise throttling
// a CPU run cannot deliver.
//
// The enabled path synchronizes each decode it accounts, which costs the
// overlap between host-side sampling and the previous decode. Crossing from
// disabled to enabled therefore has a price that no fraction below 1.0 avoids:
// 0.99 is not approximately 1.0, and the setting is worth its overhead at 0.75
// and below.
int retro_trainer_set_max_gpu_duty_cycle(
    retro_trainer * trainer,
    float fraction);

// Snapshots the duty-cycle limiter's wall-clock accounting. Never fails on a
// trainer that has throttled nothing: the seconds are simply zero.
int retro_trainer_duty_cycle_stats(
    retro_trainer * trainer,
    retro_duty_cycle_stats * out_stats);

// Snapshots the generation context's prefix-reuse counters. Never fails on a
// trainer that has generated nothing: the counters are simply zero.
int retro_trainer_generation_stats(
    retro_trainer * trainer,
    retro_generation_stats * out_stats);

// Advances the logical learning-rate scheduler without running an optimizer
// step. GRPO uses this for filtered rollout slots so warm-up and decay remain
// aligned with the configured horizon while avoiding a zero-gradient batch.
int retro_trainer_advance_scheduler_steps(
    retro_trainer * trainer,
    uint64_t steps,
    uint64_t * out_global_step);

// Replaces the *base* learning rate the schedule is applied to, between two
// optimizer steps. The scheduler multiplies this value by its warm-up or decay
// factor at every step, so the next step uses the new base and the shape of the
// schedule is unchanged. Nothing else is touched: the step counter, the horizon
// and the persistent optimizer state keeps its values, which is what makes this
// safe to call
// from a progress callback.
// Rejects a non-finite or non-positive rate, exactly as trainer creation does.
int retro_trainer_set_learning_rate(
    retro_trainer * trainer,
    float learning_rate);

int retro_trainer_save_lora(
    retro_trainer * trainer,
    const char * adapter_path);

// Writes the resolved base trainable tensors to a GGUF, by absolute value.
//
// Absolute and not a delta against the source model: a delta is only
// interpretable beside the exact GGUF it was taken from, while the values are
// what the run produced and what a reload needs. The file is the trainable
// half of a checkpoint bundle, never a LoRA adapter - it carries no adapter
// metadata and llama_adapter_lora_init() would refuse it.
//
// Fails for a run that trains no base tensor: an empty bundle claims a run
// trained something it did not.
int retro_trainer_save_trainable(
    retro_trainer * trainer,
    const char * trainable_path);

// Restores those values onto the live model, matching by tensor name.
//
// Every name in the file must be in the run's resolved set and every name in
// the set must be in the file, with the same shape and dtype: a partial
// restore would resume from a model that is neither the checkpoint's nor the
// base's. Call before the first training step - the values it writes are
// weights, not optimizer state, and the optimizer graph does not have to exist.
int retro_trainer_load_trainable(
    retro_trainer * trainer,
    const char * trainable_path);

// Writes the whole loaded model back out as a standalone GGUF, trained weights
// included, in the dtypes it was loaded in. Unlike the bundle above, the result
// needs neither the source model nor this loader.
//
// Refused when the run trains no base tensor, when an adapter is loaded (merging
// it into the weights is not implemented here), or when a weight is on a buffer
// the host cannot read.
//
// The architecture is the caller's check via retrograd_core::architecture_exports_model.
int retro_trainer_save_model(
    retro_trainer * trainer,
    const char * model_path);

// ---------------------------------------------------------------------------
// Training checkpoints
// The GGUF written by retro_trainer_save_lora() stays a pure adapter export.
// Everything a resume needs beyond the adapter weights is read and written
// through the checkpoint calls, which never serialize anything themselves: the
// caller owns the on-disk format.
// ---------------------------------------------------------------------------

// Optimizer and learning-rate-schedule scalars.
typedef struct retro_optimizer_state {
    // AdamW bias-correction counter (ggml starts it at 1).
    int64_t iter;
    // Whether this optimizer has allocated state slots.
    // False for a cold optimizer and false for SGD, which keeps none.
    bool has_persistent_state;
    // Whether the optimizer graph exists, i.e. whether
    // retro_trainer_prepare_optimizer() or a training step has run.
    //
    // Distinct from has_persistent_state because "initialized with zero slots" and
    // "not initialized" are different states: an SGD run has no slots and
    // still has an iteration counter, a schedule and an RNG state, and a
    // resume that read the empty slot list as a cold optimizer would restart
    // the schedule from zero.
    bool graph_ready;
    // One of retro_optimizer.
    int32_t optimizer;
    float learning_rate;
    float weight_decay;
    float max_grad_norm;
    uint64_t scheduler_step;
    uint64_t scheduler_total_steps;
    float last_learning_rate;
    // AdamW's own coefficients, as the update step reads them. Reported, not
    // configured: nothing in retro_train_config sets them, so a checkpoint
    // records the values that ran. Meaningless for optimizers without such a
    // knob (SGD).
    float adamw_beta1;
    float adamw_beta2;
    float adamw_eps;
    // One of retro_gefen_variant, and meaningless for any other optimizer. The
    // name "gefen" does not say which slot table was allocated - the two
    // variants are two layouts - so a caller rebuilding the state table needs
    // both values or it compares its plan against the wrong one.
    int32_t gefen_variant;
    // The chosen optimizer's own coefficients, the same way the AdamW three
    // above are: reported, not configured, so a checkpoint records the values
    // that ran rather than the ones a document asked for. Meaningless for an
    // optimizer that declares none of them.
    float muon_momentum;
    float muon_ns_epsilon;
    float muon_fallback_learning_rate;
    uint32_t muon_ns_steps;
    bool muon_nesterov;
    float gefen_beta1;
    float gefen_beta2;
    float gefen_eps;
    uint32_t gefen_block_size;
} retro_optimizer_state;

// Reads the optimizer and scheduler scalars. Never fails for a live trainer.
int retro_trainer_optimizer_state(
    retro_trainer * trainer,
    retro_optimizer_state * out_state);

// Restores the resumable scalars: iter, scheduler_step, scheduler_total_steps
// and last_learning_rate. The hyperparameters are owned by the run
// configuration and are only reported here for the caller's compatibility
// check; they are ignored on restore.
int retro_trainer_restore_optimizer_state(
    retro_trainer * trainer,
    const retro_optimizer_state * state);

// Declares that `completed_epochs` SFT epochs are already done, so the next
// retro_trainer_train_sft() starts there and keeps the scheduler step restored
// by retro_trainer_restore_optimizer_state() instead of resetting it. The
// schedule horizon stays the full configured run, so warm-up and decay follow
// the trajectory the first launch started. Rollout algorithms do not need this:
// their weighted epochs already accumulate the scheduler step across calls.
int retro_trainer_set_resume_point(retro_trainer * trainer, uint32_t completed_epochs);

// Builds the optimizer graph if it does not exist yet, so any per-parameter
// state is allocated and can be written before the first training step.
// Requires a created or loaded LoRA adapter unless the run trains base weights.
//
// Succeeds for an optimizer that keeps no state at all: it reports that the
// graph exists, not that any slot was allocated.
int retro_trainer_prepare_optimizer(retro_trainer * trainer);

// Every tensor carrying GGML_TENSOR_FLAG_PARAM once the optimizer graph
// exists: the set the update step actually writes, whatever the policy
// selected and whatever the optimizer keeps for it.
//
// Distinct from the slot enumeration below, and both are needed. A parameter
// with no slot is still a parameter - SGD keeps none at all - so a checkpoint
// that listed only the parameters carrying state could not record which
// optimizer owns each one, and a resume could not compare selections.
//
// Ordered adapter factors first, in the adapter's registration order, then
// base tensors by name: the same order the resolved trainable set publishes,
// so the two can be compared line by line.
int retro_trainer_marked_parameter_count(retro_trainer * trainer, size_t * out_count);

// Describes one marked parameter, reusing the inventory's per-tensor contract.
int retro_trainer_marked_parameter_info(
    retro_trainer * trainer,
    size_t index,
    retro_tensor_desc * out_tensor);

// The three reads below expose every input and output of one update step: the
// parameter before and after it, the gradient it multiplied, and (via the slot
// enumeration further down) any persistent state. An optimizer that keeps no
// slot (SGD) has no other input to check its arithmetic against.
//
// Like the slot reads, these are byte ranges: callers stream through bounded
// staging, and a range past the end is an error, not a short read.

// Copies `n_bytes` of one marked parameter, starting at `offset`.
int retro_trainer_marked_parameter_read(
    retro_trainer * trainer,
    size_t index,
    uint64_t offset,
    void * out_bytes,
    size_t n_bytes);

// Describes the gradient accumulator of one marked parameter, by the same
// index. Always F32 and parameter-shaped, whatever the parameter's own dtype.
//
// The accumulators outlive individual graph allocations, so the gradient a
// step consumed is readable after that step returned. It holds what the last
// backward accumulated: one micro-batch's gradient at opt_period 1, the
// period's sum otherwise.
int retro_trainer_parameter_gradient_info(
    retro_trainer * trainer,
    size_t index,
    retro_tensor_desc * out_tensor);

// Copies `n_bytes` of that accumulator, starting at `offset`.
int retro_trainer_parameter_gradient_read(
    retro_trainer * trainer,
    size_t index,
    uint64_t offset,
    void * out_bytes,
    size_t n_bytes);

// Persistent optimizer state, enumerated as slots rather than as AdamW pairs.
//
// A slot is one persistent tensor an optimizer keeps: AdamW has "m" and "v"
// per parameter, SGD has none, and an optimizer with block-shaped or shared
// state has whichever its layout declares. The enumeration is the contract -
// nothing here assumes two slots, assumes they are floats, or assumes they are
// parameter-shaped, because the next optimizer breaks all three.
//
// Scopes are separate namespaces: a parameter slot is owned by the trainable
// tensor it updates, a shared slot by whatever the optimizer declares as its
// owner (a codebook belongs to the optimizer, not to any one parameter), and a
// shared slot is allocated once per owner rather than once per parameter.
#define RETRO_SLOT_SCOPE_PARAMETER 0
#define RETRO_SLOT_SCOPE_SHARED    1

// Slot initializers, mirroring the optimizer descriptor's declaration.
#define RETRO_SLOT_INIT_ZERO             0
#define RETRO_SLOT_INIT_CODE             1
#define RETRO_SLOT_INIT_UNIFORM_CODEBOOK 2

// Slot storage; a byte index is no weight's dtype.
#define RETRO_SLOT_DTYPE_F32 0
#define RETRO_SLOT_DTYPE_I8  1

// The bytes a slot of `n_elements` holds before the first update: the same
// function the runtime fills a live slot with, reachable without a model.
// `out_bytes` must be exactly `n_elements` elements of the slot's dtype;
// anything else is an error, not a partial fill.
int retro_optimizer_slot_initial_bytes(
    int32_t dtype,
    int32_t init,
    uint8_t code,
    uint64_t n_elements,
    void * out_bytes,
    size_t n_bytes);

typedef struct retro_optimizer_slot {
    // Trainable tensor name for a parameter slot, optimizer-declared owner for
    // a shared one. The pair (owner, slot) is the identity a restore matches
    // on; the index is an enumeration order and never an identity.
    char owner[RETRO_TENSOR_NAME_MAX];
    // Slot name inside the optimizer's layout: "m", "v", "momentum", ...
    char slot[RETRO_MODEL_INFO_NAME_MAX];
    // ggml type name ("F32", "I8", ...), the spelling retro_tensor_desc uses.
    char type_name[RETRO_MODEL_INFO_NAME_MAX];
    int64_t ne[4];
    uint64_t n_elements;
    // Exact payload length. Reads and writes are byte ranges inside it, so a
    // caller streams through a bounded staging buffer instead of holding a
    // model-sized copy of the state.
    uint64_t n_bytes;
} retro_optimizer_slot;

// How many slots each scope currently holds. Both are zero until the optimizer
// graph is built, and the parameter count stays zero for an optimizer that
// keeps no per-parameter state - which is a fact about the optimizer, not a
// sign that nothing was initialized. retro_optimizer_state.graph_ready is what
// tells those apart. Either out pointer may be null.
int retro_trainer_state_slot_count(
    retro_trainer * trainer,
    size_t * out_parameter_slots,
    size_t * out_shared_slots);

// Describes one slot of `scope`, by enumeration index.
int retro_trainer_state_slot_info(
    retro_trainer * trainer,
    int32_t scope,
    size_t index,
    retro_optimizer_slot * out_slot);

// Copies `n_bytes` of one slot's payload out of the backend, starting at
// `offset`. A range past the end is an error, never a short read.
int retro_trainer_state_slot_read(
    retro_trainer * trainer,
    int32_t scope,
    size_t index,
    uint64_t offset,
    void * out_bytes,
    size_t n_bytes);

// Copies a byte range back into the slot identified by (owner, slot). Matching
// is by name, never by index, so a checkpoint stays valid across graph
// orderings. Fails when the pair is unknown or the range leaves the payload.
int retro_trainer_state_slot_write(
    retro_trainer * trainer,
    int32_t scope,
    const char * owner,
    const char * slot,
    uint64_t offset,
    const void * bytes,
    size_t n_bytes);

// Reads or restores the runtime's mt19937 state. Same two-call buffer contract
// as retro_trainer_describe_lora().
int retro_trainer_rng_state(
    retro_trainer * trainer,
    char * buffer,
    size_t n_buffer,
    size_t * out_n_bytes);
int retro_trainer_set_rng_state(retro_trainer * trainer, const char * state);

// Stable one-line signature of the loaded model (architecture and the shape
// hyperparameters a LoRA depends on), recorded in a checkpoint manifest and
// compared on resume. Same two-call buffer contract.
int retro_trainer_model_signature(
    retro_trainer * trainer,
    char * buffer,
    size_t n_buffer,
    size_t * out_n_bytes);

void retro_trainer_free(retro_trainer * trainer);

const char * retro_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
