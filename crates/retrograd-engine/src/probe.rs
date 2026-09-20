use super::*;
use crate::rir::kernel_run_info;

// FFI safety contract for this module: every input slice is checked against
// its declared shape before its pointer is passed, every output pointer names
// a live allocation of the advertised length, and probe calls are synchronous.

/// One shaped operand of a probe: the 4-element `[ne0, ne1, ne2, ne3]` shape and
/// the data laid out under it.
#[derive(Clone, Copy)]
pub struct ProbeOperand<'a> {
    pub ne: [i64; 4],
    pub data: &'a [f32],
}

/// The inputs of a probed op: two mandatory operands and the optional third the
/// ops that take one.
///
/// A struct rather than six positional arguments, because the shape and the
/// data of one operand are one thing - a call site that passed `ne_src1` beside
/// `src0` type-checked before, and does not now.
pub struct ProbeInputs<'a> {
    pub src0: ProbeOperand<'a>,
    pub src1: ProbeOperand<'a>,
    pub src2: Option<ProbeOperand<'a>>,
}

impl<'a> ProbeInputs<'a> {
    /// The two-operand form, which most probed ops take.
    pub fn pair(ne_src0: [i64; 4], src0: &'a [f32], ne_src1: [i64; 4], src1: &'a [f32]) -> Self {
        Self {
            src0: ProbeOperand {
                ne: ne_src0,
                data: src0,
            },
            src1: ProbeOperand {
                ne: ne_src1,
                data: src1,
            },
            src2: None,
        }
    }

    /// Adds the third operand, in the `(shape, data)` form the probe API took
    /// before the operands were named.
    #[must_use]
    pub fn with_src2(mut self, src2: Option<([i64; 4], &'a [f32])>) -> Self {
        self.src2 = src2.map(|(ne, data)| ProbeOperand { ne, data });
        self
    }
}

pub fn probe_op(
    op: ProbeOp,
    use_gpu: bool,
    inputs: ProbeInputs<'_>,
    params: [f32; 2],
    out_len: usize,
) -> Result<Vec<f32>> {
    let ProbeInputs { src0, src1, src2 } = inputs;
    let (ne_src0, src0) = (src0.ne, src0.data);
    let (ne_src1, src1) = (src1.ne, src1.data);
    let mut dst = vec![0.0_f32; out_len];
    let (ne_src2_ptr, src2_ptr) = match &src2 {
        Some(operand) => (operand.ne.as_ptr(), operand.data.as_ptr()),
        None => (std::ptr::null(), std::ptr::null()),
    };
    // SAFETY: the module contract validates input shapes and keeps every input/output allocation live.
    let code = unsafe {
        ffi::retro_probe_op_run(
            op.id(),
            if use_gpu { 1 } else { 0 },
            ne_src0.as_ptr(),
            src0.as_ptr(),
            ne_src1.as_ptr(),
            src1.as_ptr(),
            ne_src2_ptr,
            src2_ptr,
            params[0],
            params[1],
            dst.as_mut_ptr(),
            dst.len(),
        )
    };
    if code != 0 {
        return Err(runtime_error());
    }
    Ok(dst)
}

/// [`probe_op`] with an explicit implementation choice, returning both the
/// output and what actually ran.
///
/// [`KernelImpl::Rir`] fails rather than falling back: a probe that silently
/// ran the native kernel would report a green "RIR" test that never touched the
/// variant. [`KernelImpl::Native`] forces the native path for this run even
/// under `RETRO_RIR_MODE=prefer`, which is what makes the two sides of a parity
/// test independent within one process.
pub fn probe_op_ex(
    op: ProbeOp,
    use_gpu: bool,
    inputs: ProbeInputs<'_>,
    params: [f32; 2],
    out_len: usize,
    implementation: KernelImpl,
) -> Result<(Vec<f32>, KernelRunInfo)> {
    let ProbeInputs { src0, src1, src2 } = inputs;
    let (ne_src0, src0) = (src0.ne, src0.data);
    let (ne_src1, src1) = (src1.ne, src1.data);
    let mut dst = vec![0.0_f32; out_len];
    let (ne_src2_ptr, src2_ptr) = match &src2 {
        Some(operand) => (operand.ne.as_ptr(), operand.data.as_ptr()),
        None => (std::ptr::null(), std::ptr::null()),
    };
    let mut info = ffi::RetroKernelRunInfo::default();
    // SAFETY: the module contract validates input shapes and keeps every input/output allocation live.
    let code = unsafe {
        ffi::retro_probe_op_run_ex(
            op.id(),
            if use_gpu { 1 } else { 0 },
            ne_src0.as_ptr(),
            src0.as_ptr(),
            ne_src1.as_ptr(),
            src1.as_ptr(),
            ne_src2_ptr,
            src2_ptr,
            params[0],
            params[1],
            dst.as_mut_ptr(),
            dst.len(),
            implementation.id(),
            &mut info,
        )
    };
    if code != 0 {
        return Err(runtime_error());
    }
    Ok((dst, kernel_run_info(&info)))
}

/// The geometry of a fused-CE probe: the shape of the graph it builds, and how
/// the projection head is stored.
///
/// Named because thirteen positional arguments made a call site read
/// `N_EMBD, N_TOKENS, N_VOCAB, c, 0, w_type, use_gpu, …` - five widths in a row,
/// two of which are the tile count and the sequence chunk, and nothing at the
/// call site said which was which.
#[derive(Clone, Copy, Debug)]
pub struct FusedCeProbeShape {
    pub n_embd: usize,
    pub n_tokens: usize,
    pub n_vocab: usize,
    /// Column tiles the fused path splits the vocabulary into.
    pub n_tiles: usize,
    /// Sequence chunk, or `0` for the operator's own default.
    pub seq_chunk: usize,
    pub w_type: FusedCeWeightType,
}

/// The tensors a fused-CE probe reads.
#[derive(Clone, Copy)]
pub struct FusedCeProbeInputs<'a> {
    /// Hidden states, `[n_embd, n_tokens]`.
    pub h: &'a [f32],
    /// Projection head, `[n_embd, n_vocab]`.
    pub w: &'a [f32],
    /// `[n_topk, n_tokens]` column-major, so `[n_tokens]` is the one-target
    /// case and `n_topk` is read off the length rather than passed.
    pub targets: &'a [i32],
    /// `[n_topk, n_tokens]`, matching `targets`.
    pub weights: &'a [f32],
    /// Fixed `[n_vocab]` additive term applied before the softmax, or `None`.
    pub bias: Option<&'a [f32]>,
}

/// Runs the plan-03 fused cross-entropy parity probe: for one set of inputs it
/// computes both the current full-vocab training tail (real fork ggml ops) and
/// the fused/tiled operators that never materialize the full `[n_vocab,
/// n_tokens]` logits, returning both losses and both `grad_h`. Tensors are ggml
/// column-major (`ne0` fastest); `targets[i] < 0` marks a masked token. With
/// `w_type = Q8_0` the full path runs on the dequantized head (exact oracle)
/// while the fused path consumes the quantized head, so parity proves the
/// operator's on-the-fly dequantization.
///
/// `bias`, when `Some`, is a fixed `[n_vocab]` per-vocab additive term applied
/// to every logit before the softmax on both paths (full path via
/// `mul_mat(w, h) + bias`, fused path via the operator's native bias input),
/// this is the shape gemma4 builds when it has suppressed tokens
/// (`ADD(MUL_MAT(w, h), bias)`). It never receives a gradient.
pub fn fused_sparse_ce_probe(
    shape: FusedCeProbeShape,
    use_gpu: bool,
    inputs: FusedCeProbeInputs<'_>,
    grad_loss: f32,
) -> Result<FusedCeProbe> {
    fused_sparse_ce_probe_offloaded(shape, false, use_gpu, inputs, grad_loss)
}

/// Same probe as [`fused_sparse_ce_probe`], with the offload knob exposed:
/// `offload_h` places `grad_h` on top of `h`, reproducing the
/// in-place allocation the training graph gets from ggml-alloc when the fused CE
/// nodes are built with the flag. It is a pure memory knob - the loss and
/// `grad_h` this returns must be bit-identical with and without it - so the
/// point of the parameter is to prove the operators survive the aliasing.
pub fn fused_sparse_ce_probe_offloaded(
    shape: FusedCeProbeShape,
    offload_h: bool,
    use_gpu: bool,
    inputs: FusedCeProbeInputs<'_>,
    grad_loss: f32,
) -> Result<FusedCeProbe> {
    let FusedCeProbeShape {
        n_embd,
        n_tokens,
        n_vocab,
        n_tiles,
        seq_chunk,
        w_type,
    } = shape;
    let FusedCeProbeInputs {
        h,
        w,
        targets,
        weights,
        bias,
    } = inputs;
    assert_eq!(h.len(), n_embd * n_tokens, "h must be [n_embd, n_tokens]");
    assert_eq!(w.len(), n_embd * n_vocab, "w must be [n_embd, n_vocab]");
    // `k` is the layout, not a separate argument - a caller that gets them out of
    // step could not describe a consistent probe anyway.
    assert!(
        !targets.is_empty() && targets.len() % n_tokens == 0,
        "targets must be [n_topk, n_tokens]"
    );
    let n_topk = targets.len() / n_tokens;
    assert_eq!(
        weights.len(),
        targets.len(),
        "weights must have the shape of targets"
    );
    assert!(
        n_topk <= retrograd_core::FUSED_CE_K_MAX,
        "n_topk must not exceed {}",
        retrograd_core::FUSED_CE_K_MAX
    );
    if let Some(bias) = bias {
        assert_eq!(bias.len(), n_vocab, "bias must be [n_vocab]");
    }

    // The runtime takes its extents as `i32`. The asserts above constrain the
    // shapes against each other, not against that range, so the conversion is
    // where an over-large extent has to be refused.
    let extent = |value: usize, name: &'static str| -> Result<i32> {
        i32::try_from(value).map_err(|_| {
            Error::invalid(format!(
                "{name} = {value} does not fit the runtime's i32 extent"
            ))
        })
    };
    let n_embd_ffi = extent(n_embd, "n_embd")?;
    let n_tokens_ffi = extent(n_tokens, "n_tokens")?;
    let n_vocab_ffi = extent(n_vocab, "n_vocab")?;
    let n_topk_ffi = extent(n_topk, "n_topk")?;
    let n_tiles_ffi = extent(n_tiles, "n_tiles")?;
    let seq_chunk_ffi = extent(seq_chunk, "seq_chunk")?;

    let mut loss_full = 0.0_f32;
    let mut loss_fused = 0.0_f32;
    let mut grad_h_full = vec![0.0_f32; n_embd * n_tokens];
    let mut grad_h_fused = vec![0.0_f32; n_embd * n_tokens];
    let bias_ptr = bias.map_or(std::ptr::null(), |b| b.as_ptr());
    // SAFETY: the module contract validates input shapes and keeps every input/output allocation live.
    let code = unsafe {
        ffi::retro_fused_sparse_ce_probe(
            n_embd_ffi,
            n_tokens_ffi,
            n_vocab_ffi,
            n_topk_ffi,
            n_tiles_ffi,
            seq_chunk_ffi,
            i32::from(offload_h),
            w_type.id(),
            i32::from(use_gpu),
            h.as_ptr(),
            w.as_ptr(),
            targets.as_ptr(),
            weights.as_ptr(),
            bias_ptr,
            grad_loss,
            &mut loss_full,
            &mut loss_fused,
            grad_h_full.as_mut_ptr(),
            grad_h_fused.as_mut_ptr(),
        )
    };
    if code != 0 {
        return Err(runtime_error());
    }
    Ok(FusedCeProbe {
        loss_full,
        loss_fused,
        grad_h_full,
        grad_h_fused,
    })
}

/// Reads the geometry of a GGUF model without building a training context.
///
/// This is the cheap planning input `Trainer` could not provide: it opens the
/// model, reads its hyper-parameters and tensor types, and frees it - no KV
/// cache, no compute buffer, no LoRA adapter, and no device allocation. `device`
/// is validated (asking for `Gpu` on a build without one is an error) but never
/// honoured with an offload, because reading geometry must not consume the
/// budget it is being read to plan.
pub fn model_info(model_path: impl AsRef<Path>, device: Device) -> Result<ModelInfo> {
    let path = path_to_cstring(model_path.as_ref())?;
    let mut info = ffi::RetroModelInfo::default();
    // SAFETY: the module contract validates input shapes and keeps every input/output allocation live.
    let code = unsafe { ffi::retro_read_model_info(path.as_ptr(), device.as_ffi(), &mut info) };
    if code != 0 {
        return Err(runtime_error());
    }
    Ok(ModelInfo {
        n_layer: info.n_layer,
        n_embd: info.n_embd,
        n_ff: info.n_ff,
        n_head: info.n_head,
        n_head_kv: info.n_head_kv,
        n_embd_head_k: info.n_embd_head_k,
        n_embd_head_v: info.n_embd_head_v,
        n_embd_k_gqa: info.n_embd_k_gqa,
        n_embd_v_gqa: info.n_embd_v_gqa,
        n_embd_r: info.n_embd_r,
        n_embd_s: info.n_embd_s,
        n_vocab: info.n_vocab,
        n_ctx_train: info.n_ctx_train,
        n_expert: info.n_expert,
        n_expert_used: info.n_expert_used,
        n_params: info.n_params,
        model_size_bytes: info.model_size_bytes,
        file_size_bytes: info.file_size_bytes,
        dominant_weight_bytes: info.dominant_weight_bytes,
        tied_embeddings: info.tied_embeddings,
        is_recurrent: info.is_recurrent,
        has_encoder: info.has_encoder,
        architecture: fixed_name(&info.architecture),
        dominant_weight_type: fixed_name(&info.dominant_weight_type),
    })
}

/// Enumerates the tensors of a GGUF as the loader sees them, with no context,
/// no KV cache and no device allocation.
///
/// The per-tensor counterpart of [`model_info`], and the input every trainable
/// set resolves against: aggregate geometry cannot answer "which tensors does
/// `modules = ["attn"]` select, at which dtypes, for how many bytes".
///
/// `tied_embeddings` is not in the tensor list - a tied head is *absent* from
/// it - so it is read from [`model_info`] and carried on the inventory, which
/// is what makes "this head shares the embedding's storage" resolvable at all.
pub fn tensor_inventory(model_path: impl AsRef<Path>, device: Device) -> Result<TensorInventory> {
    let path = model_path.as_ref();
    let info = model_info(path, device)?;
    let c_path = path_to_cstring(path)?;

    let mut version = 0_u32;
    let mut count = 0_usize;
    // SAFETY: the module contract validates input shapes and keeps every input/output allocation live.
    let code = unsafe {
        ffi::retro_read_tensor_inventory(
            c_path.as_ptr(),
            &mut version,
            ptr::null_mut(),
            0,
            &mut count,
        )
    };
    if code != 0 {
        return Err(runtime_error());
    }
    if version != ffi::TENSOR_INVENTORY_VERSION {
        return Err(Error::runtime(format!(
            "the runtime produced a version-{version} tensor inventory, \
             but this build reads version {}",
            ffi::TENSOR_INVENTORY_VERSION
        )));
    }

    let mut descriptors = vec![ffi::RetroTensorDesc::default(); count];
    if count > 0 {
        // SAFETY: the module contract validates input shapes and keeps every input/output allocation live.
        let code = unsafe {
            ffi::retro_read_tensor_inventory(
                c_path.as_ptr(),
                &mut version,
                descriptors.as_mut_ptr(),
                descriptors.len(),
                &mut count,
            )
        };
        if code != 0 {
            return Err(runtime_error());
        }
        // A second read that found more tensors than the first would mean the
        // file changed underneath us; truncating silently would resolve a set
        // against a model that no longer exists.
        if count != descriptors.len() {
            return Err(Error::runtime(format!(
                "the tensor inventory changed between the two reads of {}: \
                 {} tensors, then {count}",
                path.display(),
                descriptors.len()
            )));
        }
    }

    let tensors = descriptors
        .iter()
        .map(|desc| TensorDesc {
            name: fixed_name(&desc.name),
            ne: desc.ne,
            dtype: TensorDtype::from_ggml_name(&fixed_name(&desc.type_name)),
            n_elements: desc.n_elements,
            n_bytes: desc.n_bytes,
            storage_id: desc.storage_id,
        })
        .collect();
    Ok(TensorInventory::new(
        info.architecture,
        info.tied_embeddings,
        tensors,
    ))
}

/// Decodes one of the NUL-terminated fixed-size name fields of
/// [`ffi::RetroModelInfo`]. The runtime writes short ASCII labels, so a lossy
/// decode is the right failure mode: a mangled label must not fail a read whose
/// numeric fields are all valid.
pub(crate) fn fixed_name(field: &[c_char]) -> String {
    let bytes: Vec<u8> = field
        .iter()
        .take_while(|byte| **byte != 0)
        .map(|byte| *byte as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Lists the ggml backend devices compiled into this build, one per line as
/// `type\tname\tdescription`. `type` is `cpu`, `gpu`, `accel`, or `other`.
pub fn backend_list() -> Result<String> {
    // SAFETY: the module contract validates input shapes and keeps every input/output allocation live.
    unsafe { read_string(|buffer, n_buffer, out| ffi::retro_backend_list(buffer, n_buffer, out)) }
}

/// Returns whether a registered GPU can create and synchronize a real backend
/// context. This is intentionally stronger than [`backend_list`]: a Metal or
/// Vulkan backend may be listed even when its runtime queue cannot be created.
pub fn gpu_runtime_available() -> bool {
    // SAFETY: the module contract validates input shapes and keeps every input/output allocation live.
    unsafe { ffi::retro_gpu_runtime_probe() == 0 }
}

/// Measured host<->device transfer rates, in bytes per second.
///
/// The gate in front of activation offloading: an offload ring is not worth
/// building if the traffic an offload ring would add exceeds
/// 30 % of the compute it is meant to hide behind, and that ratio needs a real
/// figure for what a copy of the checkpoint's size costs on this machine.
///
/// Every copy is timed with a device synchronization after it, so the rates are
/// end-to-end with nothing overlapped - an actual ring would do better, which is
/// the safe direction for a number that decides whether to open work.
#[derive(Clone, Copy, Debug, Default)]
pub struct TransferRates {
    /// Size of one transfer, echoed back so a rate is never read without it.
    pub bytes_per_transfer: u64,
    pub iterations: u64,
    pub pinned_h2d_bytes_per_second: f64,
    pub pinned_d2h_bytes_per_second: f64,
    pub pageable_h2d_bytes_per_second: f64,
    pub pageable_d2h_bytes_per_second: f64,
    /// True when the backend exposes no dedicated host buffer type, so the two
    /// `pinned_*` figures are the pageable measurement repeated. A caller must
    /// not read them as a measurement of page-locked memory.
    pub pinned_is_pageable: bool,
    /// True when the backend's own default buffer type *is* host memory, i.e. it
    /// makes no separate device allocation at all. **Not** a unified-memory flag:
    /// Metal has unified memory and answers false, because a Metal buffer is not
    /// a host buffer. What says "there is no bus here" is the rate - far above
    /// any interconnect figure, and near-identical in both directions.
    pub device_buffer_is_host: bool,
}

impl TransferRates {
    /// Seconds a round trip of `bytes` would take at the pinned rates: one copy
    /// out and one back, which is what an offloaded checkpoint costs.
    pub fn round_trip_seconds(&self, bytes: u64) -> Option<f64> {
        (self.pinned_h2d_bytes_per_second > 0.0 && self.pinned_d2h_bytes_per_second > 0.0).then(
            || {
                bytes as f64 / self.pinned_h2d_bytes_per_second
                    + bytes as f64 / self.pinned_d2h_bytes_per_second
            },
        )
    }
}

/// Times `iterations` round trips of `bytes` each on the first GPU device of this
/// build, out of pinned and out of pageable host memory.
///
/// Errors when the build has no GPU device, when the device allocation fails, or
/// when either argument is zero.
pub fn transfer_probe(bytes: usize, iterations: u32) -> Result<TransferRates> {
    let mut rates = ffi::RetroTransferRates::default();
    // SAFETY: the module contract validates input shapes and keeps every input/output allocation live.
    let status = unsafe { ffi::retro_transfer_probe(bytes, iterations, &mut rates) };
    if status != 0 {
        return Err(runtime_error());
    }
    Ok(TransferRates {
        bytes_per_transfer: rates.bytes_per_transfer,
        iterations: rates.iterations,
        pinned_h2d_bytes_per_second: rates.pinned_h2d_bytes_per_second,
        pinned_d2h_bytes_per_second: rates.pinned_d2h_bytes_per_second,
        pageable_h2d_bytes_per_second: rates.pageable_h2d_bytes_per_second,
        pageable_d2h_bytes_per_second: rates.pageable_d2h_bytes_per_second,
        pinned_is_pageable: rates.pinned_is_pageable,
        device_buffer_is_host: rates.device_buffer_is_host,
    })
}

/// The bytes one slot holds before the first update, from the runtime's own
/// initializer. No model and no allocation: the initializer can be compared
/// against its declaration without running a graph build.
///
/// `n_elements` is the slot resolved against its parameter, via
/// [`retrograd_core::SlotDefinition::resolve`].
pub fn slot_initial_bytes(
    slot: &retrograd_core::SlotDefinition,
    n_elements: u64,
) -> Result<Vec<u8>> {
    let dtype = match slot.dtype {
        retrograd_core::SlotDtype::F32 => 0,
        retrograd_core::SlotDtype::I8 => 1,
    };
    let (init, code) = match slot.init {
        retrograd_core::SlotInit::Zero => (0, 0),
        retrograd_core::SlotInit::Code(code) => (1, code),
        retrograd_core::SlotInit::UniformCodebook => (2, 0),
    };
    // A slot whose byte count does not fit is an error, not a clamped
    // allocation the host aborts on.
    let n_bytes = n_elements
        .checked_mul(slot.dtype.bytes())
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| Error::overflow("a slot larger than this host can address"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(n_bytes)
        .map_err(|_| Error::invalid("a slot larger than this host can allocate"))?;
    bytes.resize(n_bytes, 0_u8);
    // SAFETY: `bytes` is exactly `n_bytes` long and borrowed for this
    // synchronous call only.
    let code = unsafe {
        ffi::retro_optimizer_slot_initial_bytes(
            dtype,
            init,
            code,
            n_elements,
            bytes.as_mut_ptr().cast(),
            n_bytes,
        )
    };
    if code != 0 {
        return Err(runtime_error());
    }
    Ok(bytes)
}
