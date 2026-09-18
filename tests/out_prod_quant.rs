//! Model-free contracts for quantized `out_prod` on CPU and GPU backends.

mod common;

#[test]
fn out_prod_all_dequant_types_run_natively_on_cpu() {
    common::assert_out_prod_all_dequant_types_run_on_cpu();
    common::assert_out_prod_extra_types_match_cpu(false, "cpu");
}

#[test]
fn out_prod_quant_cpu_is_independent_of_the_legacy_cuda_budget() {
    common::assert_out_prod_quant_budget_independent(false, "cpu");
}

#[cfg(any(retro_cuda, retro_vulkan, retro_metal))]
#[test]
fn out_prod_quant_gpu_small_reductions_keep_the_scratch_free_path() {
    assert!(retrograd::gpu_runtime_available());
    common::assert_out_prod_all_dequant_types_match_cpu("gpu");
    if !cfg!(retro_metal) {
        common::assert_out_prod_extra_types_match_cpu(true, "gpu");
    }
    common::assert_out_prod_quant_budget_independent(true, "gpu");
}

/// Training-sized reductions must keep F32 accuracy when CUDA routes them to
/// bounded dequantize+SGEMM. Cover tails, batched weights and broadcast planes;
/// the old short-reduction probes only exercise the scratch-free kernel.
#[cfg(any(retro_cuda, retro_vulkan, retro_metal))]
#[test]
fn out_prod_quant_gpu_training_shapes_match_cpu_with_bounded_scratch() {
    assert!(retrograd::gpu_runtime_available());
    let mut types = retrograd::dequant_types();
    if !cfg!(retro_metal) {
        types.extend(
            common::OUT_PROD_EXTRA_TYPES
                .iter()
                .map(|&(id, name)| (id, name.to_owned())),
        );
    }
    let cases = [
        ([512_i64, 257, 1, 1], [33_i64, 257, 1, 1]),
        ([256, 129, 2, 1], [32, 129, 2, 1]),
        // 2 MiB of decoded weights: a 1 MiB budget forces several slices,
        // with a short final slice and broadcast over both batch axes.
        ([512, 513, 1, 2], [33, 513, 2, 4]),
    ];
    for (type_id, type_name) in types {
        for (a_shape, b_shape) in cases {
            let a = common::deterministic_f32s(
                usize::try_from(a_shape.iter().product::<i64>()).unwrap(),
                0x7150,
            );
            let b = common::deterministic_f32s(
                usize::try_from(b_shape.iter().product::<i64>()).unwrap(),
                0x7260,
            );
            let len = usize::try_from(a_shape[0] * b_shape[0] * b_shape[2] * b_shape[3]).unwrap();
            // CPU's F16 kernel requires matching batch dimensions. Expand the
            // reference weights explicitly rather than relying on its broadcast
            // support; the GPU still receives the original broadcast shape.
            let mut cpu_a = Vec::new();
            let plane_len = usize::try_from(a_shape[0] * a_shape[1]).unwrap();
            for i3 in 0..b_shape[3] {
                for i2 in 0..b_shape[2] {
                    let plane = (i3 / (b_shape[3] / a_shape[3])) * a_shape[2]
                        + i2 / (b_shape[2] / a_shape[2]);
                    let start = usize::try_from(plane).unwrap() * plane_len;
                    cpu_a.extend_from_slice(&a[start..start + plane_len]);
                }
            }
            let cpu_shape = [a_shape[0], a_shape[1], b_shape[2], b_shape[3]];
            let run = |gpu| {
                retrograd::probe_op(
                    retrograd::ProbeOp::OutProdQuant,
                    gpu,
                    retrograd::ProbeInputs::pair(
                        if gpu { a_shape } else { cpu_shape },
                        if gpu { &a } else { &cpu_a },
                        b_shape,
                        &b,
                    ),
                    [f32::from(u8::try_from(type_id).unwrap()), 0.0],
                    len,
                )
                .unwrap_or_else(|error| panic!("{type_name}: {error}"))
            };
            let cpu = run(false);
            let budgets: &[&str] = if cfg!(retro_cuda) {
                &["0", "1"]
            } else {
                &["64"]
            };
            for budget in budgets {
                let _budget = common::EnvGuard::set("GGML_CUDA_DEQUANT_BUDGET_MB", budget);
                let gpu = run(true);
                for (index, (&expected, &actual)) in cpu.iter().zip(&gpu).enumerate() {
                    assert!(
                        (expected - actual).abs() <= 2.0e-4 + 2.0e-5 * expected.abs(),
                        "{type_name} {a_shape:?} {b_shape:?}, budget={budget}, index={index}: \
                         cpu={expected}, gpu={actual}"
                    );
                }
            }
        }
    }
}
