//! Model-free contracts for `out_prod` decoding quantized weights in place on CPU.

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
