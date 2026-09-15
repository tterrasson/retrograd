//! Byte addressing and axis extents, in the Rust the CPU emitter prints.
//!
//! The GPU backends share `crate::printer::Printer::addr`; this one cannot,
//! it prints `t.nb[d]` field accesses on a `TensorRef` the emitted file
//! declares itself, not an offset into a flat buffer

use rir_lower::AddrTerm;

use super::CpuPrinter;

impl CpuPrinter<'_> {
    pub(super) fn extent_name(&self, axis: rir_core::AxisId) -> String {
        format!("n_{}", self.k.axes[axis.0 as usize].name)
    }

    pub(super) fn addr(&self, arg: rir_core::ArgId, addr: &[AddrTerm]) -> String {
        let name = &self.k.args[arg.0 as usize].name;
        let terms: Vec<String> = addr
            .iter()
            .map(|t| match t {
                AddrTerm::VarNb { var, dim } => {
                    format!("{} * {}.nb[{}]", self.var(*var), name, dim)
                }
                AddrTerm::VarConst { var, c } => {
                    if *c == 1 {
                        self.var(*var).to_string()
                    } else {
                        format!("{} * {}", self.var(*var), c)
                    }
                }
                AddrTerm::Const(c) => format!("{c}"),
            })
            .collect();
        terms.join(" + ")
    }
}
