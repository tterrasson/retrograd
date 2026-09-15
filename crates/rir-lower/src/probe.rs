//! A hand-built `LoopKernel` for the unit tests of the Loop IR and of the
//! interpreter.
//!
//! Without it, `loop_ir` is covered only by whatever the emitters happen to ask
//! it, and `interp` by device parity, that is, by a test that needs a GPU to
//! fail. The nests here are written
//! statement by statement, so what they check is the statement and not a
//! kernel's arithmetic.

use rir_core::{Access, Arg, ArgId, AxisDecl, AxisId, Extent, TensorType};

use crate::loop_ir::{LoopKernel, Stmt, VarId, VarKind};
use crate::schedule::Schedule;

/// The single argument (`y`), written.
pub const Y: ArgId = ArgId(0);
/// The only axis (`i`), the contiguous dimension of `y`.
pub const I: AxisId = AxisId(0);

/// One written F32 1-D argument, one axis, and whatever registers the test
/// names. Enough for the interpreter, which needs a place to publish what it
/// computed.
pub fn kernel(
    vars: &[(&str, VarKind)],
    shared: Vec<(VarId, u32)>,
    body: Vec<Stmt>,
    schedule: Schedule,
) -> LoopKernel {
    LoopKernel {
        name: "probe".into(),
        args: vec![Arg {
            name: "y".into(),
            ty: TensorType::f32(1),
            access: Access::Write,
        }],
        params: vec![],
        axes: vec![AxisDecl {
            name: "i".into(),
            extent: Extent::Dim { arg: Y, dim: 0 },
        }],
        arg_axes: vec![vec![Some(I)]],
        folds: vec![],
        constraints: vec![],
        reduction_semantics: vec![],
        var_names: vars.iter().map(|(n, _)| (*n).to_string()).collect(),
        var_kinds: vars.iter().map(|(_, k)| *k).collect(),
        shared,
        body,
        schedule,
    }
}
