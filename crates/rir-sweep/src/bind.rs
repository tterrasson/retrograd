//! Binding a kernel to a shape: buffers, byte strides and the constant block.
//!
//! **The geometry of an argument is not an input, it is a consequence.** Given
//! the extents of the axes, which dimension of which argument each axis indexes
//! (`LoopKernel::arg_axes`, published because the manifest publishes it)
//! decides `ne[]`, and the element type decides `nb[]`. That is what makes a
//! sweep over shapes possible without a hand-written binding per kernel, and it
//! is the same derivation `family_parity` performs to bind the whole registry
//! against the oracle.
//!
//! It lives here, in the tool, rather than in either of the two crates it
//! spans: it needs the Loop IR (`rir-lower`) *and* the runtime's `Values`
//! (`rir-runtime`), which share no dependency but `rir-core`. The parity lane
//! reads it back from here, so the shapes a sweep times are bound exactly as
//! the shapes the lane validates.

use rir_core::DType;
use rir_lower::LoopKernel;
use rir_runtime::{Arg, Manifest, Values};

use crate::SweepError;

/// The shape of one argument under a shape assignment.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    /// Logical extents, ggml order.
    pub ne: [usize; 4],
    /// Byte strides, packed. `nb[0]` is one stride **unit**: an element, or a
    /// whole block for a quantized format.
    pub nb: [usize; 4],
    /// Bytes the packed tensor occupies.
    pub bytes: usize,
}

/// `ne[]`, `nb[]` and the byte size of argument `arg` under `extents`
/// (axis order).
pub fn geometry(lk: &LoopKernel, arg: usize, extents: &[usize]) -> Result<Geometry, SweepError> {
    let fail = || SweepError::Unbindable {
        kernel: lk.name.clone(),
        why: format!("binding '{}' geometry overflows usize", lk.args[arg].name),
    };
    let dtype = lk.args[arg].ty.dtype;
    let mut ne = [1usize; 4];
    for (d, axis) in lk.arg_axes[arg].iter().enumerate() {
        if let Some(a) = axis {
            ne[d] = *extents
                .get(a.0 as usize)
                .ok_or_else(|| SweepError::Unbindable {
                    kernel: lk.name.clone(),
                    why: format!("binding '{}' refers to a missing axis", lk.args[arg].name),
                })?;
        }
    }
    let (unit, per_unit) = match dtype {
        DType::Quant(q) => {
            let d = q.desc();
            (d.block_bytes as usize, d.block_elements as usize)
        }
        other => (other.size_bytes(), 1),
    };
    let mut nb = [unit, 0, 0, 0];
    nb[1] = unit
        .checked_mul(ne[0].div_ceil(per_unit))
        .ok_or_else(fail)?;
    nb[2] = nb[1].checked_mul(ne[1]).ok_or_else(fail)?;
    nb[3] = nb[2].checked_mul(ne[2]).ok_or_else(fail)?;
    let bytes = nb[3].checked_mul(ne[3]).ok_or_else(fail)?;
    Ok(Geometry { ne, nb, bytes })
}

/// A buffer of one argument, in the shape the runtime needs it: floats for an
/// F32 binding, bytes for an F16 or quantized one.
pub enum Buf {
    F32(Vec<f32>),
    Bytes(Vec<u8>),
}

impl Buf {
    fn as_arg(&self) -> Arg<'_> {
        match self {
            Buf::F32(v) => Arg::input(v),
            Buf::Bytes(v) => Arg::input(v),
        }
    }

    fn as_arg_mut(&mut self) -> Arg<'_> {
        match self {
            Buf::F32(v) => Arg::output(v),
            Buf::Bytes(v) => Arg::output(v),
        }
    }

    /// The values a comparison reads, decoded when the buffer holds halves.
    pub fn values(&self) -> Vec<f32> {
        match self {
            Buf::F32(v) => v.clone(),
            Buf::Bytes(v) => v
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| rir_lower::interp::f16_to_f32(u16::from_le_bytes(*c)))
                .collect(),
        }
    }
}

fn rng(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let u = ((*seed >> 33) as u32) as f32 / u32::MAX as f32;
    u * 2.0 - 1.0
}

/// Inputs allocated and filled once for one (kernel, shape), reused by every
/// candidate timed on that shape.
///
/// Reused deliberately: two candidates must read the **same bytes**, or the
/// agreement check below compares two answers to two questions. It is also what
/// keeps a sweep affordable - a 2 M-element input is filled once per shape, not
/// once per candidate.
pub struct Fixture {
    pub geo: Vec<Geometry>,
    pub extents: Vec<usize>,
    inputs: Vec<Buf>,
    out_index: usize,
}

impl Fixture {
    /// Fills every input of `lk` for `extents`, deterministically: the seed is
    /// the shape, so the same shape gives the same bytes in every process.
    pub fn build(lk: &LoopKernel, extents: &[usize]) -> Result<Fixture, SweepError> {
        let unbindable = |why: String| SweepError::Unbindable {
            kernel: lk.name.clone(),
            why,
        };
        let writes: Vec<usize> = lk
            .args
            .iter()
            .enumerate()
            .filter(|(_, a)| a.access == rir_core::Access::Write)
            .map(|(i, _)| i)
            .collect();
        let [out_index] = writes[..] else {
            return Err(unbindable(format!(
                "{} written arguments; this harness binds exactly one",
                writes.len()
            )));
        };

        let mut seed = 0x9e37_79b9u64 ^ extents.iter().fold(1u64, |a, e| a * 31 + *e as u64);
        let geo: Vec<Geometry> = (0..lk.args.len())
            .map(|a| geometry(lk, a, extents))
            .collect::<Result<_, _>>()?;
        let mut inputs = Vec::with_capacity(lk.args.len());
        for (a, arg) in lk.args.iter().enumerate() {
            if a == out_index {
                inputs.push(Buf::F32(Vec::new()));
                continue;
            }
            let g = geo[a];
            inputs.push(match arg.ty.dtype {
                DType::F32 => Buf::F32((0..g.bytes / 4).map(|_| rng(&mut seed) * 1.5).collect()),
                DType::F16 => Buf::Bytes(
                    (0..g.bytes / 2)
                        .flat_map(|_| {
                            rir_lower::interp::f32_to_f16(rng(&mut seed) * 1.5).to_le_bytes()
                        })
                        .collect(),
                ),
                DType::Quant(q) => {
                    // `geometry` already performed the checked product. The
                    // packed byte size divided by one block is the exact block
                    // count, including row padding for a partial final block.
                    let blocks = g.bytes / q.desc().block_bytes as usize;
                    Buf::Bytes(
                        rir_core::random_block_bytes(q, blocks, &mut seed).ok_or_else(|| {
                            unbindable("quantized format without a description".into())
                        })?,
                    )
                }
                other => {
                    return Err(unbindable(format!(
                        "no fixture for a {} binding",
                        other.name()
                    )));
                }
            });
        }
        Ok(Fixture {
            geo,
            extents: extents.to_vec(),
            inputs,
            out_index,
        })
    }

    /// Byte strides of every argument, in argument order.
    pub fn nbs(&self) -> Vec<[usize; 4]> {
        self.geo.iter().map(|g| g.nb).collect()
    }

    /// A zeroed destination buffer of the written argument's type.
    pub fn output(&self, lk: &LoopKernel) -> Buf {
        let g = self.geo[self.out_index];
        if lk.args[self.out_index].ty.dtype == DType::F32 {
            Buf::F32(vec![0f32; g.bytes / 4])
        } else {
            Buf::Bytes(vec![0u8; g.bytes])
        }
    }

    /// The argument list in manifest order, with `out` in the written slot.
    pub fn bind<'a>(&'a self, out: &'a mut Buf) -> Vec<Arg<'a>> {
        let mut args: Vec<Arg<'a>> = (0..self.inputs.len())
            .filter(|a| *a != self.out_index)
            .map(|a| self.inputs[a].as_arg())
            .collect();
        args.insert(self.out_index, out.as_arg_mut());
        args
    }
}

/// The constant block, filled from the manifest's **own** list of names.
///
/// A value nobody asked for is not passed, and a name the manifest carries and
/// this cannot resolve is an error rather than a default: the second case is how
/// a harness starts dispatching with a zero extent and still reports a time.
pub fn push_values(
    m: &Manifest,
    lk: &LoopKernel,
    extents: &[usize],
    nbs: &[[usize; 4]],
) -> Result<Values, SweepError> {
    let unbindable = |why: String| SweepError::Unbindable {
        kernel: lk.name.clone(),
        why,
    };
    let mut v = Values::new();
    let as_u32 = |name: &str, value: usize| {
        u32::try_from(value).map_err(|_| {
            unbindable(format!(
                "{name}={value} is outside the manifest's u32 range"
            ))
        })
    };
    for pc in &m.push_constants {
        if let Some(axis) = pc.name.strip_prefix("n_") {
            let i = lk
                .axes
                .iter()
                .position(|a| a.name == axis)
                .ok_or_else(|| unbindable(format!("push constant {} is no axis", pc.name)))?;
            v.u32(&pc.name, as_u32(&pc.name, extents[i])?);
            continue;
        }
        if let Some((arg, dim)) = pc
            .name
            .rsplit_once("_nb")
            .and_then(|(a, d)| d.parse::<usize>().ok().map(|d| (a, d)))
            && let Some(i) = lk.args.iter().position(|x| x.name == arg)
        {
            let stride = nbs[i].get(dim).copied().ok_or_else(|| {
                unbindable(format!(
                    "push constant {} names no stride dimension",
                    pc.name
                ))
            })?;
            v.u32(&pc.name, as_u32(&pc.name, stride)?);
            continue;
        }
        // The flattened dispatch's own block: the total,
        // then one (divisor, multiplier, shift) triple per divisor. A linear
        // variant declares the total alone - it divides by nothing - and a name
        // the shader never declared is never written, because this walks the
        // manifest's list rather than a list of its own.
        if pc.name.starts_with("rir_flat") {
            let flat = lk.flat_axes();
            let divisor = |i: usize| -> Result<u32, SweepError> {
                let (ax, per) = flat[i];
                let extent = *extents
                    .get(ax.0 as usize)
                    .ok_or_else(|| unbindable(format!("flat axis {} is absent", ax.0)))?;
                as_u32("flat divisor", extent.div_ceil(per as usize))
            };
            if pc.name == "rir_flat_total" {
                let total = (0..flat.len()).try_fold(1u32, |product, i| {
                    product.checked_mul(divisor(i)?).ok_or_else(|| {
                        unbindable("flattened extent product exceeds u32".to_string())
                    })
                })?;
                v.u32(&pc.name, total);
                continue;
            }
            let (i, field) = pc.name["rir_flat".len()..]
                .split_once('_')
                .and_then(|(i, f)| i.parse::<usize>().ok().map(|i| (i, f)))
                .ok_or_else(|| unbindable(format!("flat push constant name {}", pc.name)))?;
            let d = divisor(i)?;
            let (mp, sh) = rir_lower::fastdiv_magic(d);
            v.u32(
                &pc.name,
                match field {
                    "div" => d,
                    "mp" => mp,
                    "sh" => sh,
                    other => {
                        return Err(unbindable(format!("flat push constant field {other}")));
                    }
                },
            );
            continue;
        }
        let i = lk
            .params
            .iter()
            .position(|p| p.name == pc.name)
            .ok_or_else(|| unbindable(format!("push constant {} resolves to nothing", pc.name)))?;
        // The parameters the kernels declare are an epsilon or a scale. A value
        // per *name*, so a new parameter is a stated refusal here rather than a
        // silent zero three layers down.
        let value = match lk.params[i].name.as_str() {
            "eps" => 1e-6,
            "scale" | "bias" => 0.75,
            other => return Err(unbindable(format!("no sweep value for parameter {other}"))),
        };
        v.f32(&pc.name, value);
    }
    Ok(v)
}
