//! Reading the generated manifest: the contract and nothing else.
//!
//! The schema is `rir_core::manifest`, written by `rir-emit` and re-exported
//! here: the fields consumed below are exactly those the emitter fills, and a
//! field added on one side is a field this one sees at compile time.
//! What lives here is what *reading* adds - structural
//! validation, the addressed byte range, the push-constant block. The runtime
//! fills no missing field: an incomplete manifest is an error.

use std::collections::HashMap;
use std::path::Path;

use rir_core::Backend;
pub use rir_core::manifest::{
    Binding, GridAxis, Manifest, PushConstant, PushType, SCHEMA_VERSION as SUPPORTED_SCHEMA,
};

use crate::{Arg, RuntimeError};

/// What a binding carried at `prepare`: direction and exact size.
///
/// It lives here, beside the manifest, because it is not a property of any one
/// backend: both the Vulkan device and the CUDA one size their buffers from what
/// was bound, and both have to refuse a read-back of a different size.
pub(crate) struct BoundArg {
    pub(crate) write: bool,
    pub(crate) len: usize,
}

/// Largest accepted push-constant offset. This is not the device limit (checked
/// against `maxPushConstantsSize` at pipeline creation), but a bound keeping
/// `push_size` away from overflow: an offset near `u32::MAX` in a crafted
/// manifest must be a recoverable error, not a panic while slicing.
const MAX_PUSH_OFFSET: u32 = 65_532;

/// Size of one stride unit (`nb[0]`, ggml `type_size`) and number of logical
/// elements it carries - 1 except for a quantized dtype, where the unit is one
/// block.
///
/// Both numbers come from the dtype itself, which is the canonical table
/// rather than a list copied into the reader. The manifest carries a `DType`,
/// so an unknown spelling is refused while deserializing and never reaches
/// here.
fn unit(b: &Binding) -> (usize, usize) {
    (b.dtype.size_bytes(), b.dtype.elements_per_unit())
}

/// The reader's half of the schema: what `rir-runtime` adds to deserializing a
/// `rir_core::manifest::Manifest`.
///
/// A trait rather than an inherent `impl`, because the schema is declared in
/// another crate - which is the point of: one type, and
/// each end adds only what it needs. `from_json` is the only way in, so nothing
/// below ever sees a manifest that was not checked.
pub trait ManifestReader: Sized {
    /// The manifest itself, so the methods below can read its fields: a trait
    /// declared outside the type's crate has no fields of its own.
    fn as_manifest(&self) -> &Manifest;

    /// Parses and validates a manifest.
    fn from_json(text: &str) -> Result<Self, RuntimeError>;

    /// Loads `manifest.vulkan.json` from a generated kernel directory.
    fn load(kernel_dir: &Path) -> Result<Self, RuntimeError> {
        Self::load_file(kernel_dir, "manifest.vulkan.json")
    }

    /// Loads a named manifest from a generated kernel directory - the variant's
    /// or the other backend's (`manifest.vec4.cuda.json`).
    fn load_file(kernel_dir: &Path, file: &str) -> Result<Self, RuntimeError> {
        let text = std::fs::read_to_string(kernel_dir.join(file))?;
        Self::from_json(&text)
    }

    /// Number of bytes a dispatch can address on a binding with the supplied
    /// extents and strides: `Σ_d (n_d − 1)·nb[d]`, plus one stride unit for the
    /// farthest element (or block).
    ///
    /// `None` if the manifest does not describe every binding dimension: then
    /// nothing can be asserted, which is better than an incorrect bound. This
    /// is not an estimate: the values are exactly the push constants read by
    /// the shader.
    fn required_bytes(&self, b: &Binding, values: &Values) -> Result<Option<usize>, RuntimeError> {
        let (unit, per_unit) = unit(b);
        let mut need = unit;
        for (d, slot) in b.extents.iter().enumerate() {
            let Some(axis) = slot else { return Ok(None) };
            let n = values.extent(&format!("n_{axis}"))? as usize;
            let stride = values.extent(&format!("{}_nb{d}", b.name))? as usize;
            // An empty axis addresses nothing: `saturating_sub` avoids counting
            // one extra unit *and* overflowing.
            let units = if d == 0 { n.div_ceil(per_unit) } else { n };
            need += units.saturating_sub(1) * stride;
        }
        Ok(Some(need))
    }

    /// Push-constant block size in bytes: the last offset plus its size. Zero if
    /// the kernel has none. Offsets are bounded by `MAX_PUSH_OFFSET` while
    /// reading, so the sum cannot overflow.
    fn push_size(&self) -> u32 {
        self.as_manifest()
            .push_constants
            .iter()
            .map(|p| p.offset + 4)
            .max()
            .unwrap_or(0)
    }

    /// Number of workgroups to dispatch, derived from supplied axis extents.
    /// Axes not carried by the grid (the reduction axis and axes beyond the
    /// third dimension) do not appear here: they are traversed **inside** the
    /// shader.
    fn groups(&self, values: &Values) -> Result<[u32; 3], RuntimeError> {
        let mut groups = [1u32; 3];
        // The flattened dispatch: one dimension over the whole parallel space.
        // Computed from the extents like the grid below
        // and not read from `rir_flat_total`, so that the constant the shader
        // bounds itself with and the grid this asks for are derived from the
        // same numbers by two paths - a disagreement is a wrong result, and it
        // shows up as one.
        if !self.as_manifest().flat.is_empty() {
            let mut total: u64 = 1;
            for f in &self.as_manifest().flat {
                let n = values.extent(&format!("n_{}", f.axis))?;
                total = total
                    .checked_mul(u64::from(n.div_ceil(f.per_index)))
                    .ok_or_else(|| {
                        RuntimeError::BadManifest(
                            "flattened dispatch size overflows u64".to_string(),
                        )
                    })?;
            }
            // Points **per workgroup**, which is the workgroup width under the
            // invocation mapping and one under the workgroup mapping: there, one
            // workgroup owns one point and its lanes cooperate on it
            //  Dividing by the width in that case would
            // dispatch a thirty-second of the rows.
            let per = match self.as_manifest().parallel_mapping {
                rir_core::manifest::ParallelMapping::Workgroup => 1,
                rir_core::manifest::ParallelMapping::Invocation => self.as_manifest().workgroup[0],
            };
            groups[0] = u32::try_from(total.div_ceil(u64::from(per))).map_err(|_| {
                RuntimeError::BadManifest(
                    "flattened dispatch exceeds the u32 grid range".to_string(),
                )
            })?;
            return Ok(groups);
        }
        // `check_structure` already rejected a fourth axis: nothing is
        // truncated here.
        for (d, g) in self.as_manifest().dispatch.iter().enumerate() {
            let n = values.extent(&format!("n_{}", g.axis))?;
            groups[d] = n.div_ceil(g.per_workgroup);
        }
        Ok(groups)
    }

    /// Serialized push-constant block in manifest layout. A declared but
    /// unsupplied entry is an error - silently inserting zero would produce a
    /// kernel that "runs" and computes nothing.
    fn push_bytes(&self, values: &Values) -> Result<Vec<u8>, RuntimeError> {
        let mut out = vec![0u8; self.push_size() as usize];
        for p in &self.as_manifest().push_constants {
            values.representable(&p.name)?;
            let v = values
                .get(&p.name)
                .ok_or_else(|| RuntimeError::MissingValue {
                    name: p.name.clone(),
                })?;
            let bits = match (p.ty, v) {
                (PushType::U32, Scalar::U32(n)) => n.to_ne_bytes(),
                (PushType::I32, Scalar::I32(n)) => n.to_ne_bytes(),
                (PushType::F32, Scalar::F32(x)) => x.to_ne_bytes(),
                _ => {
                    return Err(RuntimeError::MissingValue {
                        name: p.name.clone(),
                    });
                }
            };
            let off = p.offset as usize;
            out[off..off + 4].copy_from_slice(&bits);
        }
        Ok(out)
    }
}

impl ManifestReader for Manifest {
    fn as_manifest(&self) -> &Manifest {
        self
    }

    fn from_json(text: &str) -> Result<Self, RuntimeError> {
        // Deserializing is already half the validation: a dtype, an
        // access, a push-constant type or a backend outside the domain is a
        // `serde` error here, not a string compared further down.
        let m: Manifest = serde_json::from_str(text).map_err(unparsable_manifest)?;
        if m.schema_version != SUPPORTED_SCHEMA {
            return Err(RuntimeError::BadManifest(format!(
                "schema_version {}; this runtime reads {SUPPORTED_SCHEMA}",
                m.schema_version
            )));
        }
        // Two backends, and the same reader for both:
        // bindings, push-constant offsets and grid axes are the *contract*, and
        // CUDA introduces no new one - the params struct is the push-constant
        // layout, passed by value.
        if !matches!(m.backend, Backend::Vulkan | Backend::Cuda) {
            return Err(RuntimeError::BadManifest(format!(
                "backend '{}': this runtime executes vulkan and cuda",
                m.backend.name()
            )));
        }
        check_structure(&m)?;
        Ok(m)
    }
}

/// The dispatch half, crate-private: what `prepare` and the read-back verify
/// before a device is touched.
pub(crate) trait ManifestDispatch: ManifestReader {
    /// Everything a dispatch needs checked and encoded **before** a device is
    /// touched: binding count, binding direction, the addressed byte range of
    /// each buffer, the constant block and the workgroup counts.
    ///
    /// It is shared by the two backends, and that is
    /// the point rather than a saving of lines: these are the contract's checks,
    /// so a second copy of them is a second opinion on what the manifest says.
    /// What stays per-backend below this call is allocation, upload and
    /// submission - the three things a queue actually owns.
    fn encode_dispatch(
        &self,
        args: &[Arg],
        values: &Values,
    ) -> Result<(Vec<u8>, [u32; 3]), RuntimeError> {
        if args.len() != self.as_manifest().bindings.len() {
            return Err(RuntimeError::ArgCountMismatch {
                expected: self.as_manifest().bindings.len(),
                got: args.len(),
            });
        }
        // Binding direction is a contract, not a hint: binding an input where
        // the manifest expects an output would run the kernel without ever
        // reading the result back, while the reverse would provide an
        // uninitialized buffer. Both return a wrong result without failing.
        for (b, a) in self.as_manifest().bindings.iter().zip(args.iter()) {
            let expected = match (b.is_write(), a) {
                (true, Arg::In(_)) => "an output (Arg::output)",
                (false, Arg::Out(_)) => "an input (Arg::input)",
                _ => continue,
            };
            return Err(RuntimeError::AccessMismatch {
                binding: b.name.clone(),
                expected,
            });
        }

        let groups = self.groups(values)?;
        let push = self.push_bytes(values)?;

        // Bounds. Supplied extents and strides *are* what the kernel reads: the
        // addressed range can be computed exactly, and a shorter buffer is an
        // out-of-bounds access - silent on real hardware.
        for (b, a) in self.as_manifest().bindings.iter().zip(args.iter()) {
            let Some(need) = self.required_bytes(b, values)? else {
                continue;
            };
            if a.len() < need {
                return Err(RuntimeError::BufferTooSmall {
                    binding: b.name.clone(),
                    need,
                    got: a.len(),
                });
            }
        }
        Ok((push, groups))
    }

    /// The read-back's half of the same contract: `args` must be the set
    /// `prepare` bound - same bindings, same directions, same sizes. Device
    /// buffers were sized there, so a longer slice would copy beyond the
    /// allocation from a safe API.
    fn check_read_back(&self, bound: &[BoundArg], args: &[Arg]) -> Result<(), RuntimeError> {
        if args.len() != bound.len() {
            return Err(RuntimeError::ArgCountMismatch {
                expected: bound.len(),
                got: args.len(),
            });
        }
        for (i, (b, a)) in bound.iter().zip(args.iter()).enumerate() {
            let name = || {
                self.as_manifest()
                    .bindings
                    .get(i)
                    .map(|b| b.name.clone())
                    .unwrap_or_else(|| i.to_string())
            };
            if matches!(a, Arg::Out(_)) != b.write {
                return Err(RuntimeError::AccessMismatch {
                    binding: name(),
                    expected: if b.write {
                        "an output (Arg::output)"
                    } else {
                        "an input (Arg::input)"
                    },
                });
            }
            if a.len() != b.len {
                return Err(RuntimeError::ArgLenMismatch {
                    binding: name(),
                    expected: b.len,
                    got: a.len(),
                });
            }
        }
        Ok(())
    }
}

impl ManifestDispatch for Manifest {}

/// Is the manifest structurally executable? Version and backend say nothing
/// about offsets, dtypes, or the grid: an invalid field must fail here as
/// `BadManifest`, not later as a slicing panic, duplicate descriptor, or
/// truncated dispatch.
fn check_structure(m: &Manifest) -> Result<(), RuntimeError> {
    let bad = |s: String| Err(RuntimeError::BadManifest(s));

    if m.bindings.is_empty() {
        return bad("no bindings".into());
    }
    let mut seen_binding = std::collections::HashSet::new();
    for b in &m.bindings {
        if !seen_binding.insert(b.binding) {
            return bad(format!("binding {} declared twice", b.binding));
        }
        if b.extents.len() > 4 {
            return bad(format!(
                "binding '{}': {} dimensions, at most 4 in ggml",
                b.name,
                b.extents.len()
            ));
        }
        for name in b.extents.iter().flatten() {
            if !axis_declared(m, name) {
                return bad(format!(
                    "binding '{}': axis '{name}' absent from push constants (n_{name})",
                    b.name
                ));
            }
        }
    }

    // Push constants form one block: aligned offsets, one entry per slot,
    // and a unique name - otherwise two values would collide.
    let mut slots = std::collections::HashMap::new();
    for p in &m.push_constants {
        if p.offset % 4 != 0 {
            return bad(format!(
                "push constant '{}': offset {} not aligned to 4",
                p.name, p.offset
            ));
        }
        if p.offset > MAX_PUSH_OFFSET {
            return bad(format!(
                "push constant '{}': offset {} beyond {MAX_PUSH_OFFSET}",
                p.name, p.offset
            ));
        }
        if let Some(other) = slots.insert(p.offset, p.name.clone()) {
            return bad(format!(
                "push constants '{}' and '{other}' at the same offset {}",
                p.name, p.offset
            ));
        }
    }

    // The grid has three dimensions: a fourth axis would be silently ignored
    // and therefore never traversed.
    if m.dispatch.len() > 3 {
        return bad(format!(
            "dispatch of {} axes, 3 grid dimensions",
            m.dispatch.len()
        ));
    }
    for g in &m.dispatch {
        if g.per_workgroup == 0 {
            return bad(format!("axis '{}': per_workgroup = 0", g.axis));
        }
        if !axis_declared(m, &g.axis) {
            return bad(format!(
                "dispatch axis '{}' without push constant n_{}",
                g.axis, g.axis
            ));
        }
    }
    // The flattened dispatch, and the property that makes it safe to have
    // two ways of computing a grid: exactly one of them is populated. A
    // manifest carrying both would let a reader pick, and the two answers
    // are different geometries for the same kernel.
    if !m.flat.is_empty() && !m.dispatch.is_empty() {
        return bad("both a grid and a flattened dispatch".into());
    }
    if m.flat.is_empty() && m.dispatch.is_empty() {
        return bad("neither a grid nor a flattened dispatch".into());
    }
    for g in &m.flat {
        if g.per_index == 0 {
            return bad(format!("flat axis '{}': per_index = 0", g.axis));
        }
        if !axis_declared(m, &g.axis) {
            return bad(format!(
                "flat axis '{}' without push constant n_{}",
                g.axis, g.axis
            ));
        }
    }
    if m.workgroup.contains(&0) {
        return bad(format!("workgroup {:?}: a zero dimension", m.workgroup));
    }
    Ok(())
}

/// An axis extent is a `n_<axis>` push constant: an axis absent from it has
/// no value at dispatch.
fn axis_declared(m: &Manifest, axis: &str) -> bool {
    let want = format!("n_{axis}");
    m.push_constants.iter().any(|p| p.name == want)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar {
    U32(u32),
    I32(i32),
    F32(f32),
}

/// Push-constant values by name. The manifest says which and where;
/// the caller says how many.
///
/// `unrepresentable` holds the strides `strides` was handed that do not fit the
/// 32-bit slot the shader reads. They are kept aside rather than truncated: a
/// wrapped stride would address the wrong element *and* pass the bounds check,
/// which reads the same wrapped value, so the only honest place to say so is
/// where the value is read.
#[derive(Clone, Debug, Default)]
pub struct Values(HashMap<String, Scalar>, HashMap<String, usize>);

impl Values {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn u32(&mut self, name: &str, v: u32) -> &mut Self {
        self.0.insert(name.to_string(), Scalar::U32(v));
        self
    }

    pub fn i32(&mut self, name: &str, v: i32) -> &mut Self {
        self.0.insert(name.to_string(), Scalar::I32(v));
        self
    }

    pub fn f32(&mut self, name: &str, v: f32) -> &mut Self {
        self.0.insert(name.to_string(), Scalar::F32(v));
        self
    }

    /// The four ggml strides of an argument, in bytes, under the names used by
    /// the manifest (`<arg>_nb<d>`) - the common-case shortcut.
    pub fn strides(&mut self, arg: &str, nb: &[usize]) -> &mut Self {
        for (d, &v) in nb.iter().enumerate() {
            let name = format!("{arg}_nb{d}");
            match u32::try_from(v) {
                Ok(v) => {
                    self.1.remove(&name);
                    self.u32(&name, v);
                }
                Err(_) => {
                    self.0.remove(&name);
                    self.1.insert(name, v);
                }
            }
        }
        self
    }

    /// Refuses a value that was supplied but does not fit its slot, so that
    /// the caller reads *that* rather than a missing value.
    fn representable(&self, name: &str) -> Result<(), RuntimeError> {
        match self.1.get(name) {
            Some(&value) => Err(RuntimeError::UnrepresentableValue {
                name: name.to_string(),
                value,
            }),
            None => Ok(()),
        }
    }

    pub fn get(&self, name: &str) -> Option<Scalar> {
        self.0.get(name).copied()
    }

    /// An unsigned integer value (axis extent or stride), named in the error if
    /// missing or of the wrong type.
    fn extent(&self, name: &str) -> Result<u32, RuntimeError> {
        self.representable(name)?;
        match self.get(name) {
            Some(Scalar::U32(n)) => Ok(n),
            _ => Err(RuntimeError::MissingValue {
                name: name.to_string(),
            }),
        }
    }
}

/// A manifest that is not the JSON it claims to be. The parser's own words
/// name the line and the column, so they are the message.
fn unparsable_manifest(e: serde_json::Error) -> RuntimeError {
    RuntimeError::BadManifest(e.to_string())
}

#[cfg(test)]
mod tests {
    use rir_core::manifest::FlatAxis;

    use super::*;

    const SUM_ROWS: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../generated/rir/sum_rows_q8_0/manifest.vulkan.json"
    ));

    /// A manifest mutated **as a value**, then re-serialized: the way to build
    /// a structurally invalid manifest now that the schema is a type.
    /// A textual substitution would depend on the
    /// pretty-printer's line breaks, which are `serde_json`'s and not a
    /// contract.
    fn tweaked(f: impl FnOnce(&mut Manifest)) -> String {
        // `from_str`, not `from_json`: the point is to start from a manifest the
        // validation has *not* seen, and break one thing in it.
        let mut m: Manifest = serde_json::from_str(SUM_ROWS).unwrap();
        f(&mut m);
        serde_json::to_string(&m).unwrap()
    }

    /// The committed manifest reads back as-is - this test breaks if the
    /// generator changes the contract without the runtime following.
    #[test]
    fn the_generated_manifest_reads_back() {
        let m = Manifest::from_json(SUM_ROWS).unwrap();
        assert_eq!(m.name, "sum_rows_q8_0");
        assert_eq!(m.bindings.len(), 2);
        assert_eq!(m.workgroup, [32, 1, 1]);
        assert_eq!(m.push_size(), 20);
        assert!(m.features.iter().any(|f| f == "storage_buffer_8bit"));
    }

    /// `n_row` rows, one workgroup per row: as many workgroups as
    /// rows. The `col` axis is not in the grid - it is traversed by the
    /// lanes.
    #[test]
    fn the_workgroup_count_comes_from_the_manifest() {
        let m = Manifest::from_json(SUM_ROWS).unwrap();
        let mut v = Values::new();
        v.u32("n_row", 7).u32("n_col", 128);
        assert_eq!(m.groups(&v).unwrap(), [7, 1, 1]);
    }

    #[test]
    fn the_invocation_dispatch_rounds_up() {
        let mut m = Manifest::from_json(SUM_ROWS).unwrap();
        m.dispatch[0].per_workgroup = 64;
        let mut v = Values::new();
        v.u32("n_row", 129);
        assert_eq!(m.groups(&v).unwrap()[0], 3, "129 rows in groups of 64");
    }

    /// A crafted set of flattened extents must be rejected before its product
    /// wraps and turns a huge dispatch into a small valid-looking grid.
    #[test]
    fn a_flattened_dispatch_product_cannot_overflow() {
        let mut m = Manifest::from_json(SUM_ROWS).unwrap();
        m.dispatch.clear();
        m.flat = ["row", "col", "batch"]
            .into_iter()
            .map(|axis| FlatAxis {
                axis: axis.to_string(),
                per_index: 1,
            })
            .collect();
        let mut v = Values::new();
        v.u32("n_row", u32::MAX)
            .u32("n_col", u32::MAX)
            .u32("n_batch", u32::MAX);

        assert!(matches!(m.groups(&v), Err(RuntimeError::BadManifest(_))));
    }

    /// The written layout follows declared offsets, not `Values` call order.
    #[test]
    fn the_push_constants_follow_the_declared_offsets() {
        let m = Manifest::from_json(SUM_ROWS).unwrap();
        let mut v = Values::new();
        v.u32("n_col", 128)
            .u32("n_row", 7)
            .strides("x", &[34, 136])
            .strides("y", &[4]);
        let bytes = m.push_bytes(&v).unwrap();
        assert_eq!(bytes.len(), 20);
        assert_eq!(u32::from_ne_bytes(bytes[0..4].try_into().unwrap()), 7); // n_row
        assert_eq!(u32::from_ne_bytes(bytes[4..8].try_into().unwrap()), 128); // n_col
        assert_eq!(u32::from_ne_bytes(bytes[8..12].try_into().unwrap()), 34); // x_nb0
        assert_eq!(u32::from_ne_bytes(bytes[16..20].try_into().unwrap()), 4); // y_nb0
    }

    /// The addressed range is exact, not an upper estimate: `x` has 4 rows of 4
    /// 34-byte blocks, and that is all the shader touches.
    #[test]
    fn the_addressed_range_comes_from_the_extents_and_strides() {
        let m = Manifest::from_json(SUM_ROWS).unwrap();
        let (n_col, n_row) = (128u32, 4u32);
        let stride = 34 * (n_col / 32);
        let mut v = Values::new();
        v.u32("n_row", n_row)
            .u32("n_col", n_col)
            .strides("x", &[34, stride as usize])
            .strides("y", &[4]);

        let x = m.required_bytes(&m.bindings[0], &v).unwrap();
        assert_eq!(x, Some((stride * n_row) as usize), "4 rows of 4 blocks");
        let y = m.required_bytes(&m.bindings[1], &v).unwrap();
        assert_eq!(y, Some(4 * n_row as usize));
    }

    /// A padded row is not contiguous: the bound follows declared strides, not
    /// logical size.
    #[test]
    fn the_addressed_range_follows_a_padded_stride() {
        let m = Manifest::from_json(SUM_ROWS).unwrap();
        let mut v = Values::new();
        v.u32("n_row", 3)
            .u32("n_col", 32)
            .strides("x", &[34, 64])
            .strides("y", &[4]);
        // Two complete padded rows, then 34 useful bytes.
        assert_eq!(
            m.required_bytes(&m.bindings[0], &v).unwrap(),
            Some(2 * 64 + 34)
        );
    }

    /// An offset near `u32::MAX` must be rejected before `push_size` or
    /// `push_bytes` can overflow. It is a recoverable error.
    #[test]
    fn a_huge_offset_is_refused_without_panicking() {
        let bad = tweaked(|m| m.push_constants[4].offset = 4_294_967_292);
        assert!(matches!(
            Manifest::from_json(&bad),
            Err(RuntimeError::BadManifest(_))
        ));
    }

    /// Two entries at the same offset: the second would overwrite the first
    /// silently.
    #[test]
    fn two_push_constants_at_the_same_offset_are_refused() {
        let bad = tweaked(|m| m.push_constants[4].offset = 12);
        assert!(matches!(
            Manifest::from_json(&bad),
            Err(RuntimeError::BadManifest(_))
        ));
    }

    /// A fourth grid axis would be silently ignored and never traversed, so the
    /// manifest is rejected.
    #[test]
    fn a_four_axis_dispatch_is_refused() {
        let bad = tweaked(|m| {
            let axis = m.dispatch[0].clone();
            m.dispatch = vec![axis.clone(), axis.clone(), axis.clone(), axis];
        });
        assert!(matches!(
            Manifest::from_json(&bad),
            Err(RuntimeError::BadManifest(_))
        ));
    }

    #[test]
    fn a_zero_per_workgroup_is_refused() {
        let bad = tweaked(|m| m.dispatch[0].per_workgroup = 0);
        assert!(matches!(
            Manifest::from_json(&bad),
            Err(RuntimeError::BadManifest(_))
        ));
    }

    #[test]
    fn a_zero_workgroup_is_refused() {
        let bad = tweaked(|m| m.workgroup[0] = 0);
        assert!(matches!(
            Manifest::from_json(&bad),
            Err(RuntimeError::BadManifest(_))
        ));
    }

    /// A dtype this runtime cannot measure prevents bounding the addressed
    /// range and is therefore rejected.
    #[test]
    fn an_unknown_dtype_is_refused() {
        let bad = SUM_ROWS.replace(r#""dtype": "q8_0""#, r#""dtype": "q4_k""#);
        assert!(matches!(
            Manifest::from_json(&bad),
            Err(RuntimeError::BadManifest(_))
        ));
    }

    #[test]
    fn an_unknown_access_is_refused() {
        let bad = SUM_ROWS.replace(r#""access": "write""#, r#""access": "readwrite""#);
        assert!(matches!(
            Manifest::from_json(&bad),
            Err(RuntimeError::BadManifest(_))
        ));
    }

    #[test]
    fn two_bindings_at_the_same_number_are_refused() {
        let bad = tweaked(|m| m.bindings[1].binding = 0);
        assert!(matches!(
            Manifest::from_json(&bad),
            Err(RuntimeError::BadManifest(_))
        ));
    }

    /// A stride past 32 bits is refused by name, by both readers of it, rather
    /// than wrapped into a value that would pass the bounds check.
    #[test]
    fn a_stride_past_u32_is_refused_rather_than_wrapped() {
        let m = Manifest::from_json(SUM_ROWS).unwrap();
        let mut v = Values::new();
        v.u32("n_row", 7)
            .u32("n_col", 128)
            .strides("x", &[34, (1usize << 32) + 136])
            .strides("y", &[4]);
        for result in [
            m.push_bytes(&v).map(|_| ()),
            m.required_bytes(&m.bindings[0], &v).map(|_| ()),
        ] {
            assert!(
                matches!(
                    &result,
                    Err(RuntimeError::UnrepresentableValue { name, .. }) if name == "x_nb1"
                ),
                "{result:?}"
            );
        }
    }

    /// A missing value is a named error, not a silent zero.
    #[test]
    fn a_missing_push_constant_is_named() {
        let m = Manifest::from_json(SUM_ROWS).unwrap();
        let mut v = Values::new();
        v.u32("n_row", 7).u32("n_col", 128).strides("x", &[34, 136]);
        match m.push_bytes(&v) {
            Err(RuntimeError::MissingValue { name }) => assert_eq!(name, "y_nb0"),
            other => panic!("expected error, {other:?}"),
        }
    }
}
