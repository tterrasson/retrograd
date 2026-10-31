//! The one thing that is not a word: how a backend reaches an element.
//!
//! `Load` and `Store` are the two `Stmt` variants measured as **structurally**
//! divergent - not three spellings of one shape,
//! but three shapes:
//!
//! - **Vulkan** binds each access type as a *typed view* over the same binding
//!   (descriptor aliasing) and indexes it in **elements**, so a Loop IR byte
//!   address has to be divided by the element size. A vector read is *w*
//!   indexed reads from a common base index.
//! - **CUDA** binds a `uint8_t *` and reinterprets it at a **byte** offset.
//!   There is no index division because there is nothing typed to index. A
//!   vector read is *w* scalar reads from a common base address - deliberately
//!   not a `float4`, which would demand a sixteen-byte alignment the contract
//!   does not require.
//! - **Metal** also reinterprets at a byte offset, but its vector read is
//!   **one** `packed_float{w}` load, whose alignment is one element - so unlike
//!   the two others it has no per-component expansion at all.
//!
//! The narrowing conversions differ with them (`__float2half`, `half(x)`,
//! `float16_t(x)`), and so does the bounded store: CUDA and Metal advance by
//! `{arg}_nb0` bytes, Vulkan by one element index.
//!
//! Merging these would mean the skeleton choosing an addressing mode, which is
//! a decision, and `rir-emit`'s golden rule is that emitters take none. So they
//! stay per backend, behind this trait, and the trait is separate from
//! [`Dialect`](crate::dialect::Dialect) on purpose: everything in that one is a
//! word, and nothing here is.

use rir_core::{ArgId, AxisId};
use rir_lower::{AddrTerm, MemType, VarId};

use crate::dialect::Dialect;

use super::Printer;

/// A read of one element, as the enclosing `Stmt::Load` describes it.
pub(crate) struct LoadOp<'a> {
    pub dst: VarId,
    pub arg: ArgId,
    pub ty: MemType,
    pub addr: &'a [AddrTerm],
    pub width: u32,
}

/// A write of one element, as the enclosing `Stmt::Store` describes it.
pub(crate) struct StoreOp<'a> {
    pub arg: ArgId,
    pub ty: MemType,
    pub addr: &'a [AddrTerm],
    pub value: VarId,
    pub width: u32,
    /// `(first index, axis)` when the vector may run past the axis extent.
    pub bound: Option<(VarId, AxisId)>,
}

/// A run of consecutive elements along dimension 0, as the enclosing
/// `Stmt::LoadRun` describes it.
pub(crate) struct LoadRunOp<'a> {
    pub dsts: &'a [VarId],
    pub arg: ArgId,
    pub ty: MemType,
    pub addr: &'a [AddrTerm],
}

/// How a backend reads and writes a binding.
///
/// Bounded by [`Dialect`] rather than being a supertrait of it: the printer is
/// generic over the dialect, and these two methods need a printer of *their
/// own* dialect to write into.
pub(crate) trait MemoryModel: Dialect + Sized {
    fn load(p: &mut Printer<'_, Self>, op: &LoadOp<'_>);
    fn store(p: &mut Printer<'_, Self>, op: &StoreOp<'_>);

    /// One element of `arg` at the byte offset `at`, as an F32 expression
    /// except for `U8`, which yields an unsigned integer for bit extraction.
    fn read_at(p: &Printer<'_, Self>, arg: ArgId, ty: MemType, at: &str) -> String;

    /// `width` consecutive elements at the byte offset `at` as **one** access,
    /// spelled as a `float{width}` expression, or `None` when the backend has
    /// no access wider than an element that the contract's alignment allows.
    fn read_vector_at(
        _p: &Printer<'_, Self>,
        _arg: ArgId,
        _ty: MemType,
        _at: &str,
        _width: u32,
    ) -> Option<String> {
        None
    }

    /// `width` consecutive **unsigned bytes** at the byte offset `at` as one
    /// access, spelled as a vector expression of the type [`Self::uvec_ty`]
    /// names, or `None` when the backend reads such a run component by
    /// component (it still amortizes the address algebra).
    fn read_uvector_at(
        _p: &Printer<'_, Self>,
        _arg: ArgId,
        _at: &str,
        _width: u32,
    ) -> Option<String> {
        None
    }

    /// The type a `read_uvector_at` result is declared as.
    fn uvec_ty(width: u32) -> String;

    /// Component `c` of an unsigned-byte vector register.
    fn ucomponent(text: &str, c: usize) -> String;
}

/// The widest single access `read_vector_at` is asked for: four elements, the
/// widest vector all three shading languages name.
const RUN_CHUNK: usize = 4;

impl<D: Dialect + MemoryModel> Printer<'_, D> {
    /// A `Stmt::LoadRun`, the same skeleton on every GPU backend.
    ///
    /// Dense runs sit under a `Stmt::UnitStride` that has tested `nb[0]`
    /// against the element size; quantized payload runs are consecutive by
    /// their bit plans. Read in vectors where the backend has them, at constant
    /// offsets otherwise, with no test inside the segment loop.
    pub(crate) fn load_run(&mut self, op: &LoadRunOp<'_>) {
        let LoadRunOp {
            dsts,
            arg,
            ty,
            addr,
        } = *op;
        let elem = ty.size_bytes();
        // A run of unsigned bytes (a quantized payload) lands in integer
        // registers: the bit extraction wants the raw bits, so its components
        // are declared as the unsigned type rather than converted to float.
        let u8 = ty == MemType::U8;
        let decl = if u8 { D::UINT } else { "float" };
        let names: Vec<String> = dsts.iter().map(|d| self.var(*d).to_string()).collect();
        let Some(first) = names.first() else {
            return;
        };
        let at = format!("{first}_a");
        let a = self.addr(arg, addr);
        self.line(&format!("const {} {at} = {a};", D::UINT));
        let offset = |c: usize| {
            if c == 0 {
                at.clone()
            } else {
                format!("{at} + {}u", c as u32 * elem)
            }
        };
        for start in (0..names.len()).step_by(RUN_CHUNK) {
            let width = (names.len() - start).min(RUN_CHUNK);
            let vector = if width > 1 {
                if u8 {
                    D::read_uvector_at(self, arg, &offset(start), width as u32)
                } else {
                    D::read_vector_at(self, arg, ty, &offset(start), width as u32)
                }
            } else {
                None
            };
            match vector {
                Some(v) if u8 => {
                    let r = format!("{}_r", names[start]);
                    self.line(&format!("const {} {r} = {v};", D::uvec_ty(width as u32)));
                    for c in 0..width {
                        let comp = D::ucomponent(&r, c);
                        self.line(&format!(
                            "const {} {} = {}({comp});",
                            D::UINT,
                            names[start + c],
                            D::UINT
                        ));
                    }
                }
                Some(v) => {
                    let kind = rir_lower::VarKind::Vec(width as u32);
                    let r = format!("{}_r", names[start]);
                    self.line(&format!("const {} {r} = {v};", D::decl_ty(kind)));
                    for c in 0..width {
                        let comp = D::component(kind, &r, c);
                        self.line(&format!("const float {} = {comp};", names[start + c]));
                    }
                }
                None => {
                    for (c, name) in names.iter().enumerate().skip(start).take(width) {
                        let e = D::read_at(self, arg, ty, &offset(c));
                        self.line(&format!("const {decl} {name} = {e};"));
                    }
                }
            }
        }
    }
}
