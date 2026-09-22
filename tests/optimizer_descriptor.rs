//! The optimizer slot initializers, without a model or a graph: the runtime's
//! fill checked against the declaration in `retrograd-core`. Two of the three
//! initializers belong to an optimizer that has no update step here yet.

use retrograd::{
    GefenLayout, GefenVariant, OptimizerKind, SlotDefinition, SlotDtype, SlotInit, SlotShape,
    TensorDtype, TensorRole, TrainableEntry, slot_initial_bytes,
};

fn entry(name: &str, ne: [i64; 4]) -> TrainableEntry {
    let n_elements = ne.iter().product::<i64>() as u64;
    TrainableEntry {
        name: name.to_string(),
        role: TensorRole::Base,
        ne,
        dtype: TensorDtype::F32,
        n_elements,
        n_bytes: n_elements * 4,
        storage_id: 0,
    }
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(f32::from_ne_bytes)
        .collect()
}

/// Every declared initializer, against its own arithmetic, driven off
/// `slot_definitions()` so a new initializer cannot go unchecked.
#[test]
fn every_declared_slot_is_initialized_the_way_its_definition_says() {
    // Not a multiple of every block size, so a partial trailing block is
    // exercised.
    let parameter = entry("blk.0.ffn_up.weight", [1536, 7, 1, 1]);
    let mut seen_zero = 0;
    let mut seen_code = 0;
    let mut seen_codebook = 0;

    for optimizer in [
        OptimizerKind::AdamW,
        OptimizerKind::Sgd,
        OptimizerKind::Muon,
        OptimizerKind::Gefen(GefenLayout::default()),
        OptimizerKind::Gefen(GefenLayout {
            variant: GefenVariant::QuantizedM,
            ..GefenLayout::default()
        }),
    ] {
        let scopes = [
            optimizer.slot_definitions(),
            optimizer.shared_slot_definitions(),
        ];
        for slots in scopes {
            for slot in &slots {
                let planned = slot.resolve(&parameter);
                let bytes = slot_initial_bytes(slot, planned.n_elements)
                    .unwrap_or_else(|error| panic!("{optimizer}/{}: {error}", slot.name));
                assert_eq!(bytes.len() as u64, planned.n_bytes, "{}", slot.name);
                match slot.init {
                    // The master copy is not in any optimizer's table: it
                    // belongs to a parameter whose store is narrower than its
                    // update, so nothing enumerated here can produce one.
                    SlotInit::Parameter => unreachable!(
                        "{optimizer} declares a parameter-initialized slot '{}'",
                        slot.name
                    ),
                    SlotInit::Zero => {
                        assert!(
                            bytes.iter().all(|byte| *byte == 0),
                            "{optimizer}/{} is not zero",
                            slot.name
                        );
                        seen_zero += 1;
                    }
                    SlotInit::Code(code) => {
                        assert_eq!(slot.dtype, SlotDtype::I8, "{}", slot.name);
                        assert!(
                            bytes.iter().all(|byte| *byte == code),
                            "{optimizer}/{} is not filled with {code}",
                            slot.name
                        );
                        seen_code += 1;
                    }
                    SlotInit::UniformCodebook => {
                        assert_eq!(slot.dtype, SlotDtype::F32, "{}", slot.name);
                        let values = f32s(&bytes);
                        let last = values.len() - 1;
                        for (index, value) in values.iter().enumerate() {
                            let expected = -1.0_f32 + 2.0 * (index as f32) / (last as f32);
                            assert!(
                                (value - expected).abs() <= 1.0e-6,
                                "{optimizer}/{}[{index}] is {value}, not {expected}",
                                slot.name
                            );
                        }
                        assert_eq!(values[0], -1.0);
                        assert_eq!(values[last], 1.0);
                        // No exact zero: a zero block stores a canonical index
                        // and a zero scale instead.
                        assert!(values.iter().all(|value| *value != 0.0));
                        seen_codebook += 1;
                    }
                }
            }
        }
    }

    assert!(seen_zero > 0 && seen_code > 0 && seen_codebook > 0);
}

/// The canonical index a zero block stores, filled as an unsigned byte.
#[test]
fn a_zero_block_stores_the_canonical_index_as_an_unsigned_byte() {
    let indices = SlotDefinition {
        name: "indices",
        dtype: SlotDtype::I8,
        shape: SlotShape::Parameter,
        init: SlotInit::Code(retrograd::GEFEN_ZERO_BLOCK_INDEX),
    };
    let bytes = slot_initial_bytes(&indices, 5).expect("fill a byte-index slot");
    assert_eq!(bytes, vec![retrograd::GEFEN_ZERO_BLOCK_INDEX; 5]);
    const { assert!(retrograd::GEFEN_ZERO_BLOCK_INDEX > i8::MAX as u8 / 2) }
}

/// A block-shaped slot is `ceil(N / B)` elements.
#[test]
fn a_block_shaped_slot_costs_an_element_for_its_partial_trailing_block() {
    let scales = SlotDefinition {
        name: "scales",
        dtype: SlotDtype::F32,
        shape: SlotShape::Blocks(1024),
        init: SlotInit::Zero,
    };
    let planned = scales.resolve(&entry("blk.0.attn_q.weight", [2049, 1, 1, 1]));
    assert_eq!(planned.n_elements, 3);
    assert_eq!(
        slot_initial_bytes(&scales, planned.n_elements)
            .expect("fill a block-shaped slot")
            .len(),
        12
    );
}

/// A buffer that is not the slot's size is an error, never a partial fill.
#[test]
fn the_initializer_refuses_a_buffer_that_is_not_the_slots_size() {
    let codebook = SlotDefinition {
        name: "codebook",
        dtype: SlotDtype::F32,
        shape: SlotShape::Fixed(256),
        init: SlotInit::UniformCodebook,
    };
    let error = slot_initial_bytes(&codebook, u64::MAX).expect_err("no host holds that");
    assert!(error.to_string().contains("slot larger than"), "{error}");
}

#[test]
fn the_native_initializer_rejects_wrapped_sizes_without_writing() {
    let mut bytes = [0xA5_u8; 4];
    for count in [u64::MAX, (1_u64 << 62) + 1, 2] {
        // SAFETY: the buffer is valid for the supplied byte length. Invalid
        // element counts must be rejected before accessing it.
        let result = unsafe {
            retrograd_ffi::retro_optimizer_slot_initial_bytes(
                0,
                0,
                0,
                count,
                bytes.as_mut_ptr().cast(),
                bytes.len(),
            )
        };
        assert_ne!(result, 0, "accepted {count} elements in four bytes");
        assert_eq!(bytes, [0xA5; 4]);
    }
}

#[test]
fn a_codebook_can_be_written_to_an_unaligned_byte_buffer() {
    #[repr(align(4))]
    struct Aligned([u8; 13]);
    let mut storage = Aligned([0xA5_u8; 13]);
    let bytes = &mut storage.0[1..];
    // SAFETY: the byte buffer holds exactly three F32 values; the ABI does
    // not require the output pointer to be aligned as an F32 pointer.
    let result = unsafe {
        retrograd_ffi::retro_optimizer_slot_initial_bytes(
            0,
            2,
            0,
            3,
            bytes.as_mut_ptr().cast(),
            bytes.len(),
        )
    };
    assert_eq!(result, 0);
    assert_eq!(f32s(bytes), [-1.0, 0.0, 1.0]);
    assert_eq!(storage.0[0], 0xA5);
}
