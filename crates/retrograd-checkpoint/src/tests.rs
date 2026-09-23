use super::*;

fn sample() -> Checkpoint {
    Checkpoint {
        manifest: Manifest {
            format_version: FORMAT_VERSION,
            checkpoint_id: "step-000000000042".into(),
            global_step: 42,
            adapter: Some(ADAPTER_FILE.into()),
            trainable: None,
            trainable_policy: "lora".into(),
            files: REQUIRED_FILES.iter().map(|name| name.to_string()).collect(),
            app_version: "0.1.0".into(),
            llama_cpp_commit: "deadbeef".into(),
            model_signature: "arch=llama n_embd=64".into(),
            model_bytes: 1234,
            model_fingerprint: "model-content".into(),
            reference_fingerprint: "anchor-content".into(),
            algorithm: "sft".into(),
            trajectory_signature: "trajectory-1".into(),
            resume_boundary: "epoch".into(),
            artifacts: BTreeMap::new(),
        },
        progress: Progress {
            version: FORMAT_VERSION,
            epoch: 2,
            global_step: 42,
            cursor: 7,
            algorithm: "sft".into(),
            phase: "train".into(),
            best_eval: Some(0.25),
            stale_evaluations: 1,
            kl_multiplier: None,
        },
        scheduler: Scheduler {
            version: FORMAT_VERSION,
            step: 42,
            total_steps: 100,
            last_learning_rate: 1.0e-4,
            kind: "cosine".into(),
            learning_rate: 2.0e-4,
            warmup_steps: 5,
        },
        optimizer: Optimizer {
            version: FORMAT_VERSION,
            kind: "adamw".into(),
            layout_version: 1,
            hyperparameters: sample_hyperparameters(),
            learning_rate: 2.0e-4,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            iter: 43,
            graph_ready: true,
            slots: vec![
                StateSlot {
                    scope: SlotScope::Parameter,
                    owner: "blk.0.attn_q.weight.lora_a".into(),
                    slot: "m".into(),
                    dtype: "F32".into(),
                    shape: [4, 8, 1, 1],
                    offset: 0,
                    n_bytes: 128,
                },
                StateSlot {
                    scope: SlotScope::Parameter,
                    owner: "blk.0.attn_q.weight.lora_a".into(),
                    slot: "v".into(),
                    dtype: "F32".into(),
                    shape: [4, 8, 1, 1],
                    offset: 128,
                    n_bytes: 128,
                },
            ],
            assignment: vec![ParameterAssignment {
                parameter: "blk.0.attn_q.weight.lora_a".into(),
                optimizer: "adamw".into(),
                layout_version: 1,
            }],
            state_bytes: 256,
        },
        rng: Rng {
            version: FORMAT_VERSION,
            runtime_mt19937: Some("1 2 3".into()),
            seeds: BTreeMap::from([("sampling".into(), 99_u64)]),
        },
        dataset: Dataset {
            version: FORMAT_VERSION,
            path: "data/train.jsonl".into(),
            fingerprint: fingerprint(b"rows"),
            examples: 10,
            row_width: 256,
            format: "chat".into(),
            permutation: vec![2, 0, 1],
            cursor: 7,
        },
        artifacts: BTreeMap::new(),
    }
}

/// AdamW's declared vector, spelled out here rather than imported so a
/// layout change fails the comparisons in this crate.
fn sample_hyperparameters() -> Vec<String> {
    [
        "learning_rate=0.0002",
        "beta1=0.9",
        "beta2=0.999",
        "eps=1e-8",
        "weight_decay=0.01",
        "max_grad_norm=1.0",
    ]
    .iter()
    .map(|line| (*line).to_string())
    .collect()
}

fn compatibility() -> Compatibility {
    Compatibility {
        model_signature: "arch=llama n_embd=64".into(),
        model_bytes: 1234,
        model_fingerprint: "model-content".into(),
        reference_fingerprint: "anchor-content".into(),
        algorithm: "sft".into(),
        trajectory_signature: "trajectory-1".into(),
        dataset_fingerprint: fingerprint(b"rows"),
        scheduler_kind: "cosine".into(),
        learning_rate: 2.0e-4,
        warmup_steps: 5,
        total_steps: Some(100),
        optimizer_kind: "adamw".into(),
        optimizer_layout_version: 1,
        optimizer_hyperparameters: sample_hyperparameters(),
        weight_decay: 0.01,
        max_grad_norm: 1.0,
        trainable_policy: "lora".into(),
        trainable_signature: String::new(),
    }
}

/// Writes the payloads the sample manifest declares: a stand-in adapter and
/// a slot payload of exactly the declared length.
fn write_sample_payloads(marker: &[u8]) -> impl FnOnce(&CheckpointPaths) -> Result<()> + '_ {
    move |paths: &CheckpointPaths| {
        if let Some(path) = &paths.adapter {
            fs::write(path, marker)?;
        }
        if let Some(path) = &paths.trainable {
            fs::write(path, b"TRAINABLE")?;
        }
        if let Some(path) = &paths.optimizer_state {
            fs::write(path, vec![7_u8; 256])?;
        }
        Ok(())
    }
}

#[test]
fn every_state_file_round_trips_through_messagepack() {
    let checkpoint = sample();
    // Each file is its own schema, so each one is checked on its own.
    assert_eq!(
        decode::<Manifest>(&encode(&checkpoint.manifest, "m").unwrap(), "m").unwrap(),
        checkpoint.manifest
    );
    assert_eq!(
        decode::<Progress>(&encode(&checkpoint.progress, "p").unwrap(), "p").unwrap(),
        checkpoint.progress
    );
    assert_eq!(
        decode::<Scheduler>(&encode(&checkpoint.scheduler, "s").unwrap(), "s").unwrap(),
        checkpoint.scheduler
    );
    assert_eq!(
        decode::<Optimizer>(&encode(&checkpoint.optimizer, "o").unwrap(), "o").unwrap(),
        checkpoint.optimizer
    );
    assert_eq!(
        decode::<Rng>(&encode(&checkpoint.rng, "r").unwrap(), "r").unwrap(),
        checkpoint.rng
    );
    assert_eq!(
        decode::<Dataset>(&encode(&checkpoint.dataset, "d").unwrap(), "d").unwrap(),
        checkpoint.dataset
    );
}

#[test]
fn a_written_checkpoint_reads_back_identically() {
    let root = tempdir();
    let state = root.join("step-000000000042.state");
    let checkpoint = sample();
    checkpoint
        .write(&state, write_sample_payloads(b"GGUF"))
        .unwrap();
    // Resume owns an atomic adapter inside the directory, while the sibling
    // remains a conventional cold-load export.
    assert!(state.join(ADAPTER_FILE).is_file());
    assert!(root.join("step-000000000042.gguf").is_file());
    assert_eq!(Checkpoint::read(&state).unwrap(), checkpoint);
}

#[test]
fn replacing_a_checkpoint_never_mixes_adapter_and_state() {
    let root = tempdir();
    let state = root.join("best.state");
    let first = sample();
    first
        .write(&state, write_sample_payloads(b"FIRST"))
        .unwrap();

    let mut second = sample();
    second.progress.best_eval = Some(0.5);
    second
        .write(&state, write_sample_payloads(b"SECOND"))
        .unwrap();

    assert_eq!(Checkpoint::read(&state).unwrap(), second);
    assert_eq!(std::fs::read(state.join(ADAPTER_FILE)).unwrap(), b"SECOND");
    assert_eq!(std::fs::read(root.join("best.gguf")).unwrap(), b"SECOND");
    assert!(!backup_dir_for(&state).unwrap().exists());
}

#[test]
fn an_interrupted_replacement_falls_back_to_the_complete_backup() {
    let root = tempdir();
    let state = root.join("best.state");
    let checkpoint = sample();
    checkpoint
        .write(&state, write_sample_payloads(b"FIRST"))
        .unwrap();
    let backup = backup_dir_for(&state).unwrap();
    std::fs::rename(&state, &backup).unwrap();

    assert_eq!(Checkpoint::read(&state).unwrap(), checkpoint);
    assert_eq!(
        std::fs::read(adapter_for(&state, &checkpoint.manifest).unwrap()).unwrap(),
        b"FIRST"
    );
}

#[test]
fn a_failing_adapter_write_leaves_nothing_behind() {
    let root = tempdir();
    let state = root.join("step-000000000042.state");
    let error = sample()
        .write(&state, |_| Err(Error::runtime("disk on fire")))
        .unwrap_err();
    assert!(error.to_string().contains("disk on fire"));
    assert!(!state.exists());
    assert!(!root.join("step-000000000042.gguf").exists());
    // No temporary residue either: an interrupted write is invisible.
    assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
}

#[test]
fn a_directory_without_a_manifest_is_not_a_checkpoint() {
    let root = tempdir();
    let state = root.join("step-000000000042.state");
    let checkpoint = sample();
    checkpoint
        .write(&state, write_sample_payloads(b"GGUF"))
        .unwrap();
    std::fs::remove_file(state.join(MANIFEST_FILE)).unwrap();
    let error = Checkpoint::read(&state).unwrap_err();
    assert!(error.to_string().contains("not a complete checkpoint"));
}

#[test]
fn a_declared_state_file_that_is_missing_is_refused() {
    let root = tempdir();
    let state = root.join("step-000000000042.state");
    sample()
        .write(&state, write_sample_payloads(b"GGUF"))
        .unwrap();
    std::fs::remove_file(state.join(OPTIMIZER_FILE)).unwrap();
    let error = Checkpoint::read(&state).unwrap_err();
    assert!(error.to_string().contains(OPTIMIZER_FILE));
}

#[test]
fn an_unknown_schema_version_is_refused() {
    let root = tempdir();
    let state = root.join("step-000000000042.state");
    let mut checkpoint = sample();
    checkpoint.manifest.format_version = FORMAT_VERSION + 1;
    checkpoint
        .write(&state, write_sample_payloads(b"GGUF"))
        .unwrap();
    let error = Checkpoint::read(&state).unwrap_err();
    assert!(error.to_string().contains("schema version"));
}

#[test]
fn a_required_artifact_that_is_missing_is_refused_but_a_recreatable_one_is_not() {
    let root = tempdir();
    let mut checkpoint = sample();
    checkpoint
        .manifest
        .artifacts
        .insert("ppo_value.msgpack".into(), ArtifactPolicy::Required);
    checkpoint
        .manifest
        .artifacts
        .insert("metrics.msgpack".into(), ArtifactPolicy::Recreatable);
    checkpoint
        .artifacts
        .insert("ppo_value.msgpack".into(), b"value".to_vec());
    checkpoint
        .artifacts
        .insert("metrics.msgpack".into(), b"metrics".to_vec());

    let state = root.join("step-000000000042.state");
    checkpoint
        .write(&state, write_sample_payloads(b"GGUF"))
        .unwrap();

    std::fs::remove_file(state.join(ARTIFACTS_DIR).join("metrics.msgpack")).unwrap();
    let loaded = Checkpoint::read(&state).unwrap();
    assert!(!loaded.artifacts.contains_key("metrics.msgpack"));

    std::fs::remove_file(state.join(ARTIFACTS_DIR).join("ppo_value.msgpack")).unwrap();
    let error = Checkpoint::read(&state).unwrap_err();
    assert!(error.to_string().contains("required"));
}

/// "Keeps no state" and "was never initialized" are different states, and a
/// resume must not read the first as the second.
#[test]
fn a_zero_slot_optimizer_still_counts_as_initialized() {
    let mut checkpoint = sample();
    checkpoint.optimizer.kind = "sgd".into();
    checkpoint.optimizer.graph_ready = true;
    checkpoint.optimizer.slots.clear();
    checkpoint.optimizer.state_bytes = 0;
    checkpoint.optimizer.assignment[0].optimizer = "sgd".into();

    let mut expected = compatibility();
    expected.optimizer_kind = "sgd".into();
    checkpoint.check_compatible(&expected).unwrap();
    // ... and the optimizer it names is still compared, precisely because
    // the empty slot list no longer says which one wrote it.
    let error = checkpoint
        .check_compatible(&compatibility())
        .expect_err("adamw cannot resume an sgd trajectory");
    assert!(error.to_string().contains("optimizer was sgd"), "{error}");
}

/// A cold checkpoint pins no trajectory, so the optimizer it names is not
/// compared: nothing was allocated under it and nothing counted steps.
#[test]
fn a_checkpoint_taken_before_the_graph_existed_pins_no_optimizer() {
    let mut checkpoint = sample();
    checkpoint.optimizer.graph_ready = false;
    checkpoint.optimizer.slots.clear();
    checkpoint.optimizer.state_bytes = 0;
    checkpoint.optimizer.kind = "sgd".into();
    checkpoint.check_compatible(&compatibility()).unwrap();
}

/// The slot table addresses a byte range in a separate file, so its own
/// consistency is checked before anything reads through it.
#[test]
fn a_slot_table_that_does_not_tile_its_payload_is_refused() {
    let mut gapped = sample().optimizer;
    gapped.slots[1].offset = 192;
    let error = validate_slot_table(&gapped).unwrap_err();
    assert!(
        error.to_string().contains("payload continues at 128"),
        "{error}"
    );

    let mut duplicated = sample().optimizer;
    duplicated.slots[1].slot = "m".into();
    duplicated.slots[1].offset = 128;
    let error = validate_slot_table(&duplicated).unwrap_err();
    assert!(error.to_string().contains("two 'm' slots"), "{error}");

    let mut miscounted = sample().optimizer;
    miscounted.state_bytes = 512;
    let error = validate_slot_table(&miscounted).unwrap_err();
    assert!(error.to_string().contains("covers 256 bytes"), "{error}");
}

/// The temporary AdamW accessor exists for callers that still think in
/// momenta, and refuses every other optimizer rather than reading slots 0
/// and 1 of an unknown layout as a pair.
#[test]
fn the_adamw_accessor_refuses_another_optimizers_slots() {
    let checkpoint = sample();
    let (m, v) = checkpoint
        .optimizer
        .adamw_moments("blk.0.attn_q.weight.lora_a")
        .unwrap();
    assert_eq!((m.slot.as_str(), v.slot.as_str()), ("m", "v"));
    assert_eq!((m.offset, v.offset), (0, 128));

    let mut renamed = checkpoint.optimizer.clone();
    renamed.kind = "sgd".into();
    let error = renamed
        .adamw_moments("blk.0.attn_q.weight.lora_a")
        .unwrap_err();
    assert!(
        error.to_string().contains("keeps no AdamW momenta"),
        "{error}"
    );
}

/// A slot payload is read back in bounded chunks, and a range outside the
/// file is an error rather than a short read.
#[test]
fn a_slot_payload_streams_back_through_a_bounded_buffer() {
    let root = tempdir();
    let state = root.join("step-000000000042.state");
    let checkpoint = sample();
    checkpoint
        .write(&state, write_sample_payloads(b"GGUF"))
        .unwrap();

    let mut reader = OptimizerStateReader::open(&state, &checkpoint.optimizer).unwrap();
    let mut staging = vec![0_u8; 48];
    let mut seen = Vec::new();
    reader
        .stream(
            &checkpoint.optimizer.slots[1],
            &mut staging,
            |offset, chunk| {
                seen.push((offset, chunk.len()));
                assert!(chunk.iter().all(|byte| *byte == 7));
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(seen, [(0, 48), (48, 48), (96, 32)]);

    let past_the_end = StateSlot {
        offset: 200,
        n_bytes: 128,
        ..checkpoint.optimizer.slots[0].clone()
    };
    let error = reader
        .stream(&past_the_end, &mut staging, |_, _| Ok(()))
        .unwrap_err();
    assert!(error.to_string().contains("ends at 328"), "{error}");
}

/// The shared scope at a length above zero. Every built-in optimizer
/// declares none, so this synthetic row is the only way to exercise it:
#[test]
fn a_shared_slot_round_trips_beside_the_parameter_ones() {
    let root = tempdir();
    let state = root.join("step-000000000042.state");
    let mut checkpoint = sample();
    // 256 F32 codebook entries, owned by the optimizer, not by any
    // parameter.
    let codebook = StateSlot {
        scope: SlotScope::Shared,
        owner: "gefen".into(),
        slot: "codebook".into(),
        dtype: "F32".into(),
        shape: [256, 1, 1, 1],
        offset: 256,
        n_bytes: 1024,
    };
    checkpoint.optimizer.slots.push(codebook.clone());
    checkpoint.optimizer.state_bytes = 256 + 1024;
    checkpoint
        .write(&state, |paths| {
            if let Some(path) = &paths.adapter {
                fs::write(path, b"GGUF")?;
            }
            let path = paths
                .optimizer_state
                .as_ref()
                .expect("the record declares a payload");
            let mut payload = vec![7_u8; 256];
            payload.extend(std::iter::repeat_n(3_u8, 1024));
            fs::write(path, payload)?;
            Ok(())
        })
        .unwrap();

    let read = Checkpoint::read(&state).unwrap();
    assert_eq!(read.optimizer.slots, checkpoint.optimizer.slots);
    // Shared and parameter rows tile one payload, so a shared row is
    // streamed like a parameter one.
    let mut reader = OptimizerStateReader::open(&state, &read.optimizer).unwrap();
    let mut staging = vec![0_u8; 512];
    let mut seen = Vec::new();
    reader
        .stream(&codebook, &mut staging, |offset, chunk| {
            seen.push((offset, chunk.len()));
            assert!(chunk.iter().all(|byte| *byte == 3));
            Ok(())
        })
        .unwrap();
    assert_eq!(seen, [(0, 512), (512, 512)]);
    let _ = fs::remove_dir_all(&root);
}

/// The scope is part of a slot's identity, so the table rules hold across
/// scopes, not within each.
#[test]
fn the_scope_is_part_of_a_slots_identity_in_the_table() {
    let mut duplicated = sample().optimizer;
    // Same owner and slot name in both scopes: two rows, accepted.
    duplicated.slots.push(StateSlot {
        scope: SlotScope::Shared,
        owner: "blk.0.attn_q.weight.lora_a".into(),
        slot: "m".into(),
        dtype: "F32".into(),
        shape: [4, 8, 1, 1],
        offset: 256,
        n_bytes: 128,
    });
    duplicated.state_bytes = 384;
    validate_slot_table(&duplicated).unwrap();
    // The same row twice within the shared scope is not.
    let mut twice = duplicated.clone();
    twice.slots.push(StateSlot {
        offset: 384,
        ..duplicated.slots[2].clone()
    });
    twice.state_bytes = 512;
    let error = validate_slot_table(&twice).unwrap_err().to_string();
    assert!(error.contains("two 'm' slots for shared"), "{error}");
    // A shared row that does not continue the payload is a gap, as with a
    // parameter row.
    let mut gapped = duplicated.clone();
    gapped.slots[2].offset = 300;
    let error = validate_slot_table(&gapped).unwrap_err().to_string();
    assert!(error.contains("continues at 256"), "{error}");
}

/// A run that trains base weights leaves a bundle and no adapter, and the
/// sibling GGUF export - which every helper reads as an adapter - is not
/// published for it.
#[test]
fn a_base_weight_checkpoint_carries_a_bundle_and_publishes_no_adapter() {
    let root = tempdir();
    let state = root.join("step-000000000042.state");
    let mut checkpoint = sample();
    checkpoint.manifest.adapter = None;
    checkpoint.manifest.trainable_policy = "partial".into();
    checkpoint.manifest.trainable = Some(TrainableBundle {
        file: TRAINABLE_FILE.into(),
        // Filled in by the writer: the size and identity are only known
        // once the payload exists.
        bytes: 0,
        fingerprint: String::new(),
        signature: "set-1".into(),
        tensors: vec![TrainableTensor {
            name: "blk.0.attn_norm.weight".into(),
            role: "base".into(),
            dtype: "F32".into(),
            shape: [64, 1, 1, 1],
            n_elements: 64,
            n_bytes: 256,
            aliases: Vec::new(),
        }],
    });
    checkpoint
        .write(&state, write_sample_payloads(b"GGUF"))
        .unwrap();

    assert!(state.join(TRAINABLE_FILE).is_file());
    assert!(!state.join(ADAPTER_FILE).exists());
    assert!(!root.join("step-000000000042.gguf").exists());
    assert_eq!(adapter_for(&state, &checkpoint.manifest), None);
    assert_eq!(
        trainable_for(&state, &checkpoint.manifest),
        Some(state.join(TRAINABLE_FILE))
    );

    // Publication finalizes the bundle's size and identity from the file
    // the payload writer produced.
    let published = Checkpoint::read(&state).unwrap();
    let bundle = published.manifest.trainable.as_ref().unwrap();
    assert_eq!(bundle.bytes, 9);
    assert_eq!(
        bundle.fingerprint,
        fingerprint_file(&state.join(TRAINABLE_FILE)).unwrap()
    );
    // ... and nothing else moved.
    let mut expected = checkpoint.clone();
    let declared = expected.manifest.trainable.as_mut().unwrap();
    declared.bytes = bundle.bytes;
    declared.fingerprint.clone_from(&bundle.fingerprint);
    assert_eq!(published, expected);

    // And the resolved set is compared, not just the policy name.
    let mut expected = compatibility();
    expected.trainable_policy = "partial".into();
    expected.trainable_signature = "set-1".into();
    checkpoint.check_compatible(&expected).unwrap();
    expected.trainable_signature = "set-2".into();
    let error = checkpoint.check_compatible(&expected).unwrap_err();
    assert!(error.to_string().contains("trainable set"), "{error}");
}

/// A declared payload the writer did not produce is refused at write time:
/// the manifest is written from the declaration, so publishing anyway would
/// describe a checkpoint that is not there.
#[test]
fn a_payload_the_writer_skipped_is_refused_before_publication() {
    let root = tempdir();
    let state = root.join("step-000000000042.state");
    let error = sample()
        .write(&state, |paths| {
            fs::write(paths.adapter.as_ref().unwrap(), b"GGUF")?;
            Ok(())
        })
        .unwrap_err();
    assert!(error.to_string().contains("optimizer state"), "{error}");
    assert!(!state.exists());

    let error = sample()
        .write(&state, |paths| {
            fs::write(paths.adapter.as_ref().unwrap(), b"GGUF")?;
            fs::write(paths.optimizer_state.as_ref().unwrap(), vec![0_u8; 8])?;
            Ok(())
        })
        .unwrap_err();
    assert!(error.to_string().contains("8 bytes"), "{error}");
    assert!(!state.exists());
}

#[test]
fn a_matching_configuration_is_compatible() {
    sample().check_compatible(&compatibility()).unwrap();
}

#[test]
fn every_trajectory_changing_difference_is_refused() {
    // Each case is something a silent resume would corrupt: a different
    // model, dataset, schedule, or optimizer regularization.
    type Case = (&'static str, fn(&mut Compatibility));
    let cases: [Case; 12] = [
        ("model", |c| c.model_signature = "arch=qwen3".into()),
        ("model size", |c| c.model_bytes = 999),
        ("model content", |c| c.model_fingerprint = "other".into()),
        ("fixed reference", |c| {
            c.reference_fingerprint = "other".into()
        }),
        // A run that dropped its anchor resumes onto a penalty against
        // something else.
        ("fixed reference", |c| {
            c.reference_fingerprint = String::new()
        }),
        ("algorithm", |c| c.algorithm = "grpo".into()),
        ("training trajectory", |c| {
            c.trajectory_signature = "other".into()
        }),
        ("dataset", |c| c.dataset_fingerprint = "0".into()),
        ("learning-rate schedule", |c| c.learning_rate = 1.0e-3),
        ("learning-rate schedule", |c| c.total_steps = Some(200)),
        ("optimizer hyperparameters", |c| c.weight_decay = 0.5),
        ("optimizer", |c| c.optimizer_kind = "sgd".into()),
    ];
    for (field, mutate) in cases {
        let mut expected = compatibility();
        mutate(&mut expected);
        let error = sample().check_compatible(&expected).unwrap_err();
        assert!(
            error.to_string().contains(field),
            "expected a {field} mismatch, got {error}"
        );
    }
}

/// The name, the layout version and the vector pin three different things;
/// a resume comparing only the first would restore a payload written under
/// different arithmetic.
#[test]
fn a_changed_layout_or_coefficient_is_refused_by_the_row_that_moved() {
    let mut expected = compatibility();
    expected.optimizer_layout_version = 2;
    let error = sample().check_compatible(&expected).unwrap_err();
    assert!(error.to_string().contains("layout version"), "{error}");

    // A coefficient no document spells moved: the run configuration is
    // unchanged, the trajectory is not.
    let mut expected = compatibility();
    expected.optimizer_hyperparameters[1] = "beta1=0.95".into();
    let error = sample().check_compatible(&expected).unwrap_err();
    assert!(error.to_string().contains("beta1=0.9,"), "{error}");
    assert!(error.to_string().contains("beta1=0.95"), "{error}");

    // A layout that lost a knob is reported against its absence.
    let mut expected = compatibility();
    expected.optimizer_hyperparameters.pop();
    let error = sample().check_compatible(&expected).unwrap_err();
    assert!(error.to_string().contains("absent"), "{error}");
}

/// Like the name, a cold checkpoint commits to no layout and no
/// coefficients: it allocated nothing and counted no step.
#[test]
fn a_cold_checkpoint_pins_neither_layout_nor_coefficients() {
    let mut checkpoint = sample();
    checkpoint.optimizer.graph_ready = false;
    checkpoint.optimizer.layout_version = 7;
    checkpoint.optimizer.hyperparameters = vec!["beta1=0.5".into()];
    checkpoint.check_compatible(&compatibility()).unwrap();
}

#[test]
fn parameter_assignment_pins_owners_and_layouts_without_requiring_order() {
    let optimizer = sample().optimizer;
    optimizer.check_assignment(&optimizer.assignment).unwrap();
    for field in ["optimizer", "layout", "parameter"] {
        let mut changed = optimizer.assignment.clone();
        match field {
            "optimizer" => changed[0].optimizer = "sgd".into(),
            "layout" => changed[0].layout_version += 1,
            _ => changed[0].parameter = "another.weight".into(),
        }
        assert!(optimizer.check_assignment(&changed).is_err());
    }
    assert!(optimizer.check_assignment(&[]).is_err());
    let mut two = optimizer.clone();
    let mut second = two.assignment[0].clone();
    second.parameter = "another.weight".into();
    two.assignment.push(second);
    let mut reversed = two.assignment.clone();
    reversed.reverse();
    two.check_assignment(&reversed).unwrap();
    reversed[0] = reversed[1].clone();
    assert!(two.check_assignment(&reversed).is_err());
}

#[test]
fn a_progress_and_scheduler_step_disagreement_is_refused() {
    let mut checkpoint = sample();
    checkpoint.progress.global_step = 41;
    let error = checkpoint.check_compatible(&compatibility()).unwrap_err();
    assert!(error.to_string().contains("inconsistent"));
}

#[test]
fn state_directories_resolve_from_either_the_adapter_or_themselves() {
    let gguf = Path::new("/runs/step-000000000042.gguf");
    let state = Path::new("/runs/step-000000000042.state");
    assert_eq!(state_dir_for(gguf), state);
    assert_eq!(state_dir_for(state), state);
    // The manifest keeps a relative name, so a moved directory still
    // resolves its own adapter.
    assert_eq!(
        adapter_for(state, &sample().manifest),
        Some(state.join(ADAPTER_FILE))
    );
}

#[test]
fn fingerprints_separate_content_and_order() {
    assert_eq!(fingerprint(b"abc"), fingerprint(b"abc"));
    assert_ne!(fingerprint(b"abc"), fingerprint(b"abd"));
    assert_ne!(fingerprint(b"abc"), fingerprint(b"cba"));
    assert_ne!(fingerprint(b""), fingerprint(b"\0"));
}

#[test]
fn a_cached_file_fingerprint_answers_the_hash_and_follows_the_content() {
    let dir = tempdir();
    let model = dir.join("model.gguf");
    fs::write(&model, b"weights-a").unwrap();
    let direct = fingerprint_file(&model).unwrap();
    assert_eq!(fingerprint_file_cached(&model).unwrap(), direct);
    // Same call twice is the cache hit the checkpoint loop depends on.
    assert_eq!(fingerprint_file_cached(&model).unwrap(), direct);

    // A different length invalidates the entry whatever the clock did.
    fs::write(&model, b"weights-a-longer").unwrap();
    let replaced = fingerprint_file_cached(&model).unwrap();
    assert_ne!(replaced, direct);
    assert_eq!(replaced, fingerprint_file(&model).unwrap());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_footprint_pays_for_the_adapter_twice_and_the_bundle_once() {
    // The adapter is copied out beside the directory; a bundle has no
    // sibling.
    let adapter = CheckpointFootprint {
        adapter_bytes: 1_000,
        trainable_bytes: 0,
        optimizer_state_bytes: 4_000,
    };
    assert_eq!(adapter.bytes(), 2_000 + 4_000 + CHECKPOINT_METADATA_BYTES);
    let base = CheckpointFootprint {
        adapter_bytes: 0,
        trainable_bytes: 1_000,
        optimizer_state_bytes: 4_000,
    };
    assert_eq!(base.bytes(), 1_000 + 4_000 + CHECKPOINT_METADATA_BYTES);
}

#[test]
fn a_budget_charges_the_retained_checkpoints_and_the_one_being_written() {
    let budget = DiskBudget {
        footprint: CheckpointFootprint {
            adapter_bytes: 0,
            trainable_bytes: 6 * 1024 * 1024,
            optimizer_state_bytes: 0,
        },
        retained: 3,
    };
    let per_checkpoint = 6 * 1024 * 1024 + CHECKPOINT_METADATA_BYTES;
    assert_eq!(budget.required_bytes(), per_checkpoint * 4);
    assert_eq!(budget.affordable(per_checkpoint * 4), 4);
    // One byte short of a fourth checkpoint fits three.
    assert_eq!(budget.affordable(per_checkpoint * 4 - 1), 3);
}

#[test]
fn a_budget_larger_than_the_filesystem_is_refused_and_names_what_fits() {
    let dir = tempdir();
    let free = free_space(&dir).unwrap();
    assert!(free > 0, "a writable scratch directory has free space");
    let budget = DiskBudget {
        footprint: CheckpointFootprint {
            adapter_bytes: free,
            trainable_bytes: 0,
            optimizer_state_bytes: 0,
        },
        retained: 0,
    };
    let error = budget
        .check(&dir, "cannot write this checkpoint")
        .unwrap_err();
    let message = error.to_string();
    assert!(matches!(error, Error::Checkpoint(_)), "{message}");
    assert!(
        message.contains("cannot write this checkpoint"),
        "{message}"
    );
    assert!(message.contains("of which 0 fit"), "{message}");
    // Widening the cadence is not a fix; the message says so.
    assert!(message.contains("cadence"), "{message}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn an_empty_footprint_is_affordable_and_a_free_figure_comes_from_an_ancestor() {
    let dir = tempdir();
    // A not-yet-existing subdirectory is measured against its filesystem.
    // The figure is not compared: it is a live machine's, and it moves
    // between two calls.
    let unborn = dir.join("checkpoints").join("deeper");
    assert!(free_space(&unborn).unwrap() > 0);
    // Even the empty footprint carries the metadata.
    let empty = DiskBudget::default();
    assert_eq!(empty.required_bytes(), CHECKPOINT_METADATA_BYTES);
    assert_eq!(empty.affordable(0), 0);
    empty.check(&unborn, "nothing to write").unwrap();
    let _ = fs::remove_dir_all(&dir);
}

/// Unique scratch directory; removed by the OS, not worth a dependency.
fn tempdir() -> PathBuf {
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let index = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "retrograd-checkpoint-{}-{index}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}
