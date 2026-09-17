//! Shape probes run on a private trainer and a throwaway adapter. No reward,
//! generation, dataset or checkpoint is touched by a timing measurement.

use std::path::Path;
use std::time::Instant;

use retrograd_core::{Error, LoraConfig, MemoryReport, Result, TrainConfig};
use retrograd_engine::Trainer;

use crate::rollout::packing::WeightedStepScratch;
use crate::rollout::sampling::{Rollout, RowLayout};
use crate::rollout::step::{
    ChunkMember, GrpoObjective, GrpoStepParams, PackingSelection, grpo_chunk_step, select_packing,
};

#[derive(Debug)]
pub enum PackingBenchmark {
    Measured {
        seconds: Vec<f64>,
        /// Boxed: the report dwarfs a refusal.
        memory: Box<MemoryReport>,
    },
    /// The geometry itself cannot train this shape: it falls back to rows, a
    /// locked fanout does not fit, or the loss is not finite. An `Err` from
    /// [`benchmark`] is anything else - loading, allocation, the runtime.
    Refused(String),
}

/// One untimed warm-up followed by three complete packed optimizer updates.
/// This is a synthetic shape benchmark, not a prediction of end-to-end rollout
/// speed (which also includes generation, rewards and tools). Each update
/// re-scores its members, as every update after a run's first one does.
pub fn benchmark(
    model: &Path,
    training: &TrainConfig,
    lora: &LoraConfig,
    prompt_tokens: u32,
    completion_tokens: u32,
    group_size: u32,
) -> Result<PackingBenchmark> {
    let length = prompt_tokens
        .checked_add(completion_tokens)
        .ok_or_else(|| Error::overflow("packing benchmark sequence length overflows u32"))?;
    if prompt_tokens < 2 || completion_tokens == 0 || group_size < 2 || length > training.n_ctx {
        return Err(Error::invalid(
            "packing benchmark requires a prompt, completions and a fitting logical context",
        ));
    }
    let mut trainer = Trainer::new(model, training.clone())?;
    trainer.create_lora(lora)?;
    let layout = RowLayout::resolve(&trainer, training)?;
    let vocabulary =
        trainer.tokenize_text("A shared prompt with several different training continuations.")?;
    if vocabulary.is_empty() {
        return Err(Error::tokenize("packing benchmark text produced no tokens"));
    }
    let prompt = prompt_tokens as usize;
    let completion = completion_tokens as usize;
    let rollouts: Vec<_> = (0..group_size)
        .map(|member| {
            let mut tokens: Vec<_> = vocabulary.iter().copied().cycle().take(prompt).collect();
            tokens.extend(
                vocabulary
                    .iter()
                    .copied()
                    .cycle()
                    .skip(member as usize)
                    .take(completion),
            );
            let mut train_mask = vec![false; prompt];
            train_mask.resize(prompt + completion, true);
            Rollout {
                tokens,
                train_mask,
                old_logprobs: vec![-1.0; completion],
            }
        })
        .collect();
    let chunk: Vec<_> = rollouts
        .iter()
        .map(|rollout| ChunkMember {
            group_id: 0,
            rollout,
            advantage: 1.0,
            token_advantages: None,
            reference_logprobs: &[],
        })
        .collect();
    let capability = trainer.supports_shared_prefix_packed_training()?;
    match select_packing(&chunk, &layout, capability) {
        Ok(PackingSelection::Packed(_)) => {}
        Ok(PackingSelection::Rows { reason, .. }) => {
            return Ok(PackingBenchmark::Refused(format!(
                "geometry falls back to rows: {reason}"
            )));
        }
        Err(error) => return Ok(PackingBenchmark::Refused(error.to_string())),
    }
    let params = GrpoStepParams {
        objective: GrpoObjective {
            clip_range_low: 0.2,
            clip_range_high: 0.2,
            kl_coefficient: 0.0,
            loss_denominator: completion,
        },
        scheduler_total_steps: training.warmup_steps.max(4),
    };
    let mut scratch = WeightedStepScratch::new(&layout);
    let mut seconds = Vec::with_capacity(3);
    for sample in 0..4 {
        let started = Instant::now();
        let (metrics, _, _) = grpo_chunk_step(
            &mut trainer,
            &chunk,
            &layout,
            params,
            false,
            &mut scratch,
            &mut |_, _| Ok(true),
        )?;
        let elapsed = started.elapsed().as_secs_f64();
        if !metrics.train_loss.is_finite() {
            return Ok(PackingBenchmark::Refused("non-finite loss".into()));
        }
        if sample > 0 {
            seconds.push(elapsed);
        }
    }
    Ok(PackingBenchmark::Measured {
        seconds,
        memory: Box::new(trainer.memory_report()?),
    })
}
