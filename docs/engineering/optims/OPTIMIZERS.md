# Optimizer cost and quality: Muon and Gefen, measured

Both optimizers are correct on every backend, every update is checked against
an independent F64 oracle, on real weights and on the gradient a backward
actually emitted. This document is the separate question of what they *cost*
and what Gefen's approximation gives up; neither is yet recommended over
AdamW, and the figures below say what that hesitation is worth.

Every figure below is reproducible with one binary:

```
cargo run --release --features cuda --bin optim-report -- \
    --model <f32-or-f16.gguf> --device gpu
```

It prints three sections and decides nothing. `quality` needs no model;
`cost` and `curves` need one whose trainable tensors are F32 or F16, because
base-weight training does not admit quantized weights.

## The reference run

- **Machine** NVIDIA GeForce RTX 4090 (24 GiB), CUDA 13.4, Linux.
- **Model** a generated qwen2 of 32 layers, `n_embd = 256`, `n_ff = 512`, F32
  throughout - `scripts/gen-tiny-fixture.py --layers 32 --embd 256 --heads 8
  --kv-heads 4 --ff 512`. The shape flags exist for this: the pinned unit
  fixture is four layers of sixty-four, and an optimizer whose cost per
  eligible matrix is the question needs hundreds of matrices, not three.
- **Selection** every attention and feed-forward matrix plus every norm: 289
  trainable tensors, 18 891 008 parameters.
- **Workload** `n_ctx = 512`, `n_ubatch = 128`, one 513-token window per step,
  one untimed warm-up step before the measured ones.

## Cost

```
  optimizer                     state  bytes/par  step (exec) step (build)         peak   matrices
  adamw                     144.1 MiB     8.0000       0.0523       0.0056    897.1 MiB        289
  muon                       72.1 MiB     4.0035       0.0921       0.0125    825.1 MiB        224
  gefen/shared_v/256         72.3 MiB     4.0156       0.0508       0.0059    825.1 MiB        289
  gefen/shared_v/1024        72.1 MiB     4.0039       0.0509       0.0060    825.1 MiB        289
  gefen/quantized_m/256      18.6 MiB     1.0312       0.0504       0.0061    771.1 MiB        289
  gefen/quantized_m/1024     18.2 MiB     1.0078       0.0508       0.0062    771.1 MiB        289
```

`step (exec)` is seconds of optimizer kernels per step and `step (build)` the
graph build plus backend allocation, both from the runtime's own counters;
`peak` is the device high-water measured *inside* the steps.

What the table says:

- **Gefen's state is what its arithmetic predicts.** `quantized_m` at B=1024
  measures 1.0078 bytes per parameter, which is `1 + 8/1024` exactly, on a run
  where every tensor is eligible (`matrices` 289 of 289).
- **Gefen's step costs what AdamW's costs.** 0.0508 s against 0.0523 s, on a
  step whose second phase writes a byte per parameter instead of eight. The
  saving is memory, not time, and the time is not a tax either.
- **Muon costs 1.8x AdamW in kernels and 2.2x in graph build.** The build cost
  is the interesting one: Muon is the only optimizer here whose step is a
  *graph*, five Newton-Schulz iterations per eligible matrix, and the build is
  paid per evaluation. At this parameter count it is 12 % of the step; the
  ratio grows with the matrix count and not with the matrix size.
- **Muon owns 224 of the 289 tensors.** The other 65 are the norms, which are
  one-dimensional and have no orthogonalization to do; they fall back to AdamW,
  which is why Muon's 4.0035 bytes per parameter is not exactly 4.

## Quality: what Gefen's approximation gives up

Model-free, from the two ops driven directly. 65 536 elements, 16 steps, a
gradient whose magnitudes span four orders of magnitude - a uniform gradient
would flatter every block size equally and say nothing. The comparison is
against AdamW in F64 on the *same* weights, gradient and coefficients.

```
  variant        block      cosine   rel.error   recon.err     v.error  bytes/par
  shared_v           1    1.000000    1.875e-6     0.000e0    2.848e-8     8.0000
  shared_v          64    0.854753    5.375e-1     0.000e0     2.169e0     4.0625
  shared_v         256    0.853909    5.390e-1     0.000e0     2.185e0     4.0156
  shared_v        1024    0.853613    5.396e-1     0.000e0     2.190e0     4.0039
  shared_v        4096    0.853535    5.398e-1     0.000e0     2.191e0     4.0010
  quantized_m        1    1.000000    1.875e-6    5.912e-8    2.848e-8     9.0000
  quantized_m       64    0.854737    5.375e-1    1.456e-2     2.169e0     1.1250
  quantized_m      256    0.853899    5.391e-1    1.749e-2     2.185e0     1.0312
  quantized_m     1024    0.853563    5.398e-1    1.995e-2     2.190e0     1.0078
  quantized_m     4096    0.853667    5.396e-1    2.198e-2     2.191e0     1.0020
```

- `cosine` of Gefen's update against AdamW's, over the whole parameter.
- `rel.error` is `||gefen - adamw|| / ||adamw||` on the same update.
- `recon.err` is `||decode(stored) - m|| / ||m||`: the quantized first moment
  alone, against the unquantized moment the step actually used.
- `v.error` is the mean of `|sqrt(v_block) - sqrt(v_element)| / sqrt(v_element)`:
  the coordinate scaling a shared second moment gives up by design.

Four things this says, in order of how much they should change a decision:

**The block second moment is the whole approximation; the quantized first
moment is nearly free.** At B=1024 the reconstruction error of the byte-indexed
moment is 2.0 %, and the update's error against AdamW is 54 % - the same 54 %
`shared_v` has, which stores that moment in full F32. The two variants agree to
five digits on every quality column. So the choice between them is a choice
about *bytes*, 4.0039 against 1.0078 per parameter, and not about accuracy.

**A universal cosine above 0.99 was never the target and the numbers say why.**
`v.error` is 2.19 at every block size above one: the per-block denominator is
about three times the per-element one on a heavy-tailed gradient. That is what
a shared second moment *is*. Reading 0.854 as "85 % correct" would be reading
the design as a defect.

**Block size barely matters above 64.** Between B=64 and B=4096 the cosine
moves by 0.0012 and the reconstruction error by 0.7 points, while the bytes per
parameter fall from 1.125 to 1.002. There is no accuracy argument for a small
block; the frozen default of 1024 sits where the byte curve has already
flattened.

**B=1 is the anchor and it holds.** `shared_v` at one element per block keeps
AdamW's own second moment, and the measured relative error is 1.9e-6 - F32
noise. The quantized variant at B=1 adds a reconstruction error of 5.9e-8,
because a block of one quantizes its single value against its own magnitude and
lands on the codebook's top entry exactly.

### The adversarial block

One element `10^6` above its neighbours, B=1024, one step:

```
  variant           cosine   rel.error     cos(large)     cos(small)
  shared_v        0.031292     1.392e0       1.000000       0.867330
  quantized_m     0.031292     1.392e0       1.000000       0.867330
```

The whole-parameter cosine collapses to 0.031 because the single large
coordinate carries the norm and Gefen's denominator for it is the *block's*,
not its own. Split by coordinate, the outlier's direction is exact and the
small ones are at 0.867 - about where the uniform case puts them. The two
variants are identical to six digits here because the state starts at zero: a
first step decodes a zero moment whatever the variant, so this row measures the
block scale alone and not the quantization.

## Loss

12 steps, 3 seeds, the same selection and device. The weights come from a file,
so there is no initialization for a seed to vary; each seed starts at a
different offset in the corpus and sees the windows in a different order, which
is the only independent axis a fixed checkpoint leaves.

```
  optimizer                      lr       loss@0    loss@last       spread
  adamw                      1.0e-4      5.93777      3.34994      0.06587
  muon                       2.0e-2      5.93777      1.23219      0.08684
  gefen/shared_v/256         1.0e-4      5.93777      3.20543      0.03881
  gefen/shared_v/1024        1.0e-4      5.93777      3.18026      0.03678
  gefen/quantized_m/256      1.0e-4      5.93777      3.20548      0.03865
  gefen/quantized_m/1024     1.0e-4      5.93777      3.18039      0.03675
```

`spread` is the range of `loss@last` across the seeds, and it is the bar any
difference has to clear.

- **Gefen reaches 3.18 where AdamW reaches 3.35, at the same learning rate.**
  The gap is 0.17 against a spread of 0.04, so it is not noise at this scale.
  It is also not a recommendation: twelve steps on one generated model is a
  smoke test of the direction, not a task-quality threshold.
- **The four Gefen rows are within 0.025 of each other**, which is the quality
  table restated: the variant is a byte count and the block size is nearly
  free.
- **Muon's 1.23 is at a different learning rate and does not compare.** Muon's
  update is orthogonalized, so its scale is not AdamW's and the two cannot
  share a rate; the column is printed for exactly that reason.

## What is still open

- **Task quality.** Nothing here is a downstream evaluation, and the task
  quality thresholds that would promote one of these over AdamW are not yet
  established. That is why both stay opt-in.
- **A discrete-GPU memory story.** The `peak` column is a device high-water on
  one 24 GiB card with a 19 M-parameter model. The interesting regime - where
  the optimizer state is a real fraction of the budget - is not this one.
- **Muon on LoRA factors.** A separate experiment, and not mathematically
  equivalent to orthogonalizing `BA`.
- **Gefen-Muon.** Not implemented.
