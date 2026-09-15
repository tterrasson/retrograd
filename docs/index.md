---
layout: home

hero:
  name: Retrograd
  text: LoRA training from GGUF models
  tagline: Train a LoRA adapter on a GGUF model with SFT, PPO, or GRPO. One TOML file, no Python training stack.
  image:
    src: /logo.png
    alt: Retrograd
  actions:
    - theme: brand
      text: Quickstart
      link: /getting-started/quickstart
    - theme: alt
      text: Configuration reference
      link: /reference/configuration

features:
  - title: SFT - Supervised fine-tuning
    details: Train against known assistant responses in chat JSONL or plain text. The direct choice when the dataset already contains the behavior the model should reproduce.
    link: /training/sft
    linkText: SFT guide
  - title: PPO - Proximal policy optimization
    details: Sample one completion per prompt, score it with an external reward command, and update with an exact clipped-surrogate gradient. An optional critic estimates per-token values and lowers variance.
    link: /training/ppo
    linkText: PPO guide
  - title: GRPO - Group-relative policy optimization
    details: Sample a group of completions per prompt and center rewards within the group. Best when rewards rank alternatives without a stable absolute scale. Supports an optional judge and container-backed agentic rollouts.
    link: /training/grpo
    linkText: GRPO guide
  - title: Distillation - on-policy and offline top-k
    details: Train a small student to match a larger teacher's policy. On-policy, the student samples and the teacher scores the same tokens; offline, the teacher's truncated distribution over a fixed corpus is computed once and trained against. The choice when you have a model that already behaves well and no way to score an answer.
    link: /training/distill
    linkText: Distillation guide
---

## Start here

1. [Build Retrograd and run a small SFT job](/getting-started/quickstart).
2. [Prepare text or chat JSONL data](/getting-started/datasets).
3. Choose [SFT](/training/sft), [PPO](/training/ppo), [GRPO](/training/grpo) or
   [distillation](/training/distill) for the learning signal you have.

## What Retrograd changes

The base model stays frozen. Training updates the LoRA tensors and writes them
to the path in `[lora].output`. The training binary reads a GGUF model and a
TOML configuration; it does not require a Python training framework.

The main CLI commands are:

```text
retrograd train CONFIG.toml
retrograd bench CONFIG.toml
retrograd chat CONFIG.toml
retrograd inspect --model MODEL.gguf
retrograd preflight --model MODEL.gguf
```

## Reference and operations

- [Configuration reference](/reference/configuration) - every supported TOML setting.
- [CLI reference](/reference/cli) - commands, overrides, and runtime environment variables.
- [Build variants](/reference/builds) - CPU, CUDA, Vulkan, Metal, and container-enabled builds.
- [Checkpoints and monitoring](/operations/checkpoints) - save, resume, and inspect runs.
- [Profiling](/operations/profiling) - measure GRPO and agentic-GRPO phases.
- [Engineering documentation](/engineering/) - implementation contracts, RIR notes, backend status, and test lanes.
