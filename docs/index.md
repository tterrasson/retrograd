---
layout: home

hero:
  name: Retrograd
  text: Fine-tune GGUF models directly
  tagline: Train a LoRA adapter or the model's own weights with SFT, PPO, GRPO, agentic GRPO or distillation. One TOML file, one binary, no Python stack.
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
  - title: SFT
    details: Learn from example answers in chat JSONL or plain text.
    link: /training/sft
  - title: PPO and GRPO
    details: Learn from a reward program that scores the model's own answers.
    link: /training/grpo
  - title: Agentic GRPO
    details: Multi-turn training with tool calls, MCP servers and sandboxed environments.
    link: /training/agent
  - title: Distillation
    details: Make a small model behave like a larger one, with no reward needed.
    link: /training/distill
---

## How it works

Retrograd loads a GGUF model as is, with no conversion, and trains it with
`ggml` on CPU, Metal, Vulkan or CUDA. By default it trains a LoRA adapter,
written as a standard GGUF that llama.cpp loads with `--lora`. It can also
train some or all of the model's weights.

```text
retrograd inspect --model base.gguf    # check the model and pick LoRA targets
retrograd train run.toml               # train
retrograd bench run.toml --adapter adapter.gguf   # compare base and adapter
retrograd chat run.toml --compare      # try it interactively
```

Start with the [quickstart](/getting-started/quickstart).
