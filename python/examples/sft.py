from pathlib import Path

from retrograd import LoraConfig, Trainer, TrainingConfig

model_path = Path("/path/to/model.gguf")
train_path = Path("/path/to/train.jsonl")
output_path = Path("adapter.gguf")

with Trainer(
    model_path,
    training=TrainingConfig(epochs=3, learning_rate=1.0e-4, device="auto"),
    lora=LoraConfig(rank=8, alpha=16.0, targets=("q", "v")),
) as trainer:
    print(trainer.capability_report())
    print(trainer.preflight())

    dataset = trainer.prepare_dataset(train_path)
    metrics = trainer.fit(dataset, callback=lambda step: print(step))
    trainer.save_adapter(output_path)

print(metrics)
