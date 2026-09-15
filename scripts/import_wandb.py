#!/usr/bin/env python3
"""Import a local Retrograd W&B staging export.

The training binary deliberately has no Python, network, or W&B dependency.
Install the official SDK separately (`pip install wandb`) and run:

    WANDB_API_KEY=... scripts/import_wandb.py runs/my-run/wandb --project my-project
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity")
    args = parser.parse_args()

    try:
        import wandb
    except ImportError as error:
        raise SystemExit("W&B SDK missing; install it with: pip install wandb") from error

    manifest_path = args.export_dir / "run.json"
    metrics_path = args.export_dir / "metrics.jsonl"
    if not manifest_path.is_file() or not metrics_path.is_file():
        raise SystemExit("expected run.json and metrics.jsonl in export_dir")

    metadata = json.loads(manifest_path.read_text())
    with wandb.init(project=args.project, entity=args.entity, config=metadata) as run:
        run.define_metric("*", step_metric="global_step")
        for line in metrics_path.read_text().splitlines():
            event = json.loads(line)
            if event["event"] != "step":
                continue
            values = {value["name"]: value["value"] for value in event["values"]}
            values["global_step"] = event["global_step"]
            values["epoch"] = event["epoch"]
            run.log(values, step=event["global_step"])


if __name__ == "__main__":
    main()
