from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import reanalyze_experiment, run_experiment, run_robustness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Intent-conflict causal experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    reanalyze = subparsers.add_parser("reanalyze")
    reanalyze.add_argument("--config", type=Path, required=True)
    robustness = subparsers.add_parser("robustness")
    robustness.add_argument("--config", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.command == "run":
        result = run_experiment(config, args.config)
    elif args.command == "robustness":
        result = run_robustness(config, args.config)
    else:
        result = reanalyze_experiment(config, args.config)
    if args.command in {"run", "reanalyze"}:
        summary = {
            "output_path": config["output_path"],
            "behavior_gate": result["behavior_gate"],
            "selected_layer": (
                result["representation"]["selected_layer"]
                if result.get("representation")
                else None
            ),
            "decision": result["decision"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
