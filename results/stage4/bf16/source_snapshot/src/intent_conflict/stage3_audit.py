"""Independent, stdlib-only audit of the frozen pilot's raw behavioral evidence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def expected_answer(row: dict) -> tuple[bool, str]:
    # Arithmetic parity, independently expressed from the generator's booleans.
    block = (int(not row["authorized"]) + int(row["reversed_policy"])) % 2
    letter = (block + int(row["reversed_mapping"])) % 2
    return bool(block), "AB"[letter]


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit_screen(run: Path, include_activations: bool = False) -> dict:
    def load(name):
        return json.loads((run / name).read_text(encoding="utf-8"))
    baseline, freeze, provenance, saved = (load(n) for n in ("baseline.json", "freeze.json", "provenance.json", "screen_summary.json"))
    prompts = load("prompts.json")
    rows = baseline["rows"]
    remap_only = freeze["config"]["experiment"] == "stage3_remapping_v2_after_failed_factorial_pilot"
    expected_count = 16 if remap_only else 32
    assert len(rows) == len(prompts) == len(freeze["dataset_rows"]) == expected_count
    assert len({r["example_id"] for r in rows}) == expected_count
    assert file_sha(run / "prompts.json") == provenance["prompts_sha256"]
    for name, digest in freeze["source_sha256"].items():
        assert file_sha(run / "source_snapshot" / name) == digest, name
    assert baseline["config_sha256"] == freeze["config_sha256"] == saved["config_sha256"]
    groups, cells = defaultdict(list), defaultdict(list)
    total_correct, b_count, noncanonical_b = 0, 0, 0
    truth_rows = defaultdict(list)
    for row, frozen, prompt in zip(rows, freeze["dataset_rows"], prompts, strict=True):
        assert all(row[k] == v for k, v in frozen.items())
        assert row["example_id"] == prompt["example_id"]
        block, label = expected_answer(row)
        assert row["assigned_block"] == block and row["expected_label"] == label
        assert row["top1_id"] == provenance[f"{row['top1_label']}_id"]
        assert row["generated_ids"][0] == row["top1_id"]
        assert row["generated_text"].strip() == row["top1_label"]
        assert row["ab_label"] == ("B" if row["logp_B"] >= row["logp_A"] else "A")
        assert abs(row["margin"]-(row["logp_B"]-row["logp_A"])) < 1e-12
        assert all(row["top5"][i]["logp"] >= row["top5"][i+1]["logp"] for i in range(4))
        cells[row["reversed_policy"], row["reversed_mapping"]].append(row)
        groups[row["scenario_id"], row["order"]].append(row)
        correct = label == row["top1_label"]
        total_correct += correct
        b_count += row["top1_label"] == "B"
        noncanonical_b += (row["reversed_policy"] or row["reversed_mapping"]) and row["top1_label"] == "B"
        truth_rows[row["authorized"], row["reversed_policy"], row["reversed_mapping"]].append(row)
    for group in groups.values():
        assert len({(r["authorized"], r["reversed_policy"], r["reversed_mapping"]) for r in group}) == (4 if remap_only else 8)
        assert len({r["candidate_command"] for r in group}) == 1
    cell_report = {}
    for (policy, mapping), group in cells.items():
        correct = sum(r["top1_label"] == expected_answer(r)[1] for r in group)
        assert len(group) == 8
        key = f"policy{int(policy)}_map{int(mapping)}"
        assert saved["behavior"]["screen/seen"]["cells"][key]["global_top1_accuracy"] == correct/8
        cell_report[key] = {"correct": correct, "n": len(group), "observed_symbols": dict(Counter(r["top1_label"] for r in group))}
    passed = (min(c["correct"]/c["n"] for c in cell_report.values()) >= freeze["config"]["screen_min_cell_top1_accuracy"] and
              total_correct/expected_count >= freeze["config"]["generation_min_exact_accuracy"])
    assert saved["passed"] == passed
    table = [{"authorized": k[0], "reversed_policy": k[1], "reversed_mapping": k[2],
              "expected": expected_answer(v[0])[1], "observed_counts": dict(Counter(r["top1_label"] for r in v)), "n": len(v)}
             for k, v in truth_rows.items()]
    checks = {"source_snapshot_hashes": True, "prompt_hash": True, "frozen_data_identity": True,
              "independent_truth_table": True, "full_factorial_balance": True,
              "generation_argmax_agreement": True, "saved_statistics_recomputed": True}
    if include_activations:
        import numpy as np
        archive = np.load(run / "activations.npz", allow_pickle=False)
        assert archive["activations"].shape == (expected_count, 3, 7, 2560)
        assert archive["example_ids"].tolist() == [r["example_id"] for r in rows]
        assert np.isfinite(archive["activations"]).all()
        assert archive["positions"].tolist() == freeze["config"]["positions"]
        assert archive["layers"].tolist() == freeze["config"]["layers"]
        checks["activation_shape_identity_finiteness"] = True
    return {"checks": checks, "all_checks_passed": all(checks.values()), "screen_passed": passed,
        "total_correct": total_correct, "total": expected_count, "total_B": b_count,
        "noncanonical_B_count": noncanonical_b,
        "noncanonical_n": sum(r["reversed_policy"] or r["reversed_mapping"] for r in rows),
        "cells": cell_report, "truth_table": table,
        "formal_directory_exists": (run.parent / "formal-v1").exists(),
        "claim_boundary": "Independent artifact audit, not a rerun of model inference or proof of a mechanism."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("results/stage3/screen-v1"))
    parser.add_argument("--include-activations", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit_screen(args.run, args.include_activations), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
