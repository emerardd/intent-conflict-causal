"""Independent arithmetic and identity audit for Stage 5 raw artifacts.

This script deliberately does not import intent_conflict.stage5 or its summarizer.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/stage5/confirmation-v1"


def load(name: str):
    with (OUTPUT / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def interval(values, samples, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return np.quantile(draws, [0.025, 0.975]).tolist()


def close(left, right, tolerance=1e-12):
    return bool(np.allclose(left, right, atol=tolerance, rtol=0, equal_nan=True))


def main():
    config = load("freeze.json")["config"]
    baseline = load("baseline.json")["rows"]
    result = load("interventions.json")
    rows = result["rows"]
    saved = result["summary"]
    by_id = {row["example_id"]: row for row in baseline}
    assert len(baseline) == 48 and len(by_id) == 48
    assert len({row["scenario_id"] for row in baseline}) == 12
    assert len(rows) == 576

    scenario_cells = defaultdict(set)
    for row in baseline:
        scenario_cells[row["scenario_id"]].add((row["authorized"], row["reversed_mapping"]))
    expected_cells = {(a, m) for a in (False, True) for m in (False, True)}
    assert all(cells == expected_cells for cells in scenario_cells.values())

    condition_counts = defaultdict(int)
    clustered = defaultdict(lambda: defaultdict(list))
    for row in rows:
        recipient = by_id[row["example_id"]]
        donor = by_id[row["donor_id"]]
        assert recipient["scenario_id"] == donor["scenario_id"]
        assert recipient["candidate_command"] == donor["candidate_command"]
        assert recipient["reversed_mapping"] == donor["reversed_mapping"]
        assert recipient["authorized"] != donor["authorized"]
        sign = 1.0 if donor["expected_label"] == "B" else -1.0
        effect = sign * (float(row["margin"]) - float(recipient["margin"]))
        key = (row["target_position"], row["source_axis"], row["mode"])
        condition_counts[key] += 1
        clustered[key][recipient["scenario_id"]].append(effect)
    assert len(condition_counts) == 12
    assert set(condition_counts.values()) == {48}

    scenario_means = {
        key: {sid: float(np.mean(values)) for sid, values in groups.items()}
        for key, groups in clustered.items()
    }
    independent_effects = {}
    effect_seed = config["seed"] + 10
    for key, groups in scenario_means.items():
        values = list(groups.values())
        label = "/".join(key)
        independent_effects[label] = {
            "mean": float(np.mean(values)),
            "ci95": interval(values, config["bootstrap_samples"], effect_seed),
        }
        assert close(independent_effects[label]["mean"], saved["effects"][label]["mean"])
        assert close(independent_effects[label]["ci95"], saved["effects"][label]["ci95"])

    def paired(left, right, seed):
        assert set(scenario_means[left]) == set(scenario_means[right])
        differences = [scenario_means[left][sid] - scenario_means[right][sid]
                       for sid in scenario_means[left]]
        return {"mean": float(np.mean(differences)),
                "ci95": interval(differences, config["bootstrap_samples"], seed)}

    independent_primary = {
        "answer_local_direction_specific": paired(
            ("answer", "answer", "parallel"),
            ("answer", "answer", "random_parallel"), effect_seed + 6),
        "answer_full_state_positive_control": paired(
            ("answer", "none", "full"),
            ("answer", "none", "random_full"), effect_seed + 4),
        "position_interaction": paired(
            ("answer", "answer", "parallel"),
            ("pre_mapping", "pre_mapping", "parallel"), effect_seed + 7),
    }
    saved_keys = {
        "answer_local_direction_specific": "answer/answer/parallel-minus-random",
        "answer_full_state_positive_control": "answer/full-minus-random_full",
        "position_interaction": "position-interaction/local-parallel",
    }
    for key, value in independent_primary.items():
        reference = saved["paired_comparisons"][saved_keys[key]]
        assert close(value["mean"], reference["mean"])
        assert close(value["ci95"], reference["ci95"])

    pre_full = independent_effects["pre_mapping/none/full"]
    decisions = {
        "answer_local_direction_specific": independent_primary["answer_local_direction_specific"]["ci95"][0] > 0,
        "answer_full_state_positive_control": independent_primary["answer_full_state_positive_control"]["ci95"][0] > 0,
        "position_interaction_positive": independent_primary["position_interaction"]["ci95"][0] > 0,
        "pre_mapping_full_within_equivalence_bound": (
            pre_full["ci95"][0] > -config["pre_mapping_equivalence_bound"]
            and pre_full["ci95"][1] < config["pre_mapping_equivalence_bound"]
        ),
    }
    assert decisions == saved["predeclared_decisions"]

    mapping_accuracy = {
        str(mapping): float(np.mean([
            row["top1_label"] == row["expected_label"] for row in baseline
            if row["reversed_mapping"] == mapping
        ])) for mapping in (False, True)
    }
    assert mapping_accuracy == result["mapping_accuracy"]
    baseline_errors = [row["example_id"] for row in baseline
                       if row["top1_label"] != row["expected_label"]]
    assert baseline_errors == ["stage5__constellation_card__auth0__map1"]

    flips = []
    toward_donor = []
    for row in rows:
        recipient, donor = by_id[row["example_id"]], by_id[row["donor_id"]]
        if row["top1_id"] != recipient["top1_id"]:
            flips.append(row)
            if row["top1_label"] == donor["expected_label"]:
                toward_donor.append(row)
    assert len(flips) == 7 and not toward_donor
    assert {row["example_id"] for row in flips} == set(baseline_errors)
    assert result["generation_agreement"]
    assert len(result["generation_audit"]) == 24
    assert len(result["null_checks"]) == 2
    max_norm_error = max(abs(float(row["requested_delta_norm"])
                             - float(row["delivered_delta_norm"])) for row in rows)
    assert max_norm_error < 0.003

    frozen = load("freeze.json")
    frozen_sources_ok = True
    for relative, expected in frozen["source_sha256"].items():
        path = OUTPUT / "source_snapshot" / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        frozen_sources_ok &= actual == expected
    assert frozen_sources_ok

    audit = {
        "status": "pass",
        "independent_of_stage5_summary_code": True,
        "identity_checks": {
            "baseline_rows": len(baseline),
            "scenario_clusters": len(scenario_cells),
            "intervention_rows": len(rows),
            "conditions": len(condition_counts),
            "rows_per_condition": sorted(set(condition_counts.values())),
        },
        "mapping_accuracy": mapping_accuracy,
        "baseline_errors": baseline_errors,
        "independent_primary": independent_primary,
        "independent_pre_mapping_full": pre_full,
        "independent_decisions": decisions,
        "categorical": {
            "all_top1_flips": len(flips),
            "flips_toward_donor": len(toward_donor),
            "all_flips_from_single_tied_baseline": True,
        },
        "delivery": {
            "generation_checks": len(result["generation_audit"]),
            "generation_agreement": result["generation_agreement"],
            "null_checks": len(result["null_checks"]),
            "max_patch_norm_error": max_norm_error,
            "frozen_source_hashes_ok": frozen_sources_ok,
        },
    }
    with (OUTPUT / "independent_audit.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
