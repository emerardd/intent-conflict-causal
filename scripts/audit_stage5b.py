"""Independent raw-artifact audit for Stage 5b; does not import its analysis code."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/stage5b/confirmation-v1"


def load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def interval(values, samples, seed):
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return np.quantile(draws, [0.025, 0.975]).tolist()


def close(left, right, tolerance=1e-12):
    return bool(np.allclose(left, right, atol=tolerance, rtol=0, equal_nan=True))


def independently_derive_scale(config, axes):
    reference = ROOT / config["stage3_reference"]
    rows = load(reference / "baseline.json")["rows"]
    archive = np.load(reference / "activations.npz", allow_pickle=False)
    assert archive["example_ids"].tolist() == [row["example_id"] for row in rows]
    layer_index = archive["layers"].tolist().index(config["layer"])
    positions = config["positions"]
    position_indices = [archive["positions"].tolist().index(position) for position in positions]
    activations = archive["activations"][:, position_indices, layer_index].astype(float)
    fields = ("scenario_id", "grammar", "order", "reversed_mapping")
    lookup = {tuple(row[field] for field in fields) + (row["authorized"],): index
              for index, row in enumerate(rows)}
    train = [index for index, row in enumerate(rows) if row["split"] == "train"]
    medians = {}
    for position_index, position in enumerate(positions):
        direction = axes[position] / np.linalg.norm(axes[position])
        values = []
        for index in train:
            row = rows[index]
            donor_index = lookup[tuple(row[field] for field in fields) + (not row["authorized"],)]
            values.append(abs(float((activations[donor_index, position_index]
                                     - activations[index, position_index]) @ direction)))
        medians[position] = float(np.median(values))
    return {"position_medians": medians, "common_scale": float(min(medians.values())),
            "n_train_rows": len(train)}


def main():
    frozen = load(OUTPUT / "freeze.json")
    config = frozen["config"]
    baselines = load(OUTPUT / "baseline.json")["rows"]
    result = load(OUTPUT / "interventions.json")
    rows = result["rows"]
    saved = result["summary"]
    baseline = {row["example_id"]: row for row in baselines}

    assert len(baselines) == 48 and len(baseline) == 48
    assert len({row["scenario_id"] for row in baselines}) == 12
    assert len(rows) == 1344
    cells = defaultdict(set)
    for row in baselines:
        cells[row["scenario_id"]].add((row["authorized"], row["reversed_mapping"]))
    assert all(value == {(False, False), (False, True), (True, False), (True, True)}
               for value in cells.values())

    axes_archive = np.load(OUTPUT / "frozen_axes.npz", allow_pickle=False)
    axes = {position: axes_archive[position].astype(float) for position in config["source_axes"]}
    independent_scale = independently_derive_scale(config, axes)
    assert close(independent_scale["common_scale"], frozen["derived_scale"]["common_scale"])
    assert independent_scale["position_medians"] == frozen["derived_scale"]["position_medians"]
    assert independent_scale["n_train_rows"] == frozen["derived_scale"]["n_train_rows"]

    real_matrix = np.stack([axes[position] / np.linalg.norm(axes[position])
                            for position in config["source_axes"]])
    axis_checks = {}
    for target in config["positions"]:
        bank = np.stack([axes_archive[f"{target}_random_{index:02d}"].astype(float)
                         for index in range(config["n_random_axes"])])
        check = {
            "max_norm_error": float(np.max(np.abs(np.linalg.norm(bank, axis=1) - 1))),
            "max_mutual_dot": float(np.max(np.abs(bank @ bank.T - np.eye(len(bank))))),
            "max_real_axis_dot": float(np.max(np.abs(bank @ real_matrix.T))),
        }
        assert check["max_norm_error"] < 1e-12
        assert check["max_mutual_dot"] < 1e-12
        assert check["max_real_axis_dot"] < 1e-12
        axis_checks[target] = check

    grouped = defaultdict(lambda: defaultdict(list))
    condition_counts = defaultdict(int)
    for row in rows:
        recipient = baseline[row["example_id"]]
        donor = baseline[row["donor_id"]]
        assert recipient["scenario_id"] == donor["scenario_id"]
        assert recipient["candidate_command"] == donor["candidate_command"]
        assert recipient["reversed_mapping"] == donor["reversed_mapping"]
        assert recipient["authorized"] != donor["authorized"]
        sign = 1.0 if donor["expected_label"] == "B" else -1.0
        effect = sign * (float(row["margin"]) - float(recipient["margin"]))
        key = (row["target_position"], row["source_axis"], row["mode"])
        grouped[key][recipient["scenario_id"]].append(effect)
        condition_counts[key] += 1
    assert len(condition_counts) == 28
    assert set(condition_counts.values()) == {48}
    scenario = {key: {sid: float(np.mean(values)) for sid, values in groups.items()}
                for key, groups in grouped.items()}

    random_sources = [f"random_{index:02d}" for index in range(config["n_random_axes"])]
    random_means = {}
    for target in config["positions"]:
        random_means[target] = {
            sid: float(np.mean([scenario[target, source, "equal_norm"][sid]
                                for source in random_sources]))
            for sid in scenario[target, random_sources[0], "equal_norm"]
        }

    def paired(left, right, seed):
        left_values = scenario[left] if isinstance(left, tuple) else random_means[left]
        right_values = scenario[right] if isinstance(right, tuple) else random_means[right]
        assert set(left_values) == set(right_values)
        values = [left_values[sid] - right_values[sid] for sid in left_values]
        return {"mean": float(np.mean(values)),
                "ci95": interval(values, config["bootstrap_samples"], seed),
                "scenario_differences": dict(zip(left_values, values))}

    base_seed = config["seed"] + 10
    primary = {
        "answer/answer-minus-random_mean": paired(
            ("answer", "answer", "equal_norm"), "answer", base_seed + 1),
        "answer/answer-minus-pre_mapping": paired(
            ("answer", "answer", "equal_norm"),
            ("answer", "pre_mapping", "equal_norm"), base_seed + 2),
    }
    answer_identity = primary["answer/answer-minus-pre_mapping"]["scenario_differences"]
    pre_identity_raw = paired(
        ("pre_mapping", "answer", "equal_norm"),
        ("pre_mapping", "pre_mapping", "equal_norm"), base_seed + 8)
    interaction_values = [answer_identity[sid] - pre_identity_raw["scenario_differences"][sid]
                          for sid in answer_identity]
    primary["position-by-direction-interaction"] = {
        "mean": float(np.mean(interaction_values)),
        "ci95": interval(interaction_values, config["bootstrap_samples"], base_seed + 3),
        "scenario_differences": dict(zip(answer_identity, interaction_values)),
    }
    for key, value in primary.items():
        reference = saved["paired_comparisons"][key]
        assert close(value["mean"], reference["mean"])
        assert close(value["ci95"], reference["ci95"])
        assert close(list(value["scenario_differences"].values()),
                     list(reference["scenario_differences"].values()))

    decisions = {
        "answer_direction_over_random": primary["answer/answer-minus-random_mean"]["ci95"][0] > 0,
        "answer_direction_over_equal_norm_cross": primary["answer/answer-minus-pre_mapping"]["ci95"][0] > 0,
        "position_by_direction_interaction": primary["position-by-direction-interaction"]["ci95"][0] > 0,
    }
    assert decisions == saved["predeclared_checks"]

    mapping_accuracy = {
        str(mapping): float(np.mean([
            row["top1_label"] == row["expected_label"] for row in baselines
            if row["reversed_mapping"] == mapping
        ])) for mapping in (False, True)
    }
    assert mapping_accuracy == {"False": 1.0, "True": 1.0}
    assert mapping_accuracy == result["mapping_accuracy"]
    top1_flips = [row for row in rows
                  if row["top1_id"] != baseline[row["example_id"]]["top1_id"]]
    donor_flips = [row for row in top1_flips
                   if row["top1_label"] == baseline[row["donor_id"]]["expected_label"]]
    assert not top1_flips and not donor_flips
    assert not [row for row in rows if row["top1_label"] == "OTHER"]

    equal_rows = [row for row in rows if row["mode"] == "equal_norm"]
    assert len(equal_rows) == 1152
    requested = [float(row["requested_delta_norm"]) for row in equal_rows]
    delivered = [float(row["delivered_delta_norm"]) for row in equal_rows]
    scale_value = frozen["derived_scale"]["common_scale"]
    assert max(abs(value - scale_value) for value in requested) < 1e-12
    max_delivered_error = max(abs(value - scale_value) for value in delivered)
    assert max_delivered_error < 0.003
    assert result["generation_agreement"] and len(result["generation_audit"]) == 32
    assert len(result["null_checks"]) == 2

    source_hashes_ok = True
    for relative, expected in frozen["source_sha256"].items():
        actual = hashlib.sha256((OUTPUT / "source_snapshot" / relative).read_bytes()).hexdigest()
        source_hashes_ok &= actual == expected
    assert source_hashes_ok

    random_axis_effects = {
        target: {source: float(np.mean(list(scenario[target, source, "equal_norm"].values())))
                 for source in random_sources}
        for target in config["positions"]
    }
    audit = {
        "status": "pass",
        "independent_of_stage5b_analysis_code": True,
        "identity": {
            "baseline_rows": len(baselines),
            "scenario_clusters": len(cells),
            "intervention_rows": len(rows),
            "conditions": len(condition_counts),
            "rows_per_condition": sorted(set(condition_counts.values())),
        },
        "independent_scale": independent_scale,
        "axis_checks": axis_checks,
        "mapping_accuracy": mapping_accuracy,
        "independent_primary": primary,
        "independent_decisions": decisions,
        "random_axis_effects": random_axis_effects,
        "categorical": {
            "top1_flips": len(top1_flips),
            "donor_directed_flips": len(donor_flips),
            "invalid_first_tokens": 0,
        },
        "delivery": {
            "equal_norm_rows": len(equal_rows),
            "requested_norm_min": min(requested),
            "requested_norm_max": max(requested),
            "delivered_norm_min": min(delivered),
            "delivered_norm_max": max(delivered),
            "max_delivered_norm_error": max_delivered_error,
            "generation_checks": len(result["generation_audit"]),
            "generation_agreement": result["generation_agreement"],
            "null_checks": len(result["null_checks"]),
            "frozen_source_hashes_ok": source_hashes_ok,
        },
    }
    with (OUTPUT / "independent_audit.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
