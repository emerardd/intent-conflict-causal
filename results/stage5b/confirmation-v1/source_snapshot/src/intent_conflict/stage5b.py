"""Stage 5b equal-norm position by direction identity confirmation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import time

import numpy as np

from .stage3 import dump, manifest, now, read, sha
from .stage3_data import tokenize_trial
from .stage3_remap import render_remap
from .stage5b_data import build_stage5b


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Cannot normalize zero vector")
    return vector / norm


def _pair_indices(rows: list[dict], fields: tuple[str, ...]) -> list[int]:
    lookup: dict[tuple, int] = {}
    for index, row in enumerate(rows):
        key = tuple(row[field] for field in fields) + (row["authorized"],)
        if key in lookup:
            raise ValueError(f"Duplicate pair key: {key}")
        lookup[key] = index
    result: list[int] = []
    for row in rows:
        key = tuple(row[field] for field in fields) + (not row["authorized"],)
        if key not in lookup:
            raise ValueError(f"Missing paired row: {key}")
        result.append(lookup[key])
    return result


def derive_common_scale(
    rows: list[dict],
    activations: np.ndarray,
    positions: list[str],
    axes: dict[str, np.ndarray],
) -> dict:
    if activations.shape[:2] != (len(rows), len(positions)):
        raise ValueError("Reference activation identity or position mismatch")
    pairs = _pair_indices(rows, ("scenario_id", "grammar", "order", "reversed_mapping"))
    train_indices = [index for index, row in enumerate(rows) if row["split"] == "train"]
    medians: dict[str, float] = {}
    ranges: dict[str, dict] = {}
    for position_index, position in enumerate(positions):
        direction = unit(axes[position])
        values = [
            abs(float((activations[pairs[index], position_index].astype(float)
                       - activations[index, position_index].astype(float)) @ direction))
            for index in train_indices
        ]
        if not values or min(values) <= 0:
            raise ValueError("Training reference has a nonpositive paired projection scale")
        medians[position] = float(np.median(values))
        ranges[position] = {
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "mean": float(np.mean(values)),
            "n": len(values),
        }
    scale = float(min(medians.values()))
    return {
        "method": "minimum of positionwise median absolute local donor projection on Stage 3 train",
        "position_medians": medians,
        "position_ranges": ranges,
        "common_scale": scale,
        "n_train_rows": len(train_indices),
    }


def random_axis_banks(
    axes: dict[str, np.ndarray], positions: list[str], count: int, seed: int
) -> dict[str, np.ndarray]:
    real = np.stack([unit(axes[position]) for position in positions], axis=1)
    real_basis, _ = np.linalg.qr(real)
    rng = np.random.default_rng(seed)
    output: dict[str, np.ndarray] = {}
    for position in positions:
        vectors: list[np.ndarray] = []
        for _ in range(count):
            candidate = rng.normal(size=real.shape[0])
            candidate = candidate - real_basis @ (real_basis.T @ candidate)
            for previous in vectors:
                candidate = candidate - previous * float(previous @ candidate)
            vectors.append(unit(candidate))
        output[position] = np.stack(vectors)
    return output


def equal_norm_vector(
    original: np.ndarray,
    direction: np.ndarray,
    scale: float,
    donor_unauthorized: bool,
) -> np.ndarray:
    sign = 1.0 if donor_unauthorized else -1.0
    return original.astype(float) + sign * scale * unit(direction)


def pair_confirmation_rows(rows: list[dict]) -> list[int]:
    pairs = _pair_indices(rows, ("scenario_id", "reversed_mapping"))
    for index, donor_index in enumerate(pairs):
        if rows[index]["candidate_command"] != rows[donor_index]["candidate_command"]:
            raise ValueError("Candidate command mismatch in Stage 5b pair")
    return pairs


def _interval(values: list[float], samples: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=float)
    if len(array) < 4:
        raise ValueError("At least four scenario clusters are required")
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return np.quantile(draws, [0.025, 0.975]).tolist()


def summarize(
    rows: list[dict], baselines: list[dict], bootstrap_samples: int, seed: int
) -> dict:
    baseline = {row["example_id"]: row for row in baselines}
    grouped: dict[tuple[str, str, str], list[tuple[dict, float, float]]] = defaultdict(list)
    for row in rows:
        recipient = baseline[row["example_id"]]
        donor = baseline[row["donor_id"]]
        sign = 1.0 if donor["expected_label"] == "B" else -1.0
        effect = sign * (float(row["margin"]) - float(recipient["margin"]))
        gap = sign * (float(donor["margin"]) - float(recipient["margin"]))
        grouped[row["target_position"], row["source_axis"], row["mode"]].append((row, effect, gap))

    effects: dict[str, dict] = {}
    scenario_means: dict[tuple[str, str, str], dict[str, float]] = {}
    for key, group in grouped.items():
        scenarios: dict[str, list[float]] = defaultdict(list)
        for row, effect, _ in group:
            scenarios[baseline[row["example_id"]]["scenario_id"]].append(effect)
        means = {scenario: float(np.mean(values)) for scenario, values in scenarios.items()}
        scenario_means[key] = means
        label = "/".join(key)
        values = list(means.values())
        effects[label] = {
            "mean": float(np.mean(values)),
            "ci95": _interval(values, bootstrap_samples, seed),
            "n_scenarios": len(values),
            "n_forwards": len(group),
            "scenario_effects": means,
            "mapping_means": {
                str(mapping): float(np.mean([
                    effect for row, effect, _ in group
                    if baseline[row["example_id"]]["reversed_mapping"] == mapping
                ])) for mapping in (False, True)
            },
            "global_top1_flips": sum(
                row["top1_id"] != baseline[row["example_id"]]["top1_id"]
                for row, _, _ in group
            ),
            "flips_toward_donor": sum(
                row["top1_id"] != baseline[row["example_id"]]["top1_id"]
                and row["top1_label"] == baseline[row["donor_id"]]["expected_label"]
                for row, _, _ in group
            ),
            "invalid_first_tokens": sum(row["top1_label"] == "OTHER" for row, _, _ in group),
            "mean_gap_recovery": float(np.mean([
                effect / gap for _, effect, gap in group if gap > 1e-6
            ])) if any(gap > 1e-6 for _, _, gap in group) else None,
        }

    random_sources = sorted({
        source for _, source, mode in scenario_means
        if mode == "equal_norm" and source.startswith("random_")
    })
    if not random_sources:
        raise ValueError("No random equal-norm directions found")
    for target in ("pre_mapping", "answer"):
        scenario_ids = list(scenario_means[target, random_sources[0], "equal_norm"])
        means = {
            scenario: float(np.mean([
                scenario_means[target, source, "equal_norm"][scenario]
                for source in random_sources
            ])) for scenario in scenario_ids
        }
        key = (target, "random_mean", "equal_norm")
        scenario_means[key] = means
        effects[f"{target}/random_mean/equal_norm"] = {
            "mean": float(np.mean(list(means.values()))),
            "ci95": _interval(list(means.values()), bootstrap_samples, seed),
            "n_scenarios": len(means),
            "n_axes": len(random_sources),
            "scenario_effects": means,
        }

    def compare(left: tuple[str, str, str], right: tuple[str, str, str], offset: int) -> dict:
        if set(scenario_means[left]) != set(scenario_means[right]):
            raise ValueError("Paired comparison scenario mismatch")
        differences = [
            scenario_means[left][scenario] - scenario_means[right][scenario]
            for scenario in scenario_means[left]
        ]
        return {
            "mean": float(np.mean(differences)),
            "ci95": _interval(differences, bootstrap_samples, seed + offset),
            "scenario_differences": dict(zip(scenario_means[left], differences)),
        }

    comparisons = {
        "answer/answer-minus-random_mean": compare(
            ("answer", "answer", "equal_norm"),
            ("answer", "random_mean", "equal_norm"), 1),
        "answer/answer-minus-pre_mapping": compare(
            ("answer", "answer", "equal_norm"),
            ("answer", "pre_mapping", "equal_norm"), 2),
        "answer-axis/answer-minus-pre-site": compare(
            ("answer", "answer", "equal_norm"),
            ("pre_mapping", "answer", "equal_norm"), 4),
        "pre-axis/answer-minus-pre-site": compare(
            ("answer", "pre_mapping", "equal_norm"),
            ("pre_mapping", "pre_mapping", "equal_norm"), 5),
        "local-axis/answer-minus-pre-site": compare(
            ("answer", "answer", "equal_norm"),
            ("pre_mapping", "pre_mapping", "equal_norm"), 6),
        "full/answer-minus-pre-site": compare(
            ("answer", "none", "full"),
            ("pre_mapping", "none", "full"), 7),
    }
    answer_identity = comparisons["answer/answer-minus-pre_mapping"]
    pre_identity = compare(
        ("pre_mapping", "answer", "equal_norm"),
        ("pre_mapping", "pre_mapping", "equal_norm"), 8)
    interaction_differences = [
        answer_identity["scenario_differences"][scenario]
        - pre_identity["scenario_differences"][scenario]
        for scenario in answer_identity["scenario_differences"]
    ]
    comparisons["position-by-direction-interaction"] = {
        "mean": float(np.mean(interaction_differences)),
        "ci95": _interval(interaction_differences, bootstrap_samples, seed + 3),
        "scenario_differences": dict(zip(answer_identity["scenario_differences"], interaction_differences)),
    }
    comparisons["pre_mapping/answer-minus-pre_mapping"] = pre_identity

    checks = {
        "answer_direction_over_random": comparisons["answer/answer-minus-random_mean"]["ci95"][0] > 0,
        "answer_direction_over_equal_norm_cross": comparisons["answer/answer-minus-pre_mapping"]["ci95"][0] > 0,
        "position_by_direction_interaction": comparisons["position-by-direction-interaction"]["ci95"][0] > 0,
    }
    return {
        "effects": effects,
        "paired_comparisons": comparisons,
        "predeclared_checks": checks,
        "all_three_checks_pass": all(checks.values()),
        "estimand": "Donor-answer-aligned B/A log-margin change, averaged within scenario.",
        "scope": "Equal-norm position by frozen-direction identity in a controlled forced-choice task.",
    }


def _copy_sources(root: Path, output: Path, config_path: Path) -> dict[str, str]:
    sources = [
        root / "src/intent_conflict/stage5b.py",
        root / "src/intent_conflict/stage5b_data.py",
        root / "src/intent_conflict/stage3.py",
        root / "src/intent_conflict/stage3_model.py",
        root / "src/intent_conflict/stage3_data.py",
        root / "src/intent_conflict/stage3_remap.py",
        root / "src/intent_conflict/stage4.py",
        root / "src/intent_conflict/model.py",
        root / "src/intent_conflict/tokenization.py",
        root / "tests/test_stage5b.py",
        root / "docs/stage5b-preregistration.md",
        config_path,
    ]
    hashes = {path.relative_to(root).as_posix(): sha(path) for path in sources}
    for path in sources:
        destination = output / "source_snapshot" / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    return hashes


def run() -> None:
    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs/stage5b_equal_norm.json"
    config = read(config_path)
    output = root / config["output"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)

    reference = root / config["stage3_reference"]
    reference_rows = read(reference / "baseline.json")["rows"]
    reference_archive = np.load(reference / "activations.npz", allow_pickle=False)
    if reference_archive["example_ids"].tolist() != [row["example_id"] for row in reference_rows]:
        raise ValueError("Stage 3 reference identity mismatch")
    layer_index = reference_archive["layers"].tolist().index(config["layer"])
    position_indices = [reference_archive["positions"].tolist().index(position)
                        for position in config["positions"]]
    reference_x = reference_archive["activations"][:, position_indices, layer_index]

    axes_path = root / config["axes_reference"]
    axes_archive = np.load(axes_path, allow_pickle=False)
    axes = {position: unit(axes_archive[position]) for position in config["source_axes"]}
    scale = derive_common_scale(reference_rows, reference_x, config["positions"], axes)
    banks = random_axis_banks(axes, config["positions"], config["n_random_axes"], config["seed"] + 1)
    np.savez_compressed(
        output / "frozen_axes.npz",
        **axes,
        **{f"{position}_random_{index:02d}": vector
           for position, bank in banks.items() for index, vector in enumerate(bank)},
    )
    source_hashes = _copy_sources(root, output, config_path)

    import torch
    from huggingface_hub import snapshot_download
    from .model import LoadedModel
    from .stage3_model import evaluate, generate
    from .stage4 import patched_evaluate

    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    snapshot = snapshot_download(config["model_name"], revision=config["revision"], local_files_only=True)
    started = time.monotonic()
    print(f"Loading cached {config['precision']} model from pinned snapshot...", flush=True)
    loaded = LoadedModel.load(
        model_name=snapshot,
        model_loader="multimodal_lm",
        device="cuda",
        dtype="bfloat16",
        local_files_only=True,
        allow_fallback_chat_template=False,
        quantization="4bit_nf4",
    )

    trials = build_stage5b(config["seed"])
    if len({trial.scenario_id for trial in trials}) != config["n_scenarios"]:
        raise ValueError("Stage 5b scenario count mismatch")
    prompts: list[dict] = []
    for trial in trials:
        ids, positions = tokenize_trial(loaded.tokenizer, trial, render_remap(trial))
        prompts.append({"example_id": trial.example_id, "ids": ids, "positions": positions})
    trial_rows = [trial.row() for trial in trials]
    pairs = pair_confirmation_rows(trial_rows)
    for index, donor_index in enumerate(pairs):
        if len(prompts[index]["ids"]) != len(prompts[donor_index]["ids"]):
            raise ValueError("Stage 5b donor token length mismatch")
        for position in config["positions"]:
            if prompts[index]["positions"][position] != prompts[donor_index]["positions"][position]:
                raise ValueError("Stage 5b donor position mismatch")

    dump(output / "prompts.json", prompts)
    dump(output / "freeze.json", {
        "created_at_utc": now(),
        "config": config,
        "python": platform.python_version(),
        "source_sha256": source_hashes,
        "reference_sha256": {
            "stage3_baseline": sha(reference / "baseline.json"),
            "stage3_activations": sha(reference / "activations.npz"),
            "stage4_axes": sha(axes_path),
        },
        "derived_scale": scale,
        "random_axis_checks": {
            position: {
                "max_norm_error": float(np.max(np.abs(np.linalg.norm(bank, axis=1) - 1))),
                "max_mutual_dot": float(np.max(np.abs(bank @ bank.T - np.eye(len(bank))))),
                "max_real_axis_dot": float(np.max(np.abs(bank @ np.stack(list(axes.values())).T))),
            } for position, bank in banks.items()
        },
        "example_ids": [trial.example_id for trial in trials],
        "donor_indices": pairs,
        "note": "Written before every Stage 5b model forward; model load and tokenization reveal no response.",
    })
    (output / "environment.txt").write_text(
        "\n".join(sorted(f"{package.metadata['Name']}=={package.version}"
                         for package in importlib.metadata.distributions())) + "\n",
        encoding="utf-8",
    )
    dump(output / "provenance.json", {
        "created_at_utc": now(),
        "snapshot": snapshot,
        "revision": config["revision"],
        "model_class": type(loaded.model).__name__,
        "A_id": loaded.execute_token_id,
        "B_id": loaded.block_token_id,
        "parameter_dtypes": sorted({str(parameter.dtype) for parameter in loaded.model.parameters()}),
        "device_map": {str(key): str(value) for key, value in getattr(loaded.model, "hf_device_map", {}).items()},
    })

    def deadline() -> None:
        if time.monotonic() - started > config["runtime_limit_seconds"]:
            raise TimeoutError("Frozen Stage 5b runtime bound exceeded; partial journals retained")

    baselines: list[dict] = []
    hidden: list[np.ndarray] = []
    with (output / "baseline-progress.jsonl").open("x", encoding="utf-8") as journal:
        for index, (trial, prompt) in enumerate(zip(trials, prompts)):
            deadline()
            positions = {position: prompt["positions"][position] for position in config["positions"]}
            measured, activations = evaluate(loaded, prompt["ids"], positions, [config["layer"]])
            row = {**trial.row(), **measured}
            row["observed_block"] = ((row["top1_label"] == "B") != row["reversed_mapping"]) \
                if row["top1_label"] in {"A", "B"} else None
            baselines.append(row)
            hidden.append(activations[:, 0])
            journal.write(json.dumps(row) + "\n")
            journal.flush()
            if (index + 1) % 12 == 0:
                print(f"baseline {index + 1}/{len(trials)}, elapsed={time.monotonic()-started:.1f}s", flush=True)
    x = np.stack(hidden)
    dump(output / "baseline.json", {"rows": baselines})
    np.savez_compressed(output / "activations.npz", activations=x,
                        example_ids=np.array([row["example_id"] for row in baselines]),
                        positions=np.array(config["positions"]), layer=config["layer"])
    accuracies = {
        str(mapping): float(np.mean([
            row["top1_label"] == row["expected_label"] for row in baselines
            if row["reversed_mapping"] == mapping
        ])) for mapping in (False, True)
    }
    if min(accuracies.values()) < config["baseline_min_mapping_accuracy"]:
        dump(output / "stopped.json", {
            "reason": "baseline mapping accuracy gate failed",
            "mapping_accuracy": accuracies,
            "elapsed_seconds": time.monotonic() - started,
        })
        manifest(output)
        print(json.dumps(read(output / "stopped.json"), indent=2), flush=True)
        return

    interventions: list[dict] = []
    generated: list[dict] = []
    nulls: list[dict] = []
    first_scenario = baselines[0]["scenario_id"]
    common_scale = scale["common_scale"]
    with (output / "interventions-progress.jsonl").open("x", encoding="utf-8") as journal:
        for index, (baseline, prompt) in enumerate(zip(baselines, prompts)):
            donor_index = pairs[index]
            donor_baseline = baselines[donor_index]
            donor_unauthorized = not donor_baseline["authorized"]
            sign = 1.0 if donor_unauthorized else -1.0
            for position_index, target in enumerate(config["positions"]):
                original = x[index, position_index]
                donor = x[donor_index, position_index]
                if index == 0:
                    measured = patched_evaluate(
                        loaded, prompt, target, config["layer"], original, original, axes[target]
                    )
                    if abs(measured["margin"] - baseline["margin"]) > 1e-4 \
                            or measured["top1_id"] != baseline["top1_id"]:
                        raise ValueError("Stage 5b null patch differs from baseline")
                    nulls.append({"target_position": target, **measured})

                delta = donor.astype(float) - original.astype(float)
                full_vectors = {
                    "full": donor.astype(float),
                    "random_full": original.astype(float) + sign * float(np.linalg.norm(delta)) * banks[target][0],
                }
                for mode, vector in full_vectors.items():
                    deadline()
                    direction = axes[target] if mode == "full" else banks[target][0]
                    measured = patched_evaluate(
                        loaded, prompt, target, config["layer"], vector, original, direction
                    )
                    item = {
                        "example_id": baseline["example_id"],
                        "donor_id": donor_baseline["example_id"],
                        "target_position": target,
                        "source_axis": "none",
                        "mode": mode,
                        "natural_delta_norm": float(np.linalg.norm(delta)),
                        **measured,
                    }
                    interventions.append(item)
                    journal.write(json.dumps(item) + "\n")
                    journal.flush()
                    if baseline["scenario_id"] == first_scenario and mode == "full":
                        ids, text = generate(
                            loaded, prompt["ids"], 1,
                            patch=(config["layer"], prompt["positions"][target], vector),
                        )
                        generated.append({
                            "example_id": baseline["example_id"], "target_position": target,
                            "source_axis": "none", "mode": mode, "generated_ids": ids,
                            "generated_text": text, "matches_forward": ids[0] == measured["top1_id"],
                        })

                fixed_axes = [(source, axes[source]) for source in config["source_axes"]]
                fixed_axes.extend((f"random_{axis_index:02d}", vector)
                                  for axis_index, vector in enumerate(banks[target]))
                for source, direction in fixed_axes:
                    deadline()
                    vector = equal_norm_vector(original, direction, common_scale, donor_unauthorized)
                    measured = patched_evaluate(
                        loaded, prompt, target, config["layer"], vector, original, direction
                    )
                    item = {
                        "example_id": baseline["example_id"],
                        "donor_id": donor_baseline["example_id"],
                        "target_position": target,
                        "source_axis": source,
                        "mode": "equal_norm",
                        "common_scale": common_scale,
                        **measured,
                    }
                    interventions.append(item)
                    journal.write(json.dumps(item) + "\n")
                    journal.flush()
                    if baseline["scenario_id"] == first_scenario \
                            and source in {"pre_mapping", "answer", "random_00"}:
                        ids, text = generate(
                            loaded, prompt["ids"], 1,
                            patch=(config["layer"], prompt["positions"][target], vector),
                        )
                        generated.append({
                            "example_id": baseline["example_id"], "target_position": target,
                            "source_axis": source, "mode": "equal_norm", "generated_ids": ids,
                            "generated_text": text, "matches_forward": ids[0] == measured["top1_id"],
                        })
            if (index + 1) % 4 == 0:
                print(f"patch {index + 1}/{len(baselines)} recipients, {len(interventions)} forwards, elapsed={time.monotonic()-started:.1f}s", flush=True)

    summary = summarize(interventions, baselines, config["bootstrap_samples"], config["seed"] + 10)
    result = {
        "completed_at_utc": now(),
        "elapsed_seconds": time.monotonic() - started,
        "mapping_accuracy": accuracies,
        "derived_scale": scale,
        "rows": interventions,
        "null_checks": nulls,
        "generation_audit": generated,
        "generation_agreement": all(row["matches_forward"] for row in generated),
        "summary": summary,
    }
    dump(output / "interventions.json", result)
    manifest(output)
    print(json.dumps({
        "mapping_accuracy": accuracies,
        "elapsed_seconds": result["elapsed_seconds"],
        "generation_agreement": result["generation_agreement"],
        "derived_scale": scale,
        "primary": {
            key: summary["paired_comparisons"][key]
            for key in ("answer/answer-minus-random_mean", "answer/answer-minus-pre_mapping",
                        "position-by-direction-interaction")
        },
        "checks": summary["predeclared_checks"],
    }, indent=2), flush=True)


def verify() -> None:
    root = Path(__file__).resolve().parents[2]
    config = read(root / "configs/stage5b_equal_norm.json")
    output = root / config["output"]
    frozen = read(output / "freeze.json")
    for relative, expected in frozen["source_sha256"].items():
        if sha(output / "source_snapshot" / relative) != expected:
            raise ValueError(f"Frozen source hash mismatch: {relative}")
    baseline = read(output / "baseline.json")["rows"]
    result = read(output / "interventions.json")
    recomputed = summarize(result["rows"], baseline, config["bootstrap_samples"], config["seed"] + 10)
    if recomputed != result["summary"]:
        raise ValueError("Stage 5b exact summary recomputation failed")
    requested = [row["requested_delta_norm"] for row in result["rows"] if row["mode"] == "equal_norm"]
    if max(abs(value - frozen["derived_scale"]["common_scale"]) for value in requested) > 0.003:
        raise ValueError("Equal-norm delivered request check failed")
    print(json.dumps({
        "source_snapshot_verified": True,
        "summary_exact_recomputation": True,
        "equal_norm_requested_delivery_within_bfloat16_tolerance": True,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("run", "verify", "manifest"), required=True)
    args = parser.parse_args()
    if args.phase == "run":
        run()
    elif args.phase == "verify":
        verify()
    else:
        root = Path(__file__).resolve().parents[2]
        config = read(root / "configs/stage5b_equal_norm.json")
        manifest(root / config["output"])


if __name__ == "__main__":
    main()
