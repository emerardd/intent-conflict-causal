"""Stage 5 independent confirmation of position by direction identity."""
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
from .stage3_remap import render_remap
from .stage3_data import tokenize_trial
from .stage5_data import build_confirmation


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Cannot normalize a zero vector")
    return vector.astype(float) / norm


def pair_indices(rows: list[dict]) -> list[int]:
    lookup: dict[tuple, int] = {}
    fields = ("scenario_id", "reversed_mapping", "authorized")
    for index, row in enumerate(rows):
        key = tuple(row[field] for field in fields)
        if key in lookup:
            raise ValueError("Duplicate Stage 5 donor key")
        lookup[key] = index
    pairs: list[int] = []
    for row in rows:
        key = (row["scenario_id"], row["reversed_mapping"], not row["authorized"])
        if key not in lookup:
            raise ValueError(f"Missing donor for {row['example_id']}")
        donor = rows[lookup[key]]
        if donor["candidate_command"] != row["candidate_command"]:
            raise ValueError("Candidate mismatch within donor pair")
        pairs.append(lookup[key])
    return pairs


def random_axes(axes: dict[str, np.ndarray], positions: list[str], seed: int) -> dict[str, np.ndarray]:
    basis = np.stack([unit(axes[position]) for position in positions], axis=1)
    q, _ = np.linalg.qr(basis)
    rng = np.random.default_rng(seed)
    output: dict[str, np.ndarray] = {}
    for position in positions:
        candidate = rng.normal(size=basis.shape[0])
        candidate = candidate - q @ (q.T @ candidate)
        for previous in output.values():
            candidate = candidate - previous * float(previous @ candidate)
        output[position] = unit(candidate)
    return output


def build_components(
    recipient: np.ndarray,
    donor: np.ndarray,
    direction: np.ndarray,
    random_axis: np.ndarray,
    donor_unauthorized: bool,
) -> dict[str, np.ndarray]:
    delta = donor.astype(float) - recipient.astype(float)
    coefficient = float(delta @ direction)
    sign = 1.0 if donor_unauthorized else -1.0
    parallel = coefficient * direction
    return {
        "full": delta,
        "random_full": sign * float(np.linalg.norm(delta)) * random_axis,
        "parallel": parallel,
        "random_parallel": coefficient * random_axis,
    }


def _interval(values: list[float], samples: int, seed: int) -> list[float] | None:
    if len(values) < 4:
        return None
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    boot = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return np.quantile(boot, [0.025, 0.975]).tolist()


def summarize(
    rows: list[dict],
    baselines: list[dict],
    bootstrap_samples: int,
    seed: int,
    equivalence_bound: float,
) -> dict:
    baseline = {row["example_id"]: row for row in baselines}
    grouped: dict[tuple[str, str, str], list[tuple[dict, float, float]]] = defaultdict(list)
    for row in rows:
        recipient = baseline[row["example_id"]]
        donor = baseline[row["donor_id"]]
        sign = 1.0 if donor["expected_label"] == "B" else -1.0
        effect = sign * (float(row["margin"]) - float(recipient["margin"]))
        gap = sign * (float(donor["margin"]) - float(recipient["margin"]))
        key = (row["target_position"], row["source_axis"], row["mode"])
        grouped[key].append((row, effect, gap))

    effects: dict[str, dict] = {}
    cluster_effects: dict[tuple[str, str, str], dict[str, float]] = {}
    for key, group in grouped.items():
        scenario: dict[str, list[float]] = defaultdict(list)
        for row, effect, _ in group:
            scenario[baseline[row["example_id"]]["scenario_id"]].append(effect)
        means = {sid: float(np.mean(values)) for sid, values in scenario.items()}
        cluster_effects[key] = means
        target, source, mode = key
        label = f"{target}/{source}/{mode}"
        effects[label] = {
            "mean": float(np.mean(list(means.values()))),
            "ci95": _interval(list(means.values()), bootstrap_samples, seed),
            "n_scenarios": len(means),
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
            "excluded_nonpositive_gap": sum(gap <= 1e-6 for _, _, gap in group),
        }

    def compare(left: tuple[str, str, str], right: tuple[str, str, str], offset: int) -> dict:
        if set(cluster_effects[left]) != set(cluster_effects[right]):
            raise ValueError("Paired comparison has different scenario identities")
        diffs = [cluster_effects[left][sid] - cluster_effects[right][sid]
                 for sid in cluster_effects[left]]
        return {"mean": float(np.mean(diffs)),
                "ci95": _interval(diffs, bootstrap_samples, seed + offset),
                "scenario_differences": dict(zip(cluster_effects[left], diffs))}

    comparisons: dict[str, dict] = {}
    offset = 1
    for target in ("pre_mapping", "answer"):
        comparisons[f"{target}/full-minus-random_full"] = compare(
            (target, "none", "full"), (target, "none", "random_full"), offset)
        offset += 1
        for source in ("pre_mapping", "answer"):
            comparisons[f"{target}/{source}/parallel-minus-random"] = compare(
                (target, source, "parallel"), (target, source, "random_parallel"), offset)
            offset += 1

    comparisons["position-interaction/local-parallel"] = compare(
        ("answer", "answer", "parallel"),
        ("pre_mapping", "pre_mapping", "parallel"), offset)
    offset += 1
    comparisons["answer/axis-identity/local-minus-cross"] = compare(
        ("answer", "answer", "parallel"),
        ("answer", "pre_mapping", "parallel"), offset)
    offset += 1
    comparisons["pre_mapping/axis-identity/local-minus-cross"] = compare(
        ("pre_mapping", "pre_mapping", "parallel"),
        ("pre_mapping", "answer", "parallel"), offset)
    offset += 1
    comparisons["pre-axis/position-answer-minus-pre"] = compare(
        ("answer", "pre_mapping", "parallel"),
        ("pre_mapping", "pre_mapping", "parallel"), offset)
    offset += 1
    comparisons["answer-axis/position-answer-minus-pre"] = compare(
        ("answer", "answer", "parallel"),
        ("pre_mapping", "answer", "parallel"), offset)

    answer_specific = comparisons["answer/answer/parallel-minus-random"]["ci95"]
    answer_full = comparisons["answer/full-minus-random_full"]["ci95"]
    interaction = comparisons["position-interaction/local-parallel"]["ci95"]
    pre_full = effects["pre_mapping/none/full"]["ci95"]
    decisions = {
        "answer_local_direction_specific": bool(answer_specific and answer_specific[0] > 0),
        "answer_full_state_positive_control": bool(answer_full and answer_full[0] > 0),
        "position_interaction_positive": bool(interaction and interaction[0] > 0),
        "pre_mapping_full_within_equivalence_bound": bool(
            pre_full and pre_full[0] > -equivalence_bound and pre_full[1] < equivalence_bound
        ),
    }
    return {
        "effects": effects,
        "paired_comparisons": comparisons,
        "predeclared_decisions": decisions,
        "all_four_checks_pass": all(decisions.values()),
        "estimand": "Donor-answer-aligned change in B/A log-probability margin after scenario averaging.",
        "scope": "Independent new-scenario confirmation of position and direction identity; not natural violation awareness.",
    }


def _copy_sources(root: Path, output: Path, config_path: Path) -> tuple[list[Path], dict[str, str]]:
    sources = [
        root / "src/intent_conflict/stage5.py",
        root / "src/intent_conflict/stage5_data.py",
        root / "src/intent_conflict/stage3.py",
        root / "src/intent_conflict/stage3_model.py",
        root / "src/intent_conflict/stage3_data.py",
        root / "src/intent_conflict/stage3_remap.py",
        root / "src/intent_conflict/stage4.py",
        root / "src/intent_conflict/model.py",
        root / "src/intent_conflict/tokenization.py",
        root / "tests/test_stage5.py",
        root / "docs/stage5-preregistration.md",
        config_path,
    ]
    hashes = {path.relative_to(root).as_posix(): sha(path) for path in sources}
    for path in sources:
        destination = output / "source_snapshot" / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    return sources, hashes


def run() -> None:
    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs/stage5_confirmation.json"
    config = read(config_path)
    output = root / config["output"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)

    axes_path = root / config["axes_reference"]
    axes_archive = np.load(axes_path, allow_pickle=False)
    axes = {position: unit(axes_archive[position].astype(float)) for position in config["source_axes"]}
    controls = random_axes(axes, config["positions"], config["seed"] + 1)
    np.savez_compressed(output / "frozen_axes.npz", **axes,
                        **{f"{key}_random": value for key, value in controls.items()})
    _, source_hashes = _copy_sources(root, output, config_path)

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

    trials = build_confirmation(config["seed"])
    if len({trial.scenario_id for trial in trials}) != config["n_scenarios"]:
        raise ValueError("Frozen scenario count mismatch")
    prompts: list[dict] = []
    for trial in trials:
        ids, positions = tokenize_trial(loaded.tokenizer, trial, render_remap(trial))
        prompts.append({"example_id": trial.example_id, "ids": ids, "positions": positions})
    pairs = pair_indices([trial.row() for trial in trials])
    for index, donor_index in enumerate(pairs):
        for position in config["positions"]:
            if len(prompts[index]["ids"]) != len(prompts[donor_index]["ids"]):
                raise ValueError("Donor token lengths are not aligned")
            if prompts[index]["positions"][position] != prompts[donor_index]["positions"][position]:
                raise ValueError("Donor measurement positions are not aligned")

    dump(output / "prompts.json", prompts)
    dump(output / "freeze.json", {
        "created_at_utc": now(),
        "config": config,
        "python": platform.python_version(),
        "source_sha256": source_hashes,
        "axes_reference_sha256": sha(axes_path),
        "example_ids": [trial.example_id for trial in trials],
        "donor_indices": pairs,
        "note": "Created before any Stage 5 model forward; model load and tokenization are not response inspection.",
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
            raise TimeoutError("Frozen Stage 5 runtime bound exceeded; partial journals retained")

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
    nulls: list[dict] = []
    generated: list[dict] = []
    first_scenario = baselines[0]["scenario_id"]
    with (output / "interventions-progress.jsonl").open("x", encoding="utf-8") as journal:
        for index, (baseline, prompt) in enumerate(zip(baselines, prompts)):
            donor_index = pairs[index]
            donor_baseline = baselines[donor_index]
            for position_index, target in enumerate(config["positions"]):
                original = x[index, position_index]
                donor = x[donor_index, position_index]
                random_axis = controls[target]
                if index == 0:
                    measured = patched_evaluate(
                        loaded, prompt, target, config["layer"], original, original, axes[target]
                    )
                    if abs(measured["margin"] - baseline["margin"]) > 1e-4 \
                            or measured["top1_id"] != baseline["top1_id"]:
                        raise ValueError("Null patch differs from baseline")
                    nulls.append({"target_position": target, **measured})

                matched = build_components(
                    original, donor, axes[target], random_axis,
                    donor_unauthorized=not donor_baseline["authorized"],
                )
                for mode in ("full", "random_full"):
                    deadline()
                    vector = original.astype(float) + matched[mode]
                    measured = patched_evaluate(
                        loaded, prompt, target, config["layer"], vector, original, axes[target]
                    )
                    item = {
                        "example_id": baseline["example_id"],
                        "donor_id": donor_baseline["example_id"],
                        "target_position": target,
                        "source_axis": "none",
                        "mode": mode,
                        "natural_delta_norm": float(np.linalg.norm(donor - original)),
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

                for source in config["source_axes"]:
                    components = build_components(
                        original, donor, axes[source], random_axis,
                        donor_unauthorized=not donor_baseline["authorized"],
                    )
                    coefficient = float((donor.astype(float) - original.astype(float)) @ axes[source])
                    for mode in ("parallel", "random_parallel"):
                        deadline()
                        vector = original.astype(float) + components[mode]
                        measured = patched_evaluate(
                            loaded, prompt, target, config["layer"], vector, original, axes[source]
                        )
                        item = {
                            "example_id": baseline["example_id"],
                            "donor_id": donor_baseline["example_id"],
                            "target_position": target,
                            "source_axis": source,
                            "mode": mode,
                            "natural_coefficient": coefficient,
                            **measured,
                        }
                        interventions.append(item)
                        journal.write(json.dumps(item) + "\n")
                        journal.flush()
                        if baseline["scenario_id"] == first_scenario and mode == "parallel":
                            ids, text = generate(
                                loaded, prompt["ids"], 1,
                                patch=(config["layer"], prompt["positions"][target], vector),
                            )
                            generated.append({
                                "example_id": baseline["example_id"], "target_position": target,
                                "source_axis": source, "mode": mode, "generated_ids": ids,
                                "generated_text": text, "matches_forward": ids[0] == measured["top1_id"],
                            })
            if (index + 1) % 4 == 0:
                print(f"patch {index + 1}/{len(baselines)} recipients, {len(interventions)} forwards, elapsed={time.monotonic()-started:.1f}s", flush=True)

    result = {
        "completed_at_utc": now(),
        "elapsed_seconds": time.monotonic() - started,
        "mapping_accuracy": accuracies,
        "rows": interventions,
        "null_checks": nulls,
        "generation_audit": generated,
        "generation_agreement": all(row["matches_forward"] for row in generated),
        "summary": summarize(
            interventions, baselines, config["bootstrap_samples"], config["seed"] + 10,
            config["pre_mapping_equivalence_bound"],
        ),
    }
    dump(output / "interventions.json", result)
    manifest(output)
    print(json.dumps({
        "mapping_accuracy": accuracies,
        "elapsed_seconds": result["elapsed_seconds"],
        "generation_agreement": result["generation_agreement"],
        "summary": result["summary"],
    }, indent=2), flush=True)


def verify() -> None:
    root = Path(__file__).resolve().parents[2]
    config = read(root / "configs/stage5_confirmation.json")
    output = root / config["output"]
    frozen = read(output / "freeze.json")
    for relative, expected in frozen["source_sha256"].items():
        actual = sha(output / "source_snapshot" / relative)
        if actual != expected:
            raise ValueError(f"Frozen source hash mismatch: {relative}")
    baseline = read(output / "baseline.json")["rows"]
    interventions = read(output / "interventions.json")
    actual = summarize(
        interventions["rows"], baseline, config["bootstrap_samples"], config["seed"] + 10,
        config["pre_mapping_equivalence_bound"],
    )
    if actual != interventions["summary"]:
        raise ValueError("Stage 5 exact summary recomputation failed")
    print(json.dumps({"source_snapshot_verified": True, "summary_exact_recomputation": True}, indent=2))


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
        config = read(root / "configs/stage5_confirmation.json")
        manifest(root / config["output"])


if __name__ == "__main__":
    main()
