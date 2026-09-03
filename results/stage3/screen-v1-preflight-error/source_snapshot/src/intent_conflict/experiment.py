from __future__ import annotations

import json
import platform
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
import transformers

from .analysis import (
    ScenarioSplit,
    behavior_gate,
    behavior_summary,
    bootstrap_mean_ci,
    permutation_test,
    representation_analysis,
    split_scenarios,
)
from .data import AuthorizationExample, build_examples, paired_key, render_messages
from .model import LoadedModel
from .report import write_plots_and_report
from .tokenization import apply_chat_template


def _json_dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _update_decision(result: dict[str, Any]) -> None:
    gate = result["behavior_gate"]
    if not result.get("representation") or not result.get("patching"):
        return
    selected_metrics = result["representation"]["selected_layer_metrics"]
    permutation = result["representation"]["permutation_test"]
    representation_supported = bool(
        selected_metrics["test"]["auroc"] > 0.75
        and selected_metrics["train_test_direction_cosine"] > 0
        and permutation["p_cross_dot_greater_equal"] < 0.05
    )
    causal_supported = result["patching"]["summary"]["strong_causal_support"]
    result["support_flags"] = {
        "behavior_supported": bool(gate["passed"]),
        "representation_supported": representation_supported,
        "causal_supported": bool(causal_supported),
    }
    if gate["passed"] and representation_supported and causal_supported:
        result["decision"] = (
            "The preregistered pilot supports a cross-scenario, causally active "
            "pre-action representation of whether the exact candidate action is "
            "authorized. Replication on an unquantized model is warranted."
        )
    elif not gate["passed"]:
        result["decision"] = (
            "The behavioral positive-control gate failed; mechanistic results are not "
            "scientifically interpretable."
        )
    elif not representation_supported:
        result["decision"] = (
            "Behavioral understanding was adequate, but the authorization direction "
            "did not meet held-out representation criteria. The current representation "
            "hypothesis is not supported and should not be extended to natural violations."
        )
    else:
        result["decision"] = (
            "Authorization was behaviorally and representationally decodable, but the "
            "activation swaps did not meet the preregistered causal-specificity gate. "
            "The current evidence supports readout, not a causal mechanism."
        )


def _matched_pairs(
    examples: list[AuthorizationExample], scenario_ids: tuple[str, ...]
) -> list[tuple[int, int]]:
    allowed = set(scenario_ids)
    grouped: dict[tuple[str, str], dict[bool, int]] = {}
    for idx, example in enumerate(examples):
        if example.scenario_id in allowed:
            grouped.setdefault(paired_key(example), {})[example.authorized] = idx
    pairs = []
    for values in grouped.values():
        if set(values) == {False, True}:
            pairs.append((values[True], values[False]))
    return pairs


def _run_patching(
    loaded: LoadedModel,
    examples: list[AuthorizationExample],
    prompt_ids: list[list[int]],
    activations: np.ndarray,
    margins: np.ndarray,
    layers: list[int],
    selected_layer: int,
    test_scenarios: tuple[str, ...],
    random_controls: int,
    seed: int,
) -> dict[str, Any]:
    layer_offset = layers.index(selected_layer)
    rng = np.random.default_rng(seed)
    rows = []
    sign_aligned = []
    random_by_control: list[list[float]] = [[] for _ in range(random_controls)]
    authorized_to_unauthorized_deltas = []
    unauthorized_to_authorized_deltas = []

    for authorized_idx, unauthorized_idx in _matched_pairs(examples, test_scenarios):
        authorized_activation = activations[authorized_idx, layer_offset]
        unauthorized_activation = activations[unauthorized_idx, layer_offset]

        patched_unauthorized = loaded.patched_margin(
            prompt_ids[unauthorized_idx], selected_layer, authorized_activation
        )
        patched_authorized = loaded.patched_margin(
            prompt_ids[authorized_idx], selected_layer, unauthorized_activation
        )
        delta_a_to_u = float(patched_unauthorized - margins[unauthorized_idx])
        delta_u_to_a = float(patched_authorized - margins[authorized_idx])
        aligned = float(0.5 * (-delta_a_to_u + delta_u_to_a))
        authorized_to_unauthorized_deltas.append(delta_a_to_u)
        unauthorized_to_authorized_deltas.append(delta_u_to_a)
        sign_aligned.append(aligned)

        pair_delta = unauthorized_activation - authorized_activation
        delta_norm = float(np.linalg.norm(pair_delta))
        random_rows = []
        for control_idx in range(random_controls):
            random_a = rng.normal(size=authorized_activation.shape)
            random_a = random_a / np.linalg.norm(random_a) * delta_norm
            random_u = rng.normal(size=unauthorized_activation.shape)
            random_u = random_u / np.linalg.norm(random_u) * delta_norm
            random_patched_unauthorized = loaded.patched_margin(
                prompt_ids[unauthorized_idx],
                selected_layer,
                unauthorized_activation + random_u,
            )
            random_patched_authorized = loaded.patched_margin(
                prompt_ids[authorized_idx],
                selected_layer,
                authorized_activation + random_a,
            )
            random_delta_a_to_u = float(
                random_patched_unauthorized - margins[unauthorized_idx]
            )
            random_delta_u_to_a = float(
                random_patched_authorized - margins[authorized_idx]
            )
            random_aligned = float(
                0.5 * (-random_delta_a_to_u + random_delta_u_to_a)
            )
            random_by_control[control_idx].append(random_aligned)
            random_rows.append(
                {
                    "control": control_idx,
                    "authorized_to_unauthorized_delta": random_delta_a_to_u,
                    "unauthorized_to_authorized_delta": random_delta_u_to_a,
                    "sign_aligned_effect": random_aligned,
                }
            )

        rows.append(
            {
                "scenario_id": examples[authorized_idx].scenario_id,
                "variant_id": examples[authorized_idx].variant_id,
                "authorized_baseline_margin": float(margins[authorized_idx]),
                "unauthorized_baseline_margin": float(margins[unauthorized_idx]),
                "authorized_to_unauthorized_patched_margin": float(
                    patched_unauthorized
                ),
                "unauthorized_to_authorized_patched_margin": float(
                    patched_authorized
                ),
                "authorized_to_unauthorized_delta": delta_a_to_u,
                "unauthorized_to_authorized_delta": delta_u_to_a,
                "sign_aligned_effect": aligned,
                "source_target_activation_distance": delta_norm,
                "random_controls": random_rows,
            }
        )

    clustered_effects: dict[str, list[float]] = {}
    for row in rows:
        clustered_effects.setdefault(row["scenario_id"], []).append(
            row["sign_aligned_effect"]
        )
    scenario_mean_effects = [
        float(np.mean(values)) for values in clustered_effects.values()
    ]
    ci = bootstrap_mean_ci(scenario_mean_effects, seed=seed + 1)
    random_control_means = [float(np.mean(values)) for values in random_by_control]
    random_all = [value for values in random_by_control for value in values]
    mean_aligned = float(np.mean(sign_aligned))
    main_exceeds_all_random_means = bool(
        mean_aligned > max(random_control_means)
    ) if random_control_means else True
    summary = {
        "n_pairs": len(rows),
        "n_independent_scenarios": len(scenario_mean_effects),
        "selected_layer": selected_layer,
        "authorized_to_unauthorized_mean_delta": float(
            np.mean(authorized_to_unauthorized_deltas)
        ),
        "unauthorized_to_authorized_mean_delta": float(
            np.mean(unauthorized_to_authorized_deltas)
        ),
        "sign_aligned_mean": mean_aligned,
        "sign_aligned_ci95": [ci[0], ci[1]],
        "ci_method": "scenario_cluster_bootstrap",
        "random_sign_aligned_mean": float(np.mean(random_all)),
        "random_control_means": random_control_means,
        "main_exceeds_all_random_control_means": main_exceeds_all_random_means,
        "bidirectional_signs_correct": bool(
            np.mean(authorized_to_unauthorized_deltas) < 0
            and np.mean(unauthorized_to_authorized_deltas) > 0
        ),
        "strong_causal_support": bool(
            ci[0] > 0
            and main_exceeds_all_random_means
            and np.mean(authorized_to_unauthorized_deltas) < 0
            and np.mean(unauthorized_to_authorized_deltas) > 0
        ),
    }
    return {"rows": rows, "summary": summary}


def run_experiment(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    loaded = LoadedModel.load(
        model_name=config["model_name"],
        model_loader=config.get("model_loader", "causal_lm"),
        device=config["device"],
        dtype=config["dtype"],
        local_files_only=bool(config["local_files_only"]),
        allow_fallback_chat_template=bool(
            config.get("allow_fallback_chat_template", False)
        ),
        quantization=config.get("quantization"),
        tokenizer_loader=config.get("tokenizer_loader", "auto"),
        max_memory=config.get("max_memory"),
        offload_folder=config.get("offload_folder"),
    )
    layers = [int(layer) for layer in config["layers"]]
    if not layers:
        raise ValueError("At least one layer is required")
    if min(layers) < 0 or max(layers) >= len(loaded.layers):
        raise ValueError(
            f"Requested layers {layers}, model has {len(loaded.layers)} decoder layers"
        )

    examples = build_examples(
        max_scenarios=int(config["max_scenarios"]),
        dataset_version=config.get("dataset_version", "stage1"),
    )
    if config.get("scenario_split"):
        configured_split = config["scenario_split"]
        split = ScenarioSplit(
            train=tuple(configured_split["train"]),
            validation=tuple(configured_split["validation"]),
            test=tuple(configured_split["test"]),
        )
        available = {example.scenario_id for example in examples}
        assigned = set(split.train) | set(split.validation) | set(split.test)
        if assigned != available:
            raise ValueError(
                "Configured scenario split must cover every available scenario exactly; "
                f"missing={sorted(available - assigned)}, extra={sorted(assigned - available)}"
            )
        if (
            set(split.train) & set(split.validation)
            or set(split.train) & set(split.test)
            or set(split.validation) & set(split.test)
        ):
            raise ValueError("Configured scenario split contains overlap")
    else:
        split = split_scenarios(
            examples,
            seed=seed,
            train_count=config.get("train_scenario_count"),
            validation_count=config.get("validation_scenario_count"),
        )

    prompt_ids: list[list[int]] = []
    evaluation_rows = []
    activation_rows = []
    for example in examples:
        ids = apply_chat_template(
            loaded.tokenizer,
            render_messages(example),
            enable_thinking=config.get("enable_thinking"),
            system_role_policy=config.get("system_role_policy", "native"),
        )
        evaluation = loaded.evaluate_prompt(ids, layers)
        prompt_ids.append(ids)
        activation_rows.append(evaluation.activations)
        evaluation_rows.append(
            {
                **asdict(example),
                "prompt_token_count": len(ids),
                "margin": evaluation.margin,
                "logp_execute": evaluation.logp_execute,
                "logp_block": evaluation.logp_block,
                "predicted_label": evaluation.predicted_label,
                "correct": evaluation.predicted_label == example.expected_label,
            }
        )

    activations = np.stack(activation_rows)
    margins = np.asarray([row["margin"] for row in evaluation_rows], dtype=float)
    overall_behavior = behavior_summary(examples, margins)
    summaries = {
        "overall": overall_behavior,
        "train": behavior_summary(examples, margins, split.train),
        "validation": behavior_summary(examples, margins, split.validation),
        "test": behavior_summary(examples, margins, split.test),
    }
    gate_scopes = config.get("behavior_gate_scopes", ["overall"])
    scoped_gates = {scope: behavior_gate(summaries[scope]) for scope in gate_scopes}
    if gate_scopes == ["overall"]:
        gate = scoped_gates["overall"]
    else:
        gate = {
            "passed": all(item["passed"] for item in scoped_gates.values()),
            "checks": {
                f"{scope}_{name}": passed
                for scope, scoped in scoped_gates.items()
                for name, passed in scoped["checks"].items()
            },
            "scopes": scoped_gates,
        }

    output_path = Path(config["output_path"])
    activation_path = Path(config["activation_output_path"])
    activation_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        activation_path,
        activations=activations,
        layers=np.asarray(layers),
        example_ids=np.asarray([example.example_id for example in examples]),
    )

    checkpoint_quantization_config = getattr(
        loaded.model.config, "quantization_config", None
    )
    if hasattr(checkpoint_quantization_config, "to_dict"):
        checkpoint_quantization_config = checkpoint_quantization_config.to_dict()

    result: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "config": config,
        "provenance": {
            "model_name": config["model_name"],
            "model_class": type(loaded.model).__name__,
            "model_config_revision": getattr(loaded.model.config, "_commit_hash", None),
            "quantization": config.get("quantization"),
            "checkpoint_quantization_config": checkpoint_quantization_config,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "scikit_learn": sklearn.__version__,
            "cuda_available": torch.cuda.is_available(),
            "num_decoder_layers": len(loaded.layers),
            "execute_token_id": loaded.execute_token_id,
            "block_token_id": loaded.block_token_id,
            "hf_device_map": {
                str(key): str(value)
                for key, value in getattr(loaded.model, "hf_device_map", {}).items()
            },
            "parameter_dtypes": sorted(
                {str(parameter.dtype) for parameter in loaded.model.parameters()}
            ),
        },
        "split": asdict(split),
        "behavior_rows": evaluation_rows,
        "behavior_summary": summaries,
        "behavior_gate": gate,
        "activation_path": str(activation_path.resolve()),
        "representation": None,
        "patching": None,
        "decision": "Behavioral screening only; no mechanistic conclusion was attempted.",
    }

    formal_requested = config.get("mode", "screen") == "formal"
    require_gate = bool(config.get("require_behavior_gate", True))
    if formal_requested and (gate["passed"] or not require_gate):
        layer_rows, selected_layer = representation_analysis(
            examples, activations, layers, split
        )
        selected_metrics = next(
            row for row in layer_rows if row["layer"] == selected_layer
        )
        selected_offset = layers.index(selected_layer)
        permutation = permutation_test(
            examples,
            activations[:, selected_offset, :],
            split,
            permutations=int(config.get("permutations", 500)),
            seed=seed + 100,
            exact=bool(config.get("exact_permutation", False)),
        )
        result["representation"] = {
            "layers": layer_rows,
            "selected_layer": selected_layer,
            "selected_layer_metrics": selected_metrics,
            "permutation_test": permutation,
        }
        result["patching"] = _run_patching(
            loaded,
            examples,
            prompt_ids,
            activations,
            margins,
            layers,
            selected_layer,
            split.test,
            random_controls=int(config.get("random_controls", 5)),
            seed=seed + 200,
        )
        _update_decision(result)
    elif formal_requested:
        result["decision"] = (
            "The behavioral positive-control gate failed, so the formal representation and "
            "patching experiment was stopped as unsuitable under the preregistration."
        )

    _json_dump(output_path, result)
    plot_dir = Path(config["plot_dir"])
    write_plots_and_report(result, plot_dir)
    return result


def reanalyze_experiment(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    output_path = Path(config["output_path"])
    result = json.loads(output_path.read_text(encoding="utf-8"))
    previous_selected_layer = (
        result.get("representation", {}).get("selected_layer")
        if result.get("representation")
        else None
    )
    activation_archive = np.load(Path(config["activation_output_path"]))
    activations = activation_archive["activations"]
    layers = [int(value) for value in activation_archive["layers"].tolist()]
    examples = build_examples(
        max_scenarios=int(config["max_scenarios"]),
        dataset_version=config.get("dataset_version", "stage1"),
    )
    split = ScenarioSplit(
        train=tuple(result["split"]["train"]),
        validation=tuple(result["split"]["validation"]),
        test=tuple(result["split"]["test"]),
    )
    layer_rows, selected_layer = representation_analysis(
        examples, activations, layers, split
    )
    selected_metrics = next(row for row in layer_rows if row["layer"] == selected_layer)
    selected_offset = layers.index(selected_layer)
    permutation = permutation_test(
        examples,
        activations[:, selected_offset, :],
        split,
        permutations=int(config.get("permutations", 500)),
        seed=int(config["seed"]) + 100,
        exact=bool(config.get("exact_permutation", False)),
    )
    result["representation"] = {
        "layers": layer_rows,
        "selected_layer": selected_layer,
        "selected_layer_metrics": selected_metrics,
        "permutation_test": permutation,
    }
    result["reanalysis"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "Corrected the permutation statistic to preserve effect magnitude and cluster "
            "label flips by scenario. The prior normalized-direction AUROC null was invalid "
            "for nearly collinear matched-pair deltas, and wording variants are not "
            "independent experimental units."
        ),
        "model_forward_passes_rerun": False,
        "previous_selected_layer": previous_selected_layer,
        "layer_selection_changed": selected_layer != previous_selected_layer,
    }
    if result.get("patching"):
        patching_rows = result["patching"]["rows"]
        grouped: dict[str, list[float]] = {}
        for row in patching_rows:
            grouped.setdefault(row["scenario_id"], []).append(row["sign_aligned_effect"])
        scenario_means = [float(np.mean(values)) for values in grouped.values()]
        ci = bootstrap_mean_ci(scenario_means, seed=int(config["seed"]) + 201)
        summary = result["patching"]["summary"]
        summary["n_independent_scenarios"] = len(scenario_means)
        summary["sign_aligned_ci95"] = [ci[0], ci[1]]
        summary["ci_method"] = "scenario_cluster_bootstrap"
        summary["strong_causal_support"] = bool(
            ci[0] > 0
            and summary["main_exceeds_all_random_control_means"]
            and summary["authorized_to_unauthorized_mean_delta"] < 0
            and summary["unauthorized_to_authorized_mean_delta"] > 0
        )
    _update_decision(result)
    _json_dump(output_path, result)
    write_plots_and_report(result, Path(config["plot_dir"]))
    return result


def run_robustness(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Run post-hoc controls without modifying the preregistered primary result."""
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    primary = json.loads(Path(config["output_path"]).read_text(encoding="utf-8"))
    if not primary.get("representation") or not primary.get("patching"):
        raise ValueError("Primary formal experiment must be complete before robustness controls")
    archive = np.load(Path(config["activation_output_path"]))
    activations = archive["activations"]
    layers = [int(value) for value in archive["layers"].tolist()]
    examples = build_examples(
        max_scenarios=int(config["max_scenarios"]),
        dataset_version=config.get("dataset_version", "stage1"),
    )
    margins = np.asarray([row["margin"] for row in primary["behavior_rows"]], dtype=float)
    test_scenarios = tuple(primary["split"]["test"])
    selected_layer = int(primary["representation"]["selected_layer"])
    control_layer = int(config.get("robustness_control_layer", 4))
    if control_layer not in layers:
        raise ValueError(f"Robustness control layer {control_layer} was not saved")

    loaded = LoadedModel.load(
        model_name=config["model_name"],
        model_loader=config.get("model_loader", "causal_lm"),
        device=config["device"],
        dtype=config["dtype"],
        local_files_only=bool(config["local_files_only"]),
        allow_fallback_chat_template=bool(config.get("allow_fallback_chat_template", False)),
        quantization=config.get("quantization"),
        tokenizer_loader=config.get("tokenizer_loader", "auto"),
        max_memory=config.get("max_memory"),
        offload_folder=config.get("offload_folder"),
    )
    prompt_ids = [
        apply_chat_template(
            loaded.tokenizer,
            render_messages(example),
            enable_thinking=config.get("enable_thinking"),
            system_role_policy=config.get("system_role_policy", "native"),
        )
        for example in examples
    ]

    control_offset = layers.index(control_layer)
    wrong_layer_rows = []
    wrong_layer_aligned = []
    wrong_a_to_u = []
    wrong_u_to_a = []
    for authorized_idx, unauthorized_idx in _matched_pairs(examples, test_scenarios):
        patched_unauthorized = loaded.patched_margin(
            prompt_ids[unauthorized_idx],
            control_layer,
            activations[authorized_idx, control_offset],
        )
        patched_authorized = loaded.patched_margin(
            prompt_ids[authorized_idx],
            control_layer,
            activations[unauthorized_idx, control_offset],
        )
        delta_a_to_u = float(patched_unauthorized - margins[unauthorized_idx])
        delta_u_to_a = float(patched_authorized - margins[authorized_idx])
        aligned = float(0.5 * (-delta_a_to_u + delta_u_to_a))
        wrong_a_to_u.append(delta_a_to_u)
        wrong_u_to_a.append(delta_u_to_a)
        wrong_layer_aligned.append(aligned)
        wrong_layer_rows.append(
            {
                "scenario_id": examples[authorized_idx].scenario_id,
                "variant_id": examples[authorized_idx].variant_id,
                "authorized_to_unauthorized_delta": delta_a_to_u,
                "unauthorized_to_authorized_delta": delta_u_to_a,
                "sign_aligned_effect": aligned,
            }
        )

    # Swap between the two wording templates with the same candidate position,
    # scenario, and authorization state. This preserves the decision-relevant relation.
    same_condition_rows = []
    allowed = set(test_scenarios)
    grouped: dict[tuple[str, bool, str], list[int]] = {}
    for idx, example in enumerate(examples):
        if example.scenario_id in allowed:
            grouped.setdefault(
                (example.scenario_id, example.authorized, example.candidate_position), []
            ).append(idx)
    selected_offset = layers.index(selected_layer)
    for (scenario_id, authorized, position), indices in grouped.items():
        if len(indices) != 2:
            raise ValueError("Expected exactly two wording templates per position control")
        for source_idx, target_idx in (indices, indices[::-1]):
            patched = loaded.patched_margin(
                prompt_ids[target_idx],
                selected_layer,
                activations[source_idx, selected_offset],
            )
            delta = float(patched - margins[target_idx])
            same_condition_rows.append(
                {
                    "scenario_id": scenario_id,
                    "authorized": authorized,
                    "candidate_position": position,
                    "source_variant_id": examples[source_idx].variant_id,
                    "target_variant_id": examples[target_idx].variant_id,
                    "baseline_margin": float(margins[target_idx]),
                    "patched_margin": float(patched),
                    "delta": delta,
                    "decision_flipped": bool((margins[target_idx] >= 0) != (patched >= 0)),
                }
            )

    wrong_grouped: dict[str, list[float]] = {}
    for row in wrong_layer_rows:
        wrong_grouped.setdefault(row["scenario_id"], []).append(row["sign_aligned_effect"])
    wrong_scenario_means = [float(np.mean(values)) for values in wrong_grouped.values()]
    wrong_ci = bootstrap_mean_ci(wrong_scenario_means, seed=seed + 301)
    same_deltas = np.asarray([row["delta"] for row in same_condition_rows])
    same_flips = np.asarray([row["decision_flipped"] for row in same_condition_rows])
    main = primary["patching"]["summary"]
    wrong_mean = float(np.mean(wrong_layer_aligned))
    same_abs = float(np.mean(np.abs(same_deltas)))
    summary = {
        "status": "post_hoc_exploratory_controls",
        "selected_layer": selected_layer,
        "control_layer": control_layer,
        "main_sign_aligned_mean": float(main["sign_aligned_mean"]),
        "wrong_layer_authorized_to_unauthorized_mean_delta": float(np.mean(wrong_a_to_u)),
        "wrong_layer_unauthorized_to_authorized_mean_delta": float(np.mean(wrong_u_to_a)),
        "wrong_layer_sign_aligned_mean": wrong_mean,
        "wrong_layer_sign_aligned_ci95": [wrong_ci[0], wrong_ci[1]],
        "wrong_layer_ci_method": "scenario_cluster_bootstrap",
        "same_condition_n_swaps": len(same_condition_rows),
        "same_condition_mean_delta": float(np.mean(same_deltas)),
        "same_condition_mean_absolute_delta": same_abs,
        "same_condition_decision_flip_rate": float(np.mean(same_flips)),
        "main_exceeds_wrong_layer_effect": bool(main["sign_aligned_mean"] > wrong_mean),
        "main_exceeds_same_condition_mean_absolute_effect": bool(
            main["sign_aligned_mean"] > same_abs
        ),
    }
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "post_hoc_exploratory_controls",
        "primary_result_path": str(Path(config["output_path"]).resolve()),
        "config_path": str(config_path.resolve()),
        "summary": summary,
        "wrong_layer_rows": wrong_layer_rows,
        "same_condition_rows": same_condition_rows,
    }
    robustness_path = Path(
        config.get("robustness_output_path", "results/qwen_formal_robustness.json")
    )
    _json_dump(robustness_path, result)
    return result
