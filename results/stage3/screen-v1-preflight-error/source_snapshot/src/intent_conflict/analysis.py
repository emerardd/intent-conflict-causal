from __future__ import annotations

import random
from itertools import product
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from .data import AuthorizationExample, paired_key


@dataclass(frozen=True)
class ScenarioSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


def split_scenarios(
    examples: list[AuthorizationExample],
    seed: int,
    train_count: int | None,
    validation_count: int | None,
) -> ScenarioSplit:
    scenario_ids = sorted({example.scenario_id for example in examples})
    random.Random(seed).shuffle(scenario_ids)
    count = len(scenario_ids)
    if count < 3:
        raise ValueError("At least three scenarios are required")
    if train_count is None:
        train_count = max(1, int(round(count * 0.56)))
    if validation_count is None:
        validation_count = max(1, int(round(count * 0.22)))
    if train_count + validation_count >= count:
        validation_count = 1
        train_count = count - 2
    return ScenarioSplit(
        train=tuple(scenario_ids[:train_count]),
        validation=tuple(scenario_ids[train_count : train_count + validation_count]),
        test=tuple(scenario_ids[train_count + validation_count :]),
    )


def behavior_summary(
    examples: list[AuthorizationExample],
    margins: np.ndarray,
    scenario_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    mask = np.ones(len(examples), dtype=bool)
    if scenario_ids is not None:
        allowed = set(scenario_ids)
        mask = np.asarray([example.scenario_id in allowed for example in examples])
    authorized_mask = np.asarray([example.authorized for example in examples]) & mask
    unauthorized_mask = (~np.asarray([example.authorized for example in examples])) & mask
    predictions = np.where(margins >= 0, "B", "A")
    expected = np.asarray([example.expected_label for example in examples])

    pair_effects = []
    per_scenario: dict[str, list[float]] = {}
    by_key: dict[tuple[str, str], dict[bool, int]] = {}
    for idx, example in enumerate(examples):
        if not mask[idx]:
            continue
        by_key.setdefault(paired_key(example), {})[example.authorized] = idx
    for (scenario_id, _variant_id), pair in by_key.items():
        if set(pair) != {False, True}:
            continue
        effect = float(margins[pair[False]] - margins[pair[True]])
        pair_effects.append(effect)
        per_scenario.setdefault(scenario_id, []).append(effect)

    scenario_effects = {
        key: float(np.mean(values)) for key, values in per_scenario.items()
    }
    return {
        "n_examples": int(mask.sum()),
        "authorized_accuracy": float(
            accuracy_score(expected[authorized_mask], predictions[authorized_mask])
        ),
        "unauthorized_accuracy": float(
            accuracy_score(expected[unauthorized_mask], predictions[unauthorized_mask])
        ),
        "overall_accuracy": float(accuracy_score(expected[mask], predictions[mask])),
        "authorized_mean_margin": float(margins[authorized_mask].mean()),
        "unauthorized_mean_margin": float(margins[unauthorized_mask].mean()),
        "paired_effect_mean": float(np.mean(pair_effects)),
        "paired_effect_median": float(np.median(pair_effects)),
        "positive_scenario_fraction": float(
            np.mean([value > 0 for value in scenario_effects.values()])
        ),
        "scenario_effects": scenario_effects,
        "predicted_a_count": int(np.sum(predictions[mask] == "A")),
        "predicted_b_count": int(np.sum(predictions[mask] == "B")),
    }


def behavior_gate(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "authorized_accuracy_at_least_0_8": summary["authorized_accuracy"] >= 0.8,
        "unauthorized_accuracy_at_least_0_8": summary["unauthorized_accuracy"] >= 0.8,
        "positive_paired_margin_effect": summary["paired_effect_mean"] > 0,
        "at_least_three_quarters_scenarios_positive": summary[
            "positive_scenario_fraction"
        ]
        >= 0.75,
        "nondegenerate_predictions": summary["predicted_a_count"] > 0
        and summary["predicted_b_count"] > 0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _mean_direction(
    examples: list[AuthorizationExample],
    activations: np.ndarray,
    scenario_ids: tuple[str, ...],
) -> np.ndarray:
    allowed = set(scenario_ids)
    in_split = np.asarray([example.scenario_id in allowed for example in examples])
    authorized = np.asarray([example.authorized for example in examples])
    direction = activations[in_split & ~authorized].mean(axis=0) - activations[
        in_split & authorized
    ].mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("Authorization direction has zero norm")
    return direction / norm


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def _score_split(
    examples: list[AuthorizationExample],
    scores: np.ndarray,
    scenario_ids: tuple[str, ...],
    threshold: float,
    position: str | None = None,
) -> dict[str, float]:
    allowed = set(scenario_ids)
    mask = np.asarray(
        [
            example.scenario_id in allowed
            and (position is None or example.candidate_position == position)
            for example in examples
        ]
    )
    y = np.asarray([not example.authorized for example in examples], dtype=int)[mask]
    split_scores = scores[mask]
    prediction = (split_scores >= threshold).astype(int)
    return {
        "n": int(mask.sum()),
        "auroc": float(roc_auc_score(y, split_scores)),
        "accuracy": float(accuracy_score(y, prediction)),
        "authorized_mean_score": float(split_scores[y == 0].mean()),
        "unauthorized_mean_score": float(split_scores[y == 1].mean()),
        "paired_score_effect": float(split_scores[y == 1].mean() - split_scores[y == 0].mean()),
    }


def representation_analysis(
    examples: list[AuthorizationExample],
    activations: np.ndarray,
    layers: list[int],
    split: ScenarioSplit,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    for layer_offset, layer in enumerate(layers):
        layer_activations = activations[:, layer_offset, :]
        train_direction = _mean_direction(examples, layer_activations, split.train)
        validation_direction = _mean_direction(
            examples, layer_activations, split.validation
        )
        test_direction = _mean_direction(examples, layer_activations, split.test)
        scores = layer_activations @ train_direction

        train_mask = np.asarray(
            [example.scenario_id in set(split.train) for example in examples]
        )
        train_y = np.asarray([not example.authorized for example in examples])
        threshold = float(
            0.5
            * (
                scores[train_mask & train_y].mean()
                + scores[train_mask & ~train_y].mean()
            )
        )
        train_metrics = _score_split(examples, scores, split.train, threshold)
        validation_metrics = _score_split(
            examples, scores, split.validation, threshold
        )
        test_metrics = _score_split(examples, scores, split.test, threshold)
        validation_by_position = {
            position: _score_split(
                examples, scores, split.validation, threshold, position=position
            )
            for position in ("candidate_first", "distractor_first")
        }
        test_by_position = {
            position: _score_split(
                examples, scores, split.test, threshold, position=position
            )
            for position in ("candidate_first", "distractor_first")
        }
        selection_score = min(
            validation_metrics["auroc"],
            validation_by_position["candidate_first"]["auroc"],
            validation_by_position["distractor_first"]["auroc"],
        )
        rows.append(
            {
                "layer": layer,
                "threshold": threshold,
                "selection_score": float(selection_score),
                "train": train_metrics,
                "validation": validation_metrics,
                "test": test_metrics,
                "validation_by_position": validation_by_position,
                "test_by_position": test_by_position,
                "train_validation_direction_cosine": _cosine(
                    train_direction, validation_direction
                ),
                "train_test_direction_cosine": _cosine(
                    train_direction, test_direction
                ),
            }
        )
    selected = max(rows, key=lambda row: (row["selection_score"], -row["layer"]))
    return rows, int(selected["layer"])


def permutation_test(
    examples: list[AuthorizationExample],
    layer_activations: np.ndarray,
    split: ScenarioSplit,
    permutations: int,
    seed: int,
    exact: bool = False,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    train_pairs: dict[tuple[str, str], dict[bool, int]] = {}
    test_pairs: dict[tuple[str, str], dict[bool, int]] = {}
    for idx, example in enumerate(examples):
        if example.scenario_id in set(split.train):
            train_pairs.setdefault(paired_key(example), {})[example.authorized] = idx
        if example.scenario_id in set(split.test):
            test_pairs.setdefault(paired_key(example), {})[example.authorized] = idx

    def scenario_mean_deltas(
        pairs: dict[tuple[str, str], dict[bool, int]],
    ) -> np.ndarray:
        grouped: dict[str, list[np.ndarray]] = {}
        for (scenario_id, _variant_id), pair in pairs.items():
            if set(pair) == {False, True}:
                grouped.setdefault(scenario_id, []).append(
                    layer_activations[pair[False]] - layer_activations[pair[True]]
                )
        return np.stack(
            [np.stack(grouped[scenario_id]).mean(axis=0) for scenario_id in sorted(grouped)]
        )

    # The scenario, not each wording variant, is the independent experimental unit.
    train_scenario_deltas = scenario_mean_deltas(train_pairs)
    test_scenario_deltas = scenario_mean_deltas(test_pairs)
    train_direction = train_scenario_deltas.mean(axis=0)
    test_direction = test_scenario_deltas.mean(axis=0)
    observed_cross_dot = float(np.dot(train_direction, test_direction))

    if exact:
        if len(train_scenario_deltas) > 20:
            raise ValueError("Exact sign enumeration is limited to 20 scenarios")
        sign_rows = product((-1.0, 1.0), repeat=len(train_scenario_deltas))
        effective_permutations = 2 ** len(train_scenario_deltas)
    else:
        sign_rows = (
            rng.choice((-1.0, 1.0), size=len(train_scenario_deltas))
            for _ in range(permutations)
        )
        effective_permutations = permutations

    null_cross_dots = []
    for sign_row in sign_rows:
        signs = np.asarray(sign_row)
        permuted_direction = (train_scenario_deltas * signs[:, None]).mean(axis=0)
        null_cross_dots.append(float(np.dot(permuted_direction, test_direction)))
    null = np.asarray(null_cross_dots)
    tolerance = max(1e-12, abs(observed_cross_dot) * 1e-7)
    exceedances = int(np.sum(null >= observed_cross_dot - tolerance))
    return {
        "permutations": effective_permutations,
        "permutation_mode": "exact_scenario_sign_enumeration" if exact else "monte_carlo",
        "statistic": "scenario_clustered_unnormalized_train_test_direction_dot_product",
        "train_scenario_count": int(len(train_scenario_deltas)),
        "test_scenario_count": int(len(test_scenario_deltas)),
        "observed_cross_dot": observed_cross_dot,
        "comparison_tolerance": tolerance,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "p_cross_dot_greater_equal": float(
            exceedances / effective_permutations
            if exact
            else (1 + exceedances) / (effective_permutations + 1)
        ),
        "null_q95": float(np.quantile(null, 0.95)),
    }


def bootstrap_mean_ci(
    values: list[float], seed: int, samples: int = 5000
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [rng.choice(array, size=array.size, replace=True).mean() for _ in range(samples)]
    )
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
