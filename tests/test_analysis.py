import numpy as np

from intent_conflict.analysis import (
    ScenarioSplit,
    behavior_gate,
    behavior_summary,
    permutation_test,
    split_scenarios,
)
from intent_conflict.data import build_examples


def test_behavior_summary_and_gate_on_perfect_margins() -> None:
    examples = build_examples(max_scenarios=4)
    margins = np.asarray([-2.0 if example.authorized else 2.0 for example in examples])
    summary = behavior_summary(examples, margins)
    gate = behavior_gate(summary)
    assert summary["authorized_accuracy"] == 1.0
    assert summary["unauthorized_accuracy"] == 1.0
    assert summary["paired_effect_mean"] == 4.0
    assert gate["passed"]


def test_scenario_split_has_no_overlap() -> None:
    examples = build_examples(max_scenarios=6)
    split = split_scenarios(examples, seed=3, train_count=3, validation_count=1)
    assert set(split.train).isdisjoint(split.validation)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.validation).isdisjoint(split.test)


def test_permutation_preserves_effect_magnitude() -> None:
    examples = build_examples(max_scenarios=13)
    scenario_ids = sorted({item.scenario_id for item in examples})
    split = ScenarioSplit(
        train=tuple(scenario_ids[:8]),
        validation=(scenario_ids[8],),
        test=tuple(scenario_ids[9:]),
    )
    activations = []
    for example in examples:
        base = float(scenario_ids.index(example.scenario_id))
        authorization_signal = 3.0 if not example.authorized else -3.0
        activations.append([authorization_signal, base, 0.1 * authorization_signal])
    result = permutation_test(
        examples,
        np.asarray(activations),
        split,
        permutations=1000,
        seed=17,
    )
    assert result["observed_cross_dot"] > result["null_q95"]
    assert result["p_cross_dot_greater_equal"] < 0.05


def test_exact_permutation_enumerates_scenario_signs() -> None:
    examples = build_examples(max_scenarios=8)
    scenario_ids = sorted({item.scenario_id for item in examples})
    split = ScenarioSplit(
        train=tuple(scenario_ids[:5]),
        validation=(scenario_ids[5],),
        test=tuple(scenario_ids[6:]),
    )
    activations = np.asarray(
        [
            [2.0 if not example.authorized else -2.0, 1.0]
            for example in examples
        ]
    )
    result = permutation_test(
        examples,
        activations,
        split,
        permutations=3,
        seed=0,
        exact=True,
    )
    assert result["permutations"] == 2**5
    assert result["permutation_mode"] == "exact_scenario_sign_enumeration"
    assert result["p_cross_dot_greater_equal"] == 1 / 2**5
    assert result["p_cross_dot_greater_equal"] > 0
