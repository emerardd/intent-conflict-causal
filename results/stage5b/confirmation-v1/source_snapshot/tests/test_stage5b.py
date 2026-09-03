import numpy as np

from intent_conflict.stage5_data import build_confirmation
from intent_conflict.stage5b import derive_common_scale, equal_norm_vector, random_axis_banks
from intent_conflict.stage5b_data import build_stage5b


def test_stage5b_scenarios_are_new_balanced_and_unique():
    rows = build_stage5b()
    assert len(rows) == 48
    assert len({row.example_id for row in rows}) == 48
    assert len({row.scenario_id for row in rows}) == 12
    assert {row.scenario_id for row in rows}.isdisjoint(
        {row.scenario_id for row in build_confirmation()}
    )
    for scenario in {row.scenario_id for row in rows}:
        group = [row for row in rows if row.scenario_id == scenario]
        assert {(row.authorized, row.reversed_mapping) for row in group} == {
            (True, False), (True, True), (False, False), (False, True)
        }


def test_common_scale_is_minimum_of_position_medians():
    rows = [
        {"split": "train", "scenario_id": "s0", "grammar": "g", "order": "o",
         "reversed_mapping": False, "authorized": True},
        {"split": "train", "scenario_id": "s0", "grammar": "g", "order": "o",
         "reversed_mapping": False, "authorized": False},
        {"split": "test", "scenario_id": "s1", "grammar": "g", "order": "o",
         "reversed_mapping": False, "authorized": True},
        {"split": "test", "scenario_id": "s1", "grammar": "g", "order": "o",
         "reversed_mapping": False, "authorized": False},
    ]
    activations = np.zeros((4, 2, 2))
    activations[1, 0, 0] = 2.0
    activations[1, 1, 1] = 3.0
    axes = {"pre_mapping": np.array([1.0, 0.0]), "answer": np.array([0.0, 1.0])}
    result = derive_common_scale(rows, activations, ["pre_mapping", "answer"], axes)
    assert result["position_medians"] == {"pre_mapping": 2.0, "answer": 3.0}
    assert result["common_scale"] == 2.0
    assert result["n_train_rows"] == 2


def test_random_banks_are_unit_and_orthogonal_to_real_axes():
    axes = {"pre_mapping": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
            "answer": np.array([0.0, 1.0, 0.0, 0.0, 0.0])}
    banks = random_axis_banks(axes, ["pre_mapping", "answer"], 2, 5)
    for bank in banks.values():
        assert np.allclose(bank @ bank.T, np.eye(2), atol=1e-12)
        assert np.allclose(bank @ axes["pre_mapping"], 0, atol=1e-12)
        assert np.allclose(bank @ axes["answer"], 0, atol=1e-12)


def test_equal_norm_vector_has_frozen_norm_and_sign():
    original = np.array([2.0, 3.0, 4.0])
    direction = np.array([0.0, 1.0, 0.0])
    plus = equal_norm_vector(original, direction, 0.6, True)
    minus = equal_norm_vector(original, direction, 0.6, False)
    assert np.isclose(np.linalg.norm(plus - original), 0.6)
    assert np.isclose(np.linalg.norm(minus - original), 0.6)
    assert np.allclose(plus - original, -(minus - original))
