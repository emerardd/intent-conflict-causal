import numpy as np

from intent_conflict.stage5 import build_components, pair_indices, summarize
from intent_conflict.stage5_data import build_confirmation


def test_confirmation_is_new_balanced_and_unique():
    rows = build_confirmation()
    assert len(rows) == 48
    assert len({r.scenario_id for r in rows}) == 12
    assert len({r.example_id for r in rows}) == 48
    assert len({r.candidate_code for r in rows}) == 12
    for sid in {r.scenario_id for r in rows}:
        group = [r for r in rows if r.scenario_id == sid]
        assert {(r.authorized, r.reversed_mapping) for r in group} == {
            (True, False), (True, True), (False, False), (False, True)
        }


def test_pairing_preserves_mapping_and_reverses_authorization():
    rows = [r.row() for r in build_confirmation()]
    pairs = pair_indices(rows)
    for i, j in enumerate(pairs):
        assert rows[i]["scenario_id"] == rows[j]["scenario_id"]
        assert rows[i]["reversed_mapping"] == rows[j]["reversed_mapping"]
        assert rows[i]["authorized"] != rows[j]["authorized"]


def test_components_are_norm_matched_and_full_is_exact():
    recipient = np.array([1.0, 2.0, 3.0, 4.0])
    donor = np.array([2.0, 0.0, 3.0, 8.0])
    direction = np.array([1.0, 0.0, 0.0, 0.0])
    random_axis = np.array([0.0, 1.0, 0.0, 0.0])
    c = build_components(recipient, donor, direction, random_axis, donor_unauthorized=True)
    assert np.allclose(recipient + c["full"], donor)
    assert np.isclose(np.linalg.norm(c["full"]), np.linalg.norm(c["random_full"]))
    assert np.isclose(np.linalg.norm(c["parallel"]), np.linalg.norm(c["random_parallel"]))


def test_summary_uses_scenario_clusters_and_donor_alignment():
    source = [row.row() for row in build_confirmation() if row.scenario_id in {
        "musicbox_cylinder", "shadowbox_layer", "kinetic_mobile", "stainedglass_panel"
    }]
    pairs = pair_indices(source)
    baselines = []
    interventions = []
    for index, row in enumerate(source):
        margin = 2.0 if row["expected_label"] == "B" else -2.0
        top1 = 2 if row["expected_label"] == "B" else 1
        baselines.append({**row, "margin": margin, "top1_id": top1,
                          "top1_label": row["expected_label"]})
        donor = source[pairs[index]]
        donor_sign = 1.0 if donor["expected_label"] == "B" else -1.0
        for target in ("pre_mapping", "answer"):
            for mode in ("full", "random_full"):
                interventions.append({"example_id": row["example_id"],
                                      "donor_id": donor["example_id"],
                                      "target_position": target, "source_axis": "none",
                                      "mode": mode, "margin": margin + donor_sign * 2.0,
                                      "top1_id": top1, "top1_label": row["expected_label"]})
            for axis in ("pre_mapping", "answer"):
                for mode in ("parallel", "random_parallel"):
                    interventions.append({"example_id": row["example_id"],
                                          "donor_id": donor["example_id"],
                                          "target_position": target, "source_axis": axis,
                                          "mode": mode, "margin": margin + donor_sign * 2.0,
                                          "top1_id": top1, "top1_label": row["expected_label"]})
    report = summarize(interventions, baselines, 200, 7, 0.1)
    effect = report["effects"]["answer/answer/parallel"]
    assert effect["n_scenarios"] == 4
    assert effect["mean"] == 2.0
