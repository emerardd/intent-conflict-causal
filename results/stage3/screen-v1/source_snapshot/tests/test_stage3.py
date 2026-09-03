from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from intent_conflict.data import STAGE2_NEUTRAL_V2_SCENARIOS
from intent_conflict.stage3_data import (SYSTEM, build_trials, expected_block,
    label_for_block, observed_block, render_trial, tokenize_trial, unauthorized_margin_sign)
from intent_conflict.stage3_analysis import (behavior_summary, evaluate_site, mask_for,
    residualize, select_site, summarize_interventions, target)
from intent_conflict.stage3_model import checked_patch


def test_counts_splits_codes_and_old_data_isolation():
    screen, formal = build_trials("screen"), build_trials("formal")
    assert len(screen) == 32 and len(formal) == 576
    assert len({r.example_id for r in formal}) == 576
    sets = {s: {r.scenario_id for r in formal if r.split == s} for s in ("train", "validation", "test")}
    assert [len(sets[s]) for s in sets] == [8, 4, 12]
    assert not sets["train"] & sets["validation"] and not sets["train"] & sets["test"]
    assert not sets["validation"] & sets["test"]
    assert not {r.scenario_id for r in screen} & {r.scenario_id for r in formal}
    assert not {s.scenario_id for s in STAGE2_NEUTRAL_V2_SCENARIOS} & {r.scenario_id for r in formal}
    code_pairs = {r.scenario_id: (r.candidate_code, r.distractor_code) for r in screen+formal}
    assert len({c for pair in code_pairs.values() for c in pair}) == 52
    assert all(r.split == "test" for r in formal if r.grammar == "unseen")


def test_factorial_cells_balanced_and_independent():
    rows = [r.row() for r in build_trials("formal")]
    for split, grammar in (("train", "seen"), ("validation", "seen"), ("test", "seen"), ("test", "unseen")):
        mask = mask_for(rows, split=split, grammar=grammar)
        labels = np.stack([target(rows, n)[mask] for n in ("auth", "action", "token")], 1)
        combos, counts = np.unique(labels, axis=0, return_counts=True)
        assert len(combos) == 8 and len(set(counts)) == 1
        np.testing.assert_allclose(np.corrcoef(labels.T), np.eye(3), atol=1e-12)


def test_expected_labels_and_causal_signs():
    for auth, policy, mapping in product((True, False), repeat=3):
        block = expected_block(auth, policy)
        label = label_for_block(block, mapping)
        assert observed_block(label, mapping) == block
        assert (not block) == (auth != policy)
        ua_label = label_for_block(expected_block(False, policy), mapping)
        sign = unauthorized_margin_sign({"reversed_policy": policy, "reversed_mapping": mapping})
        assert sign == (1 if ua_label == "B" else -1)
    assert observed_block("OTHER", False) is None


def test_tokenizer_accepts_mapping_native_chat_result():
    class Tokenizer:
        def apply_chat_template(self, messages, tokenize, **kwargs):
            text = "\n".join(m["content"] for m in messages) + "\nASSISTANT:"
            return {"input_ids": [ord(c) for c in text]} if tokenize else text

        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i+1) for i in range(len(text))]}

    ids, positions = tokenize_trial(Tokenizer(), build_trials("screen")[0])
    assert positions["pre_policy"] < positions["pre_mapping"] < positions["answer"] == len(ids)-1


def test_prompt_temporal_separation_and_candidate_invariance():
    trials = build_trials("screen")
    assert "A means" not in SYSTEM and "B means" not in SYSTEM
    assert "A =" not in SYSTEM and "B =" not in SYSTEM
    first = trials[0]
    same = [t for t in trials if t.scenario_id == first.scenario_id and t.order == first.order and t.authorized == first.authorized]
    assert len({t.facts for t in same}) == 1
    for trial in same:
        text = render_trial(trial)[1]["content"]
        before = text.split("[FACTS_END]")[0]
        assert "NORMAL" not in before and "REVERSED" not in before
        assert "A means" not in text.split("[RULE_END]")[0]
        assert text.index("[FACTS_END]") < text.index("[RULE_END]") < text.index("A means")
    assert len({t.candidate_command for t in trials if t.scenario_id == first.scenario_id}) == 1


def test_residualization_removes_nuisance_span():
    clean, rank = residualize(np.array([1., 3., 5.]), [np.array([0., 2., 0.]), np.array([0., 0., 4.])])
    np.testing.assert_allclose(clean, [1, 0, 0], atol=1e-12)
    assert rank == 2
    clean, rank = residualize(np.array([1., 0., 0.]), [np.zeros(3)])
    assert rank == 0 and clean[0] == 1


def test_analysis_recovers_factorial_signal_and_selection_ignores_test():
    rows = [t.row() for t in build_trials("formal")]
    x = np.stack([target(rows, n).astype(float)*2-1 for n in ("auth", "action", "token")], 1)
    metrics, directions = evaluate_site(rows, x)
    assert metrics["selection_score"] == 1
    np.testing.assert_allclose(directions["auth_clean"], [1, 0, 0], atol=1e-12)
    modified = x.copy()
    modified[mask_for(rows, split="test")] *= -1
    changed, _ = evaluate_site(rows, modified)
    assert changed["selection_score"] == metrics["selection_score"]
    assert changed["overall_auroc"]["test/seen"] == 0
    assert select_site([dict(metrics, position="pre_mapping", layer=16), dict(changed, position="pre_mapping", layer=4)], "pre_mapping")["layer"] == 4


def test_behavior_does_not_substitute_ab_preference_for_generated_action():
    rows = []
    for t in build_trials("screen"):
        r = t.row()
        r.update(top1_label="OTHER", top1_id=9, ab_label=t.expected_label, ab_mass=0.2,
                 generated_ids=[9], generated_text="Explanation")
        rows.append(r)
    report = behavior_summary(rows)
    assert report["screen/seen"]["min_cell_top1_accuracy"] == 0
    assert all(c["ab_accuracy"] == 1 for c in report["screen/seen"]["cells"].values())
    assert report["generation"]["exact_accuracy"] == 0


def test_checked_patch_position_identity_and_cleanup():
    layer = torch.nn.Identity()
    loaded = SimpleNamespace(layers=[layer])
    h = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    vector = np.array([20., 21., 22., 23.])
    with checked_patch(loaded, 0, 1, vector, 3) as count:
        out = layer(h)
    assert count == [1]
    np.testing.assert_array_equal(out[0, 1].numpy(), vector)
    torch.testing.assert_close(out[0, 0], h[0, 0])
    torch.testing.assert_close(layer(h), h)
    assert len(layer._forward_hooks) == 0
    with pytest.raises(RuntimeError, match="Expected one"):
        with checked_patch(loaded, 0, 1, vector, 3):
            pass
    assert len(layer._forward_hooks) == 0


def test_intervention_alignment_and_clustered_pairs():
    baseline = {t.example_id: dict(t.row(), top1_id=1, ab_label=t.expected_label) for t in build_trials("screen")}
    interventions = []
    for eid, row in baseline.items():
        for axis in ("auth_clean", "random0"):
            for sign in (-1, 1):
                interventions.append(dict(example_id=eid, axis=axis, alpha=1., sign=sign,
                    margin=sign*unauthorized_margin_sign(row)*(2 if axis == "auth_clean" else 0),
                    top1_id=1, top1_label="A", ab_label=row["expected_label"]))
    result = summarize_interventions(interventions, baseline, 1, 100)
    assert result["axes"]["auth_clean@1"]["mean"] == 2
    assert result["axes"]["auth_clean@1"]["n_scenarios"] == 2
    assert result["conditional_rule_consistent_effect"]
    with pytest.raises(ValueError):
        summarize_interventions(interventions[:-1], baseline, 1, 100)
