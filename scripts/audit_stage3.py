"""Read-only Stage 3 evidence audit, independent of the analysis implementation.

Needs Python >=3.11 and NumPy, but not torch, sklearn, a GPU, or model weights.
Run from the repository root: python scripts/audit_stage3.py
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from intent_conflict.stage3_audit import audit_screen


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unit(x):
    norm = np.linalg.norm(x)
    return x / norm if norm > 1e-12 else np.zeros_like(x)


def pairwise_auc(labels, scores):
    pos, neg = scores[labels], scores[~labels]
    assert len(pos) and len(neg)
    differences = pos[:, None] - neg[None, :]
    return float(((differences > 0) + 0.5 * (differences == 0)).mean())


def ci(values, config, offset):
    values = np.asarray(values)
    rng = np.random.default_rng(config["seed"] + offset)
    means = rng.choice(values, size=(config["bootstrap_samples"], len(values)), replace=True).mean(axis=1)
    return np.quantile(means, [0.025, 0.975])


def audit_formal(run):
    base, frozen, saved = (load(run / name) for name in ("baseline.json", "freeze.json", "analysis.json"))
    rows, config = base["rows"], base["config"]
    prompts, provenance = load(run / "prompts.json"), load(run / "provenance.json")
    assert len(rows) == len(prompts) == 288
    assert len({r["example_id"] for r in rows}) == 288
    assert sha(run / "prompts.json") == provenance["prompts_sha256"]
    assert base["config_sha256"] == frozen["config_sha256"]
    for name, digest in frozen["source_sha256"].items():
        assert sha(run / "source_snapshot" / name) == digest
    group_rows = defaultdict(list)
    generated = []
    for row, original, prompt in zip(rows, frozen["dataset_rows"], prompts, strict=True):
        assert all(row[k] == value for k, value in original.items())
        assert prompt["example_id"] == row["example_id"]
        assert not row["reversed_policy"]
        expected = "AB"[(int(not row["authorized"]) + int(row["reversed_mapping"])) % 2]
        assert expected == row["expected_label"]
        aid, bid = provenance["A_id"], provenance["B_id"]
        label = "A" if row["top1_id"] == aid else "B" if row["top1_id"] == bid else "OTHER"
        assert label == row["top1_label"]
        assert row["top5"][0]["id"] == row["top1_id"]
        assert abs(row["logp_B"] - row["logp_A"] - row["margin"]) < 1e-12
        group_rows[row["split"], row["grammar"], row["reversed_mapping"]].append(row)
        if "generated_text" in row:
            generated.append(row)
    split_scenes = {s: {r["scenario_id"] for r in rows if r["split"] == s} for s in ("train", "validation", "test")}
    assert [len(v) for v in split_scenes.values()] == [8, 4, 12]
    assert sum(map(len, split_scenes.values())) == len(set.union(*split_scenes.values()))
    for split, grammar in (("train", "seen"), ("validation", "seen"), ("test", "seen"), ("test", "unseen")):
        counts = Counter((r["authorized"], r["reversed_mapping"], r["order"]) for r in rows if r["split"] == split and r["grammar"] == grammar)
        assert len(counts) == 8 and len(set(counts.values())) == 1
    behavior = {}
    for (split, grammar, mapping), group in group_rows.items():
        n = len(group)
        correct = sum(r["expected_label"] == r["top1_label"] for r in group)
        ab_correct = sum(r["expected_label"] == r["ab_label"] for r in group)
        valid = sum(r["top1_label"] != "OTHER" for r in group)
        actual = dict(n=n, global_top1_accuracy=correct/n, ab_accuracy=ab_correct/n,
                      format_rate=valid/n, mean_ab_mass=np.mean([r["ab_mass"] for r in group]))
        expected = saved["behavior"][f"{split}/{grammar}"]["cells"][f"policy0_map{int(mapping)}"]
        for key in actual:
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-12)
        behavior[f"{split}/{grammar}/map{int(mapping)}"] = dict(correct=correct, n=n)
    assert len(generated) == 32
    generation_correct = sum(r["generated_text"].strip() == r["expected_label"] for r in generated)
    agreement = all(r["generated_ids"][0] == r["top1_id"] for r in generated)
    assert saved["behavior"]["generation"]["exact_accuracy"] == generation_correct/32
    assert saved["behavior"]["generation"]["first_token_agreement"] == agreement

    archive = np.load(run / "activations.npz", allow_pickle=False)
    x = archive["activations"]
    assert x.shape == (288, 3, 7, 2560) and np.isfinite(x).all()
    assert archive["example_ids"].tolist() == [r["example_id"] for r in rows]
    labels = np.array([not r["authorized"] for r in rows])
    token = np.array([r["expected_label"] == "B" for r in rows])
    mapping = np.array([r["reversed_mapping"] for r in rows])

    def mask(split, grammar="seen"):
        return np.array([r["split"] == split and r["grammar"] == grammar for r in rows])

    train = mask("train")

    def difference(hidden, y, subset):
        return hidden[subset & y].astype(float).mean(0) - hidden[subset & ~y].astype(float).mean(0)

    selections, directions = [], {}
    for p, position in enumerate(config["positions"]):
        for l, layer in enumerate(config["layers"]):
            h = x[:, p, l]
            raw, nuisance = difference(h, labels, train), difference(h, token, train)
            clean = raw.copy()
            if np.linalg.norm(nuisance) > max(1e-8, np.linalg.norm(raw)*1e-5):
                dtoken = unit(nuisance)
                clean -= np.dot(clean, dtoken) * dtoken
            direction = unit(clean)
            scores = h @ direction
            report = next(s for s in saved["sites"] if s["position"] == position and s["layer"] == layer)
            vals = []
            for split, grammar in (("validation", "seen"), ("test", "seen"), ("test", "unseen")):
                for m in (False, True):
                    subset = mask(split, grammar) & (mapping == m)
                    a = pairwise_auc(labels[subset], scores[subset])
                    np.testing.assert_allclose(a, report["mapping_auroc"][f"{split}/{grammar}"][str(m)], atol=1e-12)
                    if split == "validation":
                        vals.append(a)
            for source in (False, True):
                fitted = unit(difference(h, labels, train & (mapping == source)))
                s = h @ fitted
                for split, grammar in (("validation", "seen"), ("test", "seen"), ("test", "unseen")):
                    subset = mask(split, grammar) & (mapping != source)
                    a = pairwise_auc(labels[subset], s[subset])
                    key = f"map{int(source)}_to_{int(not source)}"
                    np.testing.assert_allclose(a, report["cross_mapping"][key][f"{split}/{grammar}"], atol=1e-12)
                    if split == "validation":
                        vals.append(a)
            np.testing.assert_allclose(min(vals), report["selection_score"], atol=1e-12)
            if position == config["primary_position"]:
                selections.append((min(vals), -layer, layer))
            directions[position, layer] = direction
    selected = saved["selected"]
    assert max(selections)[2] == selected["layer"]
    min_accuracy = lambda split, grammar: min(v["correct"]/v["n"] for k, v in behavior.items() if k.startswith(f"{split}/{grammar}/"))
    gates = {
        "train_validation_behavior": min(min_accuracy(s, "seen") for s in ("train", "validation")) >= config["formal_min_cell_top1_accuracy"],
        "test_seen_behavior": min_accuracy("test", "seen") >= config["formal_min_cell_top1_accuracy"],
        "generation": generation_correct/32 >= config["generation_min_exact_accuracy"] and agreement,
        "validation_representation": selected["selection_score"] > config["validation_min_auroc"],
        "test_representation": min(selected["mapping_auroc"]["test/seen"].values()) > config["test_min_auroc"],
    }
    assert gates == saved["gates"] and all(gates.values()) == saved["intervention_eligible"]
    selected_archive = np.load(run / "selected_site.npz", allow_pickle=False)
    d = directions[selected["position"], selected["layer"]]
    np.testing.assert_allclose(d, selected_archive["auth_clean"], atol=1e-10)
    p, l = config["positions"].index(selected["position"]), config["layers"].index(selected["layer"])
    np.testing.assert_array_equal(x[:, p, l], selected_archive["activations"])
    scores = x[:, p, l] @ d
    for grammar in ("seen", "unseen"):
        effects = []
        for sid in dict.fromkeys(r["scenario_id"] for r in rows if r["split"] == "test"):
            subset = mask("test", grammar) & np.array([r["scenario_id"] == sid for r in rows])
            effects.append(scores[subset & labels].mean() - scores[subset & ~labels].mean())
        np.testing.assert_allclose(effects, saved["projection"][grammar]["scenario_effects"], atol=1e-9)
        np.testing.assert_allclose(ci(effects, config, 10), saved["projection"][grammar]["ci95"], atol=1e-9)
    result = {"frozen_sources_prompts_rows": True, "independent_behavior": behavior,
        "generation_correct": generation_correct, "generation_total": 32, "first_token_agreement": agreement,
        "pairwise_auc_all_21_sites": True, "validation_only_layer_selection": True,
        "independent_direction_and_clustered_projection_ci": True, "selected_layer": selected["layer"],
        "split_factorial_balance": True, "frozen_gates_recomputed": gates}
    if (run / "interventions.json").exists():
        result["interventions"] = audit_interventions(run, base, config)
    return result


def audit_interventions(run, base, config):
    data = load(run / "interventions.json")
    baseline = {r["example_id"]: r for r in base["rows"]}
    plan = load(run / "intervention_freeze.json")
    pairs = defaultdict(dict)
    for r in data["rows"]:
        key = (r["axis"], r["alpha"], r["example_id"])
        assert r["sign"] not in pairs[key]
        pairs[key][r["sign"]] = r
    assert len(plan["recipient_ids"]) == 48
    expected_forwards = 48 * (2*len(config["alphas"]) + 2*(len(plan["axes"])-1))
    assert len(data["rows"]) == expected_forwards
    directions = np.load(run / "intervention_axes.npz", allow_pickle=False)
    for axis in plan["axes"]:
        np.testing.assert_allclose(np.linalg.norm(directions[axis]), 1, atol=1e-12)
        if axis.startswith("random"):
            np.testing.assert_allclose(np.dot(directions[axis], directions["auth_clean"]), 0, atol=1e-12)
    index = {(r["example_id"], r["axis"], r["alpha"], r["sign"]): r for r in data["rows"]}
    assert len(data["generation_audit"]) == 8
    for g in data["generation_audit"]:
        expected = index[g["example_id"], "auth_clean", 1.0, g["sign"]]
        assert g["generated_ids"][0] == expected["top1_id"]
        assert g["matches_forward"]
    assert data["generation_agreement"]
    summaries, by_scenario = data["summary"]["axes"], {}
    for axis in plan["axes"]:
        for alpha in config["alphas"] if axis == "auth_clean" else [1.0]:
            key = f"{axis}@{alpha:g}"
            effects, mapping_cells = defaultdict(list), defaultdict(list)
            selected_rows = []
            for eid in plan["recipient_ids"]:
                pair = pairs[axis, alpha, eid]
                assert set(pair) == {-1, 1}
                b = baseline[eid]
                sign = -1 if b["reversed_mapping"] else 1
                effect = (pair[1]["margin"] - pair[-1]["margin"]) * sign / 2
                effects[b["scenario_id"]].append(effect)
                mapping_cells[f"policy0_map{int(b['reversed_mapping'])}"].append(effect)
                selected_rows.extend(pair.values())
            means = {sid: np.mean(v) for sid, v in effects.items()}
            by_scenario[key] = means
            s = summaries[key]
            np.testing.assert_allclose(np.mean(list(means.values())), s["mean"], atol=1e-12)
            np.testing.assert_allclose(ci(list(means.values()), config, 30), s["ci95"], atol=1e-12)
            for cell, values in mapping_cells.items():
                np.testing.assert_allclose(np.mean(values), s["cell_means"][cell], atol=1e-12)
            assert s["global_top1_flips"] == sum(r["top1_id"] != baseline[r["example_id"]]["top1_id"] for r in selected_rows)
            assert s["ab_flips"] == sum(r["ab_label"] != baseline[r["example_id"]]["ab_label"] for r in selected_rows)
            assert s["invalid_first_tokens"] == sum(r["top1_label"] not in {"A", "B"} for r in selected_rows)
    for key, values in by_scenario.items():
        if key.startswith("random"):
            diffs = [v - values[sid] for sid, v in by_scenario["auth_clean@1"].items()]
            np.testing.assert_allclose(ci(diffs, config, 30), data["summary"]["main_minus_random"][key]["ci95"], atol=1e-12)
    main = summaries["auth_clean@1"]
    supported = main["ci95"][0] > 0 and min(main["cell_means"].values()) > 0 and all(v["ci95"][0] > 0 for v in data["summary"]["main_minus_random"].values())
    assert bool(supported) == data["summary"]["conditional_rule_consistent_effect"]
    return {"paired_effects_and_scenario_ci": True, "flips_and_format_counts": True,
        "main_random_comparison": True, "decision_rule_recomputed": True,
        "unit_axes_random_orthogonality": True, "patched_generation_agreement": True,
        "n_forwards": len(data["rows"]), "conditional_effect_supported": bool(supported)}


def main():
    stage3 = ROOT / "results/stage3"
    result = {
        "v1": audit_screen(stage3 / "screen-v1", include_activations=True),
        "v2_pilot": audit_screen(stage3 / "remapping-v2/screen-v1", include_activations=True),
    }
    formal = stage3 / "remapping-v2/formal-v1"
    if (formal / "analysis.json").exists():
        result["v2_formal"] = audit_formal(formal)
    if (stage3 / "manifest.json").exists():
        recorded = load(stage3 / "manifest.json")
        for name, entry in recorded["files"].items():
            path = stage3 / name
            assert path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], name
        result["artifact_manifest_files_verified"] = len(recorded["files"])
    if (stage3 / "delivery.json").exists():
        delivery = load(stage3 / "delivery.json")
        for name, digest in delivery["current_delivery_source_sha256"].items():
            assert sha(ROOT / name) == digest, name
        result["current_delivery_source_hashes_verified"] = True
    result["scope"] = "Read-only recomputation from artifacts, NOT an independent model-inference replication."
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
