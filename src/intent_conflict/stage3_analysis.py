"""Train-only factorial contrasts and scenario-level uncertainty; no model I/O."""
from __future__ import annotations

from math import comb
import numpy as np
from sklearn.metrics import roc_auc_score


def target(rows: list[dict], name: str) -> np.ndarray:
    if name == "auth":
        return np.asarray([not r["authorized"] for r in rows], bool)
    if name == "action":
        return np.asarray([r["assigned_block"] for r in rows], bool)
    if name == "token":
        return np.asarray([r["expected_label"] == "B" for r in rows], bool)
    raise ValueError(name)


def mask_for(rows: list[dict], **criteria) -> np.ndarray:
    return np.asarray([all(r[k] == v for k, v in criteria.items()) for r in rows], bool)


def contrast(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not (mask & y).any() or not (mask & ~y).any():
        raise ValueError("Contrast requires both labels")
    return x[mask & y].astype(np.float64).mean(0) - x[mask & ~y].astype(np.float64).mean(0)


def unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else np.zeros_like(v)


def residualize(v: np.ndarray, nuisances: list[np.ndarray]) -> tuple[np.ndarray, int]:
    kept = [n for n in nuisances if np.linalg.norm(n) > max(1e-8, np.linalg.norm(v)*1e-5)]
    if not kept:
        return v.copy(), 0
    _, singular, vh = np.linalg.svd(np.stack([unit(n) for n in kept]), full_matrices=False)
    basis = vh[singular > singular[0]*1e-6]
    return v - (basis @ v) @ basis, len(basis)


def auc(y: np.ndarray, scores: np.ndarray, mask: np.ndarray) -> float:
    if len(np.unique(y[mask])) != 2:
        raise ValueError("AUROC subset lacks both labels")
    return float(roc_auc_score(y[mask], scores[mask]))


def behavior_summary(rows: list[dict]) -> dict:
    result = {}
    for split in dict.fromkeys(r["split"] for r in rows):
        for grammar in dict.fromkeys(r["grammar"] for r in rows if r["split"] == split):
            selected = [r for r in rows if r["split"] == split and r["grammar"] == grammar]
            cells = {}
            for policy in (False, True):
                for mapping in (False, True):
                    sub = [r for r in selected if r["reversed_policy"] == policy and r["reversed_mapping"] == mapping]
                    if not sub:
                        continue
                    cells[f"policy{int(policy)}_map{int(mapping)}"] = {
                        "n": len(sub),
                        "global_top1_accuracy": float(np.mean([r["top1_label"] == r["expected_label"] for r in sub])),
                        "ab_accuracy": float(np.mean([r["ab_label"] == r["expected_label"] for r in sub])),
                        "format_rate": float(np.mean([r["top1_label"] in {"A", "B"} for r in sub])),
                        "mean_ab_mass": float(np.mean([r["ab_mass"] for r in sub])),
                    }
            result[f"{split}/{grammar}"] = {"n": len(selected), "cells": cells,
                "min_cell_top1_accuracy": min(c["global_top1_accuracy"] for c in cells.values())}
    generated = [r for r in rows if "generated_text" in r]
    result["generation"] = {
        "n": len(generated),
        "exact_accuracy": float(np.mean([r["generated_text"].strip() == r["expected_label"] for r in generated])) if generated else None,
        "first_token_agreement": all(r["generated_ids"][0] == r["top1_id"] for r in generated),
        "errors": [r["example_id"] for r in generated if r["generated_text"].strip() != r["expected_label"]],
    }
    return result


def cell_aucs(rows: list[dict], scores: np.ndarray, y: np.ndarray, split: str, grammar: str = "seen") -> dict:
    return {f"policy{int(p)}_map{int(m)}": auc(y, scores, mask_for(rows, split=split, grammar=grammar, reversed_policy=p, reversed_mapping=m))
            for p in (False, True) for m in (False, True)}


def evaluate_site(rows: list[dict], x: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
    train = mask_for(rows, split="train", grammar="seen")
    ya, yaction, yt = (target(rows, n) for n in ("auth", "action", "token"))
    raw = {n: contrast(x, y, train) for n, y in zip(("auth", "action", "token"), (ya, yaction, yt))}
    clean, rank = residualize(raw["auth"], [raw["action"], raw["token"]])
    directions = {n: unit(v) for n, v in raw.items()}
    directions["auth_clean"] = unit(clean)
    scores = x @ directions["auth_clean"]
    validation_cells = cell_aucs(rows, scores, ya, "validation")
    cross = {}
    for field in ("assigned_block", "reversed_mapping", "reversed_policy"):
        for source in (False, True):
            source_mask = train & mask_for(rows, **{field: source})
            fitted = unit(contrast(x, ya, source_mask))
            transfer_scores = x @ fitted
            cross[f"{field}_{int(source)}_to_{int(not source)}"] = {
                f"{split}/{grammar}": auc(ya, transfer_scores, mask_for(rows, split=split, grammar=grammar, **{field: not source}))
                for split, grammar in (("validation", "seen"), ("test", "seen"), ("test", "unseen"))
            }
    score_parts = list(validation_cells.values()) + [cross[f"assigned_block_{i}_to_{1-i}"]["validation/seen"] for i in (0, 1)]
    report = {
        "selection_score": min(score_parts),
        "validation_cells": validation_cells,
        "test_seen_cells": cell_aucs(rows, scores, ya, "test", "seen"),
        "test_unseen_cells": cell_aucs(rows, scores, ya, "test", "unseen"),
        "test_seen_by_assigned_action": {str(v): auc(ya, scores, mask_for(rows, split="test", grammar="seen", assigned_block=v)) for v in (False, True)},
        "overall_auroc": {f"{split}/{grammar}": auc(ya, scores, mask_for(rows, split=split, grammar=grammar))
                          for split, grammar in (("train", "seen"), ("validation", "seen"), ("test", "seen"), ("test", "unseen"))},
        "cross_transfer": cross,
        "raw_norms": {n: float(np.linalg.norm(v)) for n, v in raw.items()},
        "nuisance_rank": rank,
        "auth_retained_norm_fraction": float(np.linalg.norm(clean) / max(np.linalg.norm(raw["auth"]), 1e-12)),
        "train_projection_gap": float(np.dot(raw["auth"], directions["auth_clean"])),
        "raw_cosines": {n: float(directions["auth"] @ directions[n]) for n in ("action", "token")},
    }
    return report, directions


def select_site(site_reports: list[dict], primary_position: str) -> dict:
    candidates = [r for r in site_reports if r["position"] == primary_position]
    return max(candidates, key=lambda r: (r["selection_score"], -r["layer"]))


def bootstrap_ci(values: list[float], seed: int, samples: int = 5000) -> list[float]:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(samples, len(x)), replace=True).mean(1)
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def projection_summary(rows: list[dict], scores: np.ndarray, seed: int, samples: int) -> dict:
    reports = {}
    for grammar in ("seen", "unseen"):
        means = []
        for sid in dict.fromkeys(r["scenario_id"] for r in rows if r["split"] == "test"):
            m = mask_for(rows, split="test", scenario_id=sid, grammar=grammar)
            y = target(rows, "auth")
            means.append(float(scores[m & y].mean() - scores[m & ~y].mean()))
        positive = sum(v > 0 for v in means)
        n = len(means)
        reports[grammar] = {"scenario_effects": means, "n_scenarios": n, "positive_scenarios": positive,
            "mean": float(np.mean(means)), "ci95": bootstrap_ci(means, seed, samples),
            "one_sided_sign_p": sum(comb(n, k) for k in range(positive, n+1))/2**n}
    return reports


def summarize_interventions(rows: list[dict], baseline: dict[str, dict], seed: int, samples: int) -> dict:
    grouped = {}
    for r in rows:
        grouped.setdefault((r["axis"], r["alpha"], r["example_id"]), {})[r["sign"]] = r
    effects = {}
    for (axis, alpha, eid), pair in grouped.items():
        if set(pair) != {-1, 1}:
            raise ValueError("Intervention is missing its antipodal pair")
        from .stage3_data import unauthorized_margin_sign
        effect = 0.5*(pair[1]["margin"]-pair[-1]["margin"])*unauthorized_margin_sign(baseline[eid])
        effects.setdefault(f"{axis}@{alpha:g}", []).append((eid, float(effect)))
    summaries = {}
    scenario_effects = {}
    for key, pairs in effects.items():
        per_scenario = {}
        for eid, effect in pairs:
            per_scenario.setdefault(baseline[eid]["scenario_id"], []).append(effect)
        per_scenario = {sid: float(np.mean(v)) for sid, v in per_scenario.items()}
        scenario_effects[key] = per_scenario
        cells = {}
        for p in (False, True):
            for m in (False, True):
                sub = [v for eid, v in pairs if baseline[eid]["reversed_policy"] == p and baseline[eid]["reversed_mapping"] == m]
                if sub:
                    cells[f"policy{int(p)}_map{int(m)}"] = float(np.mean(sub))
        selected = [r for r in rows if f"{r['axis']}@{r['alpha']:g}" == key]
        summaries[key] = {"mean": float(np.mean(list(per_scenario.values()))),
            "ci95": bootstrap_ci(list(per_scenario.values()), seed, samples),
            "n_scenarios": len(per_scenario), "cell_means": cells,
            "scenario_effects": per_scenario,
            "global_top1_flips": sum(r["top1_id"] != baseline[r["example_id"]]["top1_id"] for r in selected),
            "ab_flips": sum(r["ab_label"] != baseline[r["example_id"]]["ab_label"] for r in selected),
            "invalid_first_tokens": sum(r["top1_label"] not in {"A", "B"} for r in selected), "n_forwards": len(selected)}
    main = summaries["auth_clean@1"]
    comparisons = {}
    for key in summaries:
        if key.startswith("random"):
            diffs = [v - scenario_effects[key][sid] for sid, v in scenario_effects["auth_clean@1"].items()]
            comparisons[key] = {"mean": float(np.mean(diffs)), "ci95": bootstrap_ci(diffs, seed, samples)}
    supported = (main["ci95"][0] > 0 and min(main["cell_means"].values()) > 0 and
                 all(v["ci95"][0] > 0 for v in comparisons.values()))
    return {"axes": summaries, "main_minus_random": comparisons,
            "conditional_rule_consistent_effect": bool(supported),
            "estimand": "scenario mean of half (+alpha minus -alpha) B/A margin, aligned by policy and mapping"}
