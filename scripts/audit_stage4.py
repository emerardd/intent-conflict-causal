"""Independent read-only Stage 4 audit. Python >=3.11 + NumPy; no GPU needed."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalize(v):
    return v/np.linalg.norm(v) if np.linalg.norm(v) > 1e-12 else np.zeros_like(v)


def audit(precision):
    run = ROOT / "results/stage4" / precision
    freeze, base, data = (read(run/name) for name in ("freeze.json", "baseline.json", "interventions.json"))
    config = freeze["config"]
    reference = ROOT / config["reference"]
    for name, digest in freeze["source_sha256"].items():
        assert sha(run/"source_snapshot"/name) == digest, name
    for name, digest in freeze["reference_sha256"].items():
        assert sha(reference/name) == digest, name
    prompts, provenance = read(run/"prompts.json"), read(run/"provenance.json")
    rows, patches = base["rows"], data["rows"]
    expected_n = 48 if precision == "nf4" else 8
    assert len(rows) == len(prompts) == expected_n
    assert len(patches) == expected_n*2*5
    assert provenance["revision"] == config["revision"]
    assert freeze["recipient_ids"] == [r["example_id"] for r in rows]
    old = read(reference/"baseline.json")["rows"]
    old_by_id = {r["example_id"]: r for r in old}
    by_id = {r["example_id"]: r for r in rows}
    idx = {r["example_id"]: i for i,r in enumerate(rows)}
    pairs = freeze["donor_indices"]
    activations = np.load(run/"activations.npz", allow_pickle=False)
    x = activations["activations"]
    assert x.shape == (expected_n,2,2560) and np.isfinite(x).all()
    assert activations["example_ids"].tolist() == freeze["recipient_ids"]
    assert activations["positions"].tolist() == config["positions"]
    for i,row in enumerate(rows):
        assert row["example_id"] == prompts[i]["example_id"]
        assert row["expected_label"] == "AB"[(int(not row["authorized"])+int(row["reversed_mapping"]))%2]
        donor = rows[pairs[i]]
        assert row["authorized"] != donor["authorized"]
        for name in ("scenario_id", "candidate_command", "grammar", "order", "reversed_mapping"):
            assert row[name] == donor[name] == old_by_id[row["example_id"]][name]
        assert pairs[pairs[i]] == i
        assert len(prompts[i]["ids"]) == len(prompts[pairs[i]]["ids"])
        assert prompts[i]["positions"] == prompts[pairs[i]]["positions"]
        if precision == "nf4":
            assert abs(row["margin"]-old_by_id[row["example_id"]]["margin"]) < 1e-4
    for row in rows+patches:
        assert abs(row["margin"]-(row["logp_B"]-row["logp_A"])) < 1e-12
        assert row["ab_label"] == ("B" if row["margin"] >= 0 else "A")
        expected_label = "A" if row["top1_id"] == provenance["A_id"] else "B" if row["top1_id"] == provenance["B_id"] else "OTHER"
        assert expected_label == row["top1_label"]
    accuracies = {str(m): np.mean([r["top1_label"] == r["expected_label"] for r in rows if r["reversed_mapping"] == m]) for m in (False,True)}
    assert accuracies == data["mapping_accuracy"] and min(accuracies.values()) >= config["baseline_min_accuracy"]
    axes = np.load(run/"axes.npz",allow_pickle=False)
    old_x = np.load(reference/"activations.npz",allow_pickle=False)
    train = np.array([r["split"] == "train" for r in old])
    y, t = np.array([not r["authorized"] for r in old]), np.array([r["expected_label"] == "B" for r in old])
    for position in config["positions"]:
        h = old_x["activations"][:,old_x["positions"].tolist().index(position),old_x["layers"].tolist().index(16)].astype(float)
        auth = h[train&y].mean(0)-h[train&~y].mean(0)
        token = h[train&t].mean(0)-h[train&~t].mean(0)
        if np.linalg.norm(token) > max(1e-8,np.linalg.norm(auth)*1e-5):
            dt = normalize(token)
            auth = auth-(auth@dt)*dt
        np.testing.assert_allclose(axes[position],normalize(auth),atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(axes[position+"_random"]),1,atol=1e-12)
        np.testing.assert_allclose(axes[position]@axes[position+"_random"],0,atol=1e-12)

    groups, unique = defaultdict(list), set()
    for row in patches:
        i, position = idx[row["example_id"]], row["position"]
        key = (row["example_id"],position,row["mode"])
        assert key not in unique
        unique.add(key)
        assert row["donor_id"] == rows[pairs[i]]["example_id"]
        p = config["positions"].index(position)
        delta = x[pairs[i],p].astype(float)-x[i,p].astype(float)
        d = axes[position]
        c = float(delta@d)
        expected_norm = {"full":np.linalg.norm(delta),"parallel":abs(c),
            "perpendicular":np.linalg.norm(delta-c*d),"random_full":np.linalg.norm(delta),"random_parallel":abs(c)}[row["mode"]]
        np.testing.assert_allclose(row["requested_delta_norm"],expected_norm,atol=1e-9)
        assert row["delivered_delta_norm"] > 0 if expected_norm > 0.01 else row["delivered_delta_norm"] >= 0
        if row["mode"] == "full":
            np.testing.assert_allclose(row["delivered_delta_norm"],expected_norm,rtol=1e-6,atol=1e-7)
        donor,b = by_id[row["donor_id"]],by_id[row["example_id"]]
        sign = 1 if (not donor["authorized"]) != donor["reversed_mapping"] else -1
        groups[position,row["mode"]].append((row,sign*(row["margin"]-b["margin"]),sign*(donor["margin"]-b["margin"])))

    scenario_effects = {}
    def bootstrap(values):
        if len(values) < 4:
            return None
        rng = np.random.default_rng(config["seed"]+1)
        return np.quantile(rng.choice(values,(config["bootstrap_samples"],len(values))).mean(1),[.025,.975])
    for (position,mode),group in groups.items():
        s = data["summary"]["effects"][f"{position}/{mode}"]
        per_scene = defaultdict(list)
        for row,effect,gap in group:
            per_scene[by_id[row["example_id"]]["scenario_id"]].append(effect)
        means = {sid:np.mean(values) for sid,values in per_scene.items()}
        scenario_effects[position,mode] = means
        np.testing.assert_allclose(np.mean(list(means.values())),s["mean"],atol=1e-12)
        interval = bootstrap(list(means.values()))
        if interval is None:
            assert s["ci95"] is None
        else:
            np.testing.assert_allclose(interval,s["ci95"],atol=1e-12)
        for m in (False,True):
            np.testing.assert_allclose(np.mean([e for r,e,g in group if by_id[r["example_id"]]["reversed_mapping"] == m]),s["mapping_means"][str(m)],atol=1e-12)
        assert s["global_top1_flips"] == sum(r["top1_id"] != by_id[r["example_id"]]["top1_id"] for r,e,g in group)
        assert s["flips_toward_donor"] == sum(r["top1_id"] != by_id[r["example_id"]]["top1_id"] and r["top1_label"] == by_id[r["donor_id"]]["expected_label"] for r,e,g in group)
        assert s["invalid_first_tokens"] == sum(r["top1_label"] == "OTHER" for r,e,g in group)
        np.testing.assert_allclose(np.mean([e/g for r,e,g in group if g > 1e-6]),s["mean_gap_recovery"],atol=1e-12)
    for position in config["positions"]:
        for left,right in (("full","parallel"),("parallel","random_parallel"),("full","random_full"),("full","perpendicular")):
            differences = [v-scenario_effects[position,right][sid] for sid,v in scenario_effects[position,left].items()]
            s = data["summary"]["paired_comparisons"][f"{position}/{left}-minus-{right}"]
            np.testing.assert_allclose(np.mean(differences),s["mean"],atol=1e-12)
            interval = bootstrap(differences)
            if interval is None:
                assert s["ci95"] is None
            else:
                np.testing.assert_allclose(interval,s["ci95"],atol=1e-12)
    assert len(data["generation_audit"]) == 16 and data["generation_agreement"]
    patch_lookup = {(r["example_id"],r["position"],r["mode"]):r for r in patches}
    for g in data["generation_audit"]:
        assert g["generated_ids"][0] == patch_lookup[g["example_id"],g["position"],g["mode"]]["top1_id"] and g["matches_forward"]
    assert len(data["null_checks"]) == 2
    for null in data["null_checks"]:
        assert abs(null["margin"]-rows[0]["margin"]) < 1e-4 and null["delivered_delta_norm"] == 0
    integrity = read(run/"manifest.json")
    for name,entry in integrity["files"].items():
        assert sha(run/name) == entry["sha256"] and (run/name).stat().st_size == entry["bytes"]
    return {"precision":precision,"reference_and_snapshot_hashes":True,"donor_identity_alignment":True,
        "train_only_fixed_axes":True,"decomposition_norms_delivery":True,"raw_statistics_and_cluster_ci":True,
        "paired_comparisons":True,"generation_and_null_checks":True,"artifact_manifest":True,
        "baseline_n":len(rows),"intervention_n":len(patches),"mapping_accuracy":accuracies,
        "scope":"Independent artifact/statistical recomputation, not independent model replication."}


def main():
    result = {p:audit(p) for p in ("nf4","bf16") if (ROOT/"results/stage4"/p/"interventions.json").exists()}
    if not result:
        raise RuntimeError("No completed Stage 4 precision run to audit")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
