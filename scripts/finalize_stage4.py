"""Export audited Stage 4 comparison and manifests, without modifying Stage 3."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from intent_conflict.stage3 import dump,manifest,now,read,sha
from audit_stage4 import audit


def matched_effects(precision, ids):
    run = ROOT/"results/stage4"/precision
    base = {r["example_id"]:r for r in read(run/"baseline.json")["rows"]}
    result = {}
    rows = read(run/"interventions.json")["rows"]
    for position in ("pre_mapping","answer"):
        for mode in ("full","parallel","perpendicular","random_full","random_parallel"):
            chosen = [r for r in rows if r["example_id"] in ids and r["position"] == position and r["mode"] == mode]
            effects = [(1 if base[r["donor_id"]]["expected_label"] == "B" else -1)*(r["margin"]-base[r["example_id"]]["margin"]) for r in chosen]
            result[f"{position}/{mode}"] = {"mean":float(np.mean(effects)),"n":len(chosen),
                "flips":sum(r["top1_id"] != base[r["example_id"]]["top1_id"] for r in chosen),
                "ci95":None,"scope":"Two already-observed scenario clusters; descriptive only."}
    return result


def main():
    output = ROOT/"results/stage4"
    audits = {p:audit(p) for p in ("nf4","bf16")}
    ids = read(output/"bf16/freeze.json")["recipient_ids"]
    dump(output/"audit.json",{"created_at_utc":now(),"runs":audits})
    dump(output/"matched_precision_comparison.json",{"recipient_ids":ids,
        "nf4":matched_effects("nf4",ids),"bf16":matched_effects("bf16",ids),
        "scope":"Same eight recipients, same fixed NF4-derived direction axes, precision-specific baseline and donor states. Not a full precision replication."})
    sources = [ROOT/"src/intent_conflict/stage4.py",ROOT/"tests/test_stage4.py"]
    sources += list((ROOT/"configs").glob("stage4*.json"))
    sources += list((ROOT/"docs").glob("stage4*.md"))
    sources += [ROOT/"scripts/audit_stage4.py",Path(__file__)]
    dump(output/"delivery.json",{"created_at_utc":now(),"status":"both precision diagnostics completed and audited",
        "source_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in sources},
        "scope":"Exploratory diagnostic; no model training, downloads, Git commit or push; Stage 3 untouched."})
    manifest(output)
    print(output/"audit.json")
    print(output/"matched_precision_comparison.json")
    print(output/"manifest.json")


if __name__ == "__main__":
    main()
