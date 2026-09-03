import numpy as np
import pytest
from intent_conflict.stage4 import decompose, pair_indices, summarize


def test_decomposition_and_antipodal_controls():
    a, b = np.array([1.,2.,3.]), np.array([3.,5.,7.])
    d, r = np.array([1.,0.,0.]), np.array([0.,1.,0.])
    out = decompose(a,b,d,r)
    np.testing.assert_allclose(out["parallel"]+out["perpendicular"], out["full"])
    assert out["perpendicular"] @ d == 0
    assert np.linalg.norm(out["random_full"]) == np.linalg.norm(out["full"])
    assert np.linalg.norm(out["random_parallel"]) == np.linalg.norm(out["parallel"])
    reverse = decompose(b,a,d,r)
    for key in out:
        np.testing.assert_allclose(reverse[key], -out[key])


def test_pairing_requires_exact_candidate():
    base = dict(scenario_id="s", grammar="seen", order="first", reversed_mapping=False, candidate_command="same")
    rows = [dict(base,authorized=True), dict(base,authorized=False)]
    assert pair_indices(rows) == [1,0]
    rows[1]["candidate_command"] = "different"
    with pytest.raises(ValueError, match="Candidate mismatch"):
        pair_indices(rows)


def test_summary_donor_alignment_and_low_cluster_boundary():
    rows, patches = [], []
    modes = ("full", "parallel", "perpendicular", "random_full", "random_parallel")
    for sid in ("one", "two"):
        for m in (False,True):
            for auth in (False,True):
                label = "B" if (not auth) != m else "A"
                eid = f"{sid}_{m}_{auth}"
                rows.append(dict(example_id=eid,scenario_id=sid,reversed_mapping=m,authorized=auth,
                    expected_label=label,top1_id=32 if label == "A" else 33,margin=-2 if label == "A" else 2))
                for mode in modes:
                    patches.append(dict(example_id=eid,donor_id=f"{sid}_{m}_{not auth}",position="p",mode=mode,
                        margin=2 if label == "A" else -2,top1_label="B" if label == "A" else "A",top1_id=33 if label == "A" else 32))
    result = summarize(patches,rows,100,42)
    assert result["effects"]["p/full"]["mean"] == 4
    assert result["effects"]["p/full"]["mean_gap_recovery"] == 1
    assert result["effects"]["p/full"]["flips_toward_donor"] == 8
    assert result["effects"]["p/full"]["ci95"] is None
    assert result["paired_comparisons"]["p/full-minus-parallel"]["mean"] == 0
