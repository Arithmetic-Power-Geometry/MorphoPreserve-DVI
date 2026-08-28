from pathlib import Path
from morphopreserve.analysis import audit, feature_ablation, load_records, qc_records, sensitivity_analysis

DATA = Path(__file__).parents[1] / "data" / "ranchi_nasal_morphometry.csv"

def test_dataset_counts():
    records = load_records(DATA)
    assert len(records) == 200
    assert sum(x.sex == "M" for x in records) == 100
    assert sum(x.sex == "F" for x in records) == 100


def test_audit_is_reproducible():
    a = audit(load_records(DATA))
    assert a["ni_abs_diff_gt_0_5"] == 32
    assert a["ni_abs_diff_gt_2"] == 20
    assert a["ni_abs_diff_gt_5"] == 8
    assert a["nasal_type_mismatches"] == 19
    assert len(a["technical_plausibility_exclusions"]) == 1


def test_qc_count():
    assert len(qc_records(load_records(DATA))) == 199


def test_sensitivity_acl_positive():
    rows = sensitivity_analysis(load_records(DATA), repeats=2, splits=5, bootstrap=200)
    assert len(rows) == 3
    assert all(row["acl_auc"] > 0.10 for row in rows)


def test_primary_sensitivity_matches_ablation():
    records = load_records(DATA)
    primary = qc_records(records)
    ablation, _, _ = feature_ablation(primary, repeats=2, splits=5, bootstrap=100)
    by_name = {row["name"]: row for row in ablation}
    sens = sensitivity_analysis(records, repeats=2, splits=5, bootstrap=100)[0]
    assert abs(sens["auc_ni"] - by_name["NI only"]["auc"]) < 1e-12
    assert abs(sens["auc_nw_nh"] - by_name["NW + NH"]["auc"]) < 1e-12
    assert abs(sens["acl_auc"] - (by_name["NW + NH"]["auc"] - by_name["NI only"]["auc"])) < 1e-12
