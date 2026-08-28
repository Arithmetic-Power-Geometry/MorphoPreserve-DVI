# Copyright (C) 2026 Mohammad Amir Khusru Akhtar
# Licensed under the Apache License, Version 2.0.
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 2026
DEFAULT_WIDTH_BOUNDS = (20.0, 70.0)
DEFAULT_HEIGHT_BOUNDS = (20.0, 70.0)


@dataclass(frozen=True)
class Record:
    subject_id: str
    sex: str
    width: float
    height: float
    ni_recorded: float
    type_recorded: str

    @property
    def ni(self) -> float:
        return 100.0 * self.width / self.height

    @property
    def type_calc(self) -> str:
        return "L" if self.ni < 70 else ("M" if self.ni < 85 else "P")


def load_records(path: str | Path) -> list[Record]:
    records: list[Record] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(
                Record(
                    subject_id=row["subject_id"],
                    sex=row["sex"].strip().upper(),
                    width=float(row["nasal_width_mm"]),
                    height=float(row["nasal_height_mm"]),
                    ni_recorded=float(row["nasal_index_recorded"]),
                    type_recorded=row["nasal_type_recorded"].strip().upper(),
                )
            )
    return records


def audit(records: list[Record], width_bounds=DEFAULT_WIDTH_BOUNDS, height_bounds=DEFAULT_HEIGHT_BOUNDS) -> dict:
    diffs = np.array([abs(r.ni_recorded - r.ni) for r in records])
    type_mismatch = sum(r.type_recorded != r.type_calc for r in records)
    wlo, whi = width_bounds
    hlo, hhi = height_bounds
    implausible = [
        r.subject_id
        for r in records
        if not (wlo <= r.width <= whi and hlo <= r.height <= hhi)
    ]
    return {
        "n": len(records),
        "male_n": sum(r.sex == "M" for r in records),
        "female_n": sum(r.sex == "F" for r in records),
        "ni_abs_diff_gt_0_5": int((diffs > 0.5).sum()),
        "ni_abs_diff_gt_2": int((diffs > 2).sum()),
        "ni_abs_diff_gt_5": int((diffs > 5).sum()),
        "max_abs_ni_difference": float(diffs.max()),
        "nasal_type_mismatches": int(type_mismatch),
        "technical_plausibility_exclusions": implausible,
    }


def qc_records(records: list[Record], width_bounds=DEFAULT_WIDTH_BOUNDS, height_bounds=DEFAULT_HEIGHT_BOUNDS) -> list[Record]:
    """Broad technical screen for obvious entry errors, not biological outlier trimming."""
    wlo, whi = width_bounds
    hlo, hhi = height_bounds
    return [r for r in records if wlo <= r.width <= whi and hlo <= r.height <= hhi]


def remove_subjects(records: list[Record], subject_ids: set[str]) -> list[Record]:
    return [r for r in records if r.subject_id not in subject_ids]


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    sp = math.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    d = (np.mean(a) - np.mean(b)) / sp
    return float(d * (1 - 3 / (4 * (n1 + n2) - 9)))


def descriptives(records: list[Record]) -> list[dict]:
    rows: list[dict] = []
    variables = [("width", "Nasal width (mm)"), ("height", "Nasal height (mm)"), ("ni", "Recomputed nasal index")]
    for attr, label in variables:
        male = np.array([getattr(r, attr) if attr != "ni" else r.ni for r in records if r.sex == "M"])
        female = np.array([getattr(r, attr) if attr != "ni" else r.ni for r in records if r.sex == "F"])
        t_stat, p_welch = stats.ttest_ind(male, female, equal_var=False)
        _, p_mw = stats.mannwhitneyu(male, female, alternative="two-sided")
        rows.append(
            {
                "variable": label,
                "male_n": len(male),
                "male_mean": float(male.mean()),
                "male_sd": float(male.std(ddof=1)),
                "female_n": len(female),
                "female_mean": float(female.mean()),
                "female_sd": float(female.std(ddof=1)),
                "mean_difference": float(male.mean() - female.mean()),
                "welch_t": float(t_stat),
                "welch_p": float(p_welch),
                "mann_whitney_p": float(p_mw),
                "hedges_g": hedges_g(male, female),
            }
        )
    return rows


def xy(records: list[Record]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[r.width, r.height, r.ni] for r in records], dtype=float)
    y = np.array([1 if r.sex == "M" else 0 for r in records], dtype=int)
    return X, y


def repeated_oof(X, y, estimator, repeats=10, splits=10, seed=SEED) -> np.ndarray:
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=seed)
    probability_sum = np.zeros(len(y))
    count = np.zeros(len(y), dtype=int)
    for train, test in cv.split(X, y):
        est = clone(estimator)
        est.fit(X[train], y[train])
        probability_sum[test] += est.predict_proba(X[test])[:, 1]
        count[test] += 1
    return probability_sum / count


def metric_row(name: str, y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "name": name,
        "auc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "brier": float(brier_score_loss(y, p)),
    }


def bootstrap_auc_ci(y, p, n=2000, seed=20260827) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    N = len(y)
    for _ in range(n):
        ix = rng.integers(0, N, N)
        if len(np.unique(y[ix])) == 2:
            values.append(roc_auc_score(y[ix], p[ix]))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def bootstrap_acl_ci(y, pfull, pni, n=3000, seed=20260827) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    N = len(y)
    for _ in range(n):
        ix = rng.integers(0, N, N)
        if len(np.unique(y[ix])) == 2:
            values.append(roc_auc_score(y[ix], pfull[ix]) - roc_auc_score(y[ix], pni[ix]))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _primary_logistic() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=5000, solver="liblinear", random_state=SEED)),
        ]
    )


def feature_ablation(records: list[Record], repeats=10, splits=10, bootstrap=2000) -> tuple[list[dict], dict[str, np.ndarray], np.ndarray]:
    X, y = xy(records)
    model = _primary_logistic()
    feature_sets = {
        "NI only": [2],
        "NW only": [0],
        "NH only": [1],
        "NW + NH": [0, 1],
        "NW + NH + NI": [0, 1, 2],
    }
    predictions: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    for name, idx in feature_sets.items():
        p = repeated_oof(X[:, idx], y, model, repeats=repeats, splits=splits)
        predictions[name] = p
        row = metric_row(name, y, p)
        lo, hi = bootstrap_auc_ci(y, p, n=bootstrap)
        row.update({"auc_95ci_low": lo, "auc_95ci_high": hi})
        rows.append(row)
    return rows, predictions, y


def sensitivity_analysis(records: list[Record], repeats=10, splits=10, bootstrap=2000) -> list[dict]:
    """Cleaning-rule sensitivity under the *same* model/CV design as the primary ablation.

    Only the included records change across scenarios. Estimator, folds, repeats and random
    seed are intentionally held fixed so the primary row exactly reproduces the manuscript
    NI and NW+NH AUCs. This makes the table a pure data-cleaning sensitivity analysis.
    """
    aud = audit(records)
    flagged = set(aud["technical_plausibility_exclusions"])
    scenarios = [
        ("Primary broad technical screen", qc_records(records)),
        ("Sensitivity A: retain all 200", list(records)),
        ("Sensitivity B: remove flagged record only", remove_subjects(records, flagged)),
    ]
    rows = []
    for i, (name, recs) in enumerate(scenarios):
        X, y = xy(recs)
        model = _primary_logistic()
        # The seed is fixed across scenarios by design; only the record set changes.
        p_ni = repeated_oof(X[:, [2]], y, model, repeats=repeats, splits=splits, seed=SEED)
        p_wh = repeated_oof(X[:, [0, 1]], y, model, repeats=repeats, splits=splits, seed=SEED)
        auc_ni = float(roc_auc_score(y, p_ni))
        auc_wh = float(roc_auc_score(y, p_wh))
        acl = auc_wh - auc_ni
        lo, hi = bootstrap_acl_ci(y, p_wh, p_ni, n=bootstrap, seed=20260827 + i)
        rows.append(
            {
                "scenario": name,
                "n": len(recs),
                "male_n": int(y.sum()),
                "female_n": int(len(y) - y.sum()),
                "auc_ni": auc_ni,
                "auc_nw_nh": auc_wh,
                "acl_auc": acl,
                "acl_95ci_low": lo,
                "acl_95ci_high": hi,
            }
        )
    return rows


def model_benchmark(records: list[Record], repeats=10, splits=10, bootstrap=2000) -> list[dict]:
    X, y = xy(records)
    models = {
        "Logistic regression": _primary_logistic(),
        "LDA": LinearDiscriminantAnalysis(),
        "Linear SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="linear", C=1, probability=True, random_state=SEED)),
            ]
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=3,
            random_state=SEED,
            class_weight="balanced",
        ),
        "Gradient boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=1,
            random_state=SEED,
        ),
    }
    rows = []
    for name, estimator in models.items():
        p = repeated_oof(X[:, :2], y, estimator, repeats=repeats, splits=splits)
        row = metric_row(name, y, p)
        lo, hi = bootstrap_auc_ci(y, p, n=bootstrap)
        row.update({"auc_95ci_low": lo, "auc_95ci_high": hi})
        rows.append(row)
    return rows


def selective_table(y: np.ndarray, p: np.ndarray, thresholds=(0.50, 0.60, 0.70, 0.75, 0.80, 0.85)) -> list[dict]:
    rows = []
    for tau in thresholds:
        mask = (p >= tau) | (p <= 1 - tau)
        if not mask.any():
            continue
        pred = (p[mask] >= 0.5).astype(int)
        rows.append(
            {
                "probability_threshold": tau,
                "classified_n": int(mask.sum()),
                "coverage": float(mask.mean()),
                "abstention_rate": float(1 - mask.mean()),
                "selective_accuracy": float(accuracy_score(y[mask], pred)),
                "selective_balanced_accuracy": float(balanced_accuracy_score(y[mask], pred)),
            }
        )
    return rows


def calibration_summary(y: np.ndarray, p: np.ndarray) -> list[dict]:
    eps = 1e-6
    clipped = np.clip(p, eps, 1 - eps)
    linear_predictor = np.log(clipped / (1 - clipped))
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000).fit(linear_predictor.reshape(-1, 1), y)
    return [
        {
            "calibration_intercept": float(calibrator.intercept_[0]),
            "calibration_slope": float(calibrator.coef_[0, 0]),
            "brier_score": float(brier_score_loss(y, p)),
        }
    ]


def write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _figures(outdir: Path, X: np.ndarray, y: np.ndarray, predictions: dict[str, np.ndarray], selective: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.linspace(28, 48, 100)
    ax.plot(x, x / 0.8, label="NI = 80")
    ax.scatter([32, 40], [40, 50], s=65)
    ax.annotate("A (32, 40)", (32, 40), xytext=(29, 45), arrowprops={"arrowstyle": "->"})
    ax.annotate("B (40, 50)", (40, 50), xytext=(41, 55), arrowprops={"arrowstyle": "->"})
    ax.set(xlabel="Nasal width", ylabel="Nasal height", title="Different primary dimensions can map to the same nasal index")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "fig1_ratio_information_loss.pdf")
    fig.savefig(outdir / "figures" / "fig1_ratio_information_loss.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    male = X[y == 1]
    female = X[y == 0]
    ax.scatter(male[:, 0], male[:, 1], alpha=0.65, label="Male")
    ax.scatter(female[:, 0], female[:, 1], alpha=0.65, label="Female")
    ax.set(xlabel="Nasal width (mm)", ylabel="Nasal height (mm)", title="Primary nasal dimensions in the audited Ranchi cohort")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "fig2_primary_dimensions.pdf")
    fig.savefig(outdir / "figures" / "fig2_primary_dimensions.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for name in ["NI only", "NW only", "NH only", "NW + NH"]:
        fpr, tpr, _ = roc_curve(y, predictions[name])
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, predictions[name]):.3f})")
    ax.plot([0, 1], [0, 1], "--", linewidth=1)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Information-preserving feature ablation")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "fig3_roc_ablation.pdf")
    fig.savefig(outdir / "figures" / "fig3_roc_ablation.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.plot([r["coverage"] for r in selective], [r["selective_accuracy"] for r in selective], marker="o")
    ax.set(xlabel="Coverage", ylabel="Accuracy among classified cases", title="Selective prediction: accuracy-coverage trade-off")
    ax.set_ylim(0.5, 1.0)
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "fig4_selective_prediction.pdf")
    fig.savefig(outdir / "figures" / "fig4_selective_prediction.png", dpi=240)
    plt.close(fig)

    p = predictions["NW + NH"]
    bins = np.linspace(0, 1, 6)
    ids = np.digitize(p, bins[1:-1], right=False)
    xp, yp = [], []
    for b in range(5):
        mask = ids == b
        if mask.sum() > 0:
            xp.append(float(p[mask].mean()))
            yp.append(float(y[mask].mean()))
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.plot([0, 1], [0, 1], "--", linewidth=1, label="Ideal")
    ax.plot(xp, yp, marker="o", label="NW + NH logistic")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed male proportion", title="Internal calibration of primary morphometry model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "fig5_calibration.pdf")
    fig.savefig(outdir / "figures" / "fig5_calibration.png", dpi=240)
    plt.close(fig)


def run(data_path: str | Path, outdir: str | Path, repeats=10, splits=10, bootstrap=2000) -> dict:
    outdir = Path(outdir)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    source = load_records(data_path)
    aud = audit(source)
    primary = qc_records(source)
    X, y = xy(primary)

    desc = descriptives(primary)
    write_csv(outdir / "tables" / "descriptive_statistics.csv", desc)

    ablation, predictions, y = feature_ablation(primary, repeats=repeats, splits=splits, bootstrap=bootstrap)
    write_csv(outdir / "tables" / "feature_ablation.csv", ablation)

    acl = float(roc_auc_score(y, predictions["NW + NH"]) - roc_auc_score(y, predictions["NI only"]))
    acl_low, acl_high = bootstrap_acl_ci(y, predictions["NW + NH"], predictions["NI only"], n=max(bootstrap, 3000))

    models = model_benchmark(primary, repeats=min(repeats, 3), splits=splits, bootstrap=bootstrap)
    write_csv(outdir / "tables" / "model_comparison.csv", models)

    selective = selective_table(y, predictions["NW + NH"])
    write_csv(outdir / "tables" / "selective_prediction.csv", selective)

    calibration = calibration_summary(y, predictions["NW + NH"])
    write_csv(outdir / "tables" / "calibration.csv", calibration)

    sensitivity = sensitivity_analysis(source, repeats=repeats, splits=splits, bootstrap=bootstrap)
    write_csv(outdir / "tables" / "sensitivity_analysis.csv", sensitivity)

    _figures(outdir, X, y, predictions, selective)

    summary = {
        "audit": aud,
        "analysis_n": len(primary),
        "male_n": int(y.sum()),
        "female_n": int(len(y) - y.sum()),
        "acl_auc": acl,
        "acl_auc_95ci": [acl_low, acl_high],
        "feature_ablation": ablation,
        "models": models,
        "selective_prediction": selective,
        "calibration": calibration,
        "sensitivity_analysis": sensitivity,
        "analysis_settings": {"feature_ablation_repeats": repeats, "model_benchmark_repeats": min(repeats, 3), "sensitivity_repeats": repeats, "splits": splits, "bootstrap": bootstrap, "seed": SEED},
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
