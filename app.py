# Copyright (C) 2026 Mohammad Amir Khusru Akhtar
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from pathlib import Path
import sys

# Make the local src/ package visible on Streamlit Cloud
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import streamlit as st

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from morphopreserve.analysis import (
    SEED,
    audit,
    bootstrap_acl_ci,
    bootstrap_auc_ci,
    calibration_summary,
    descriptives,
    load_records,
    metric_row,
    qc_records,
    repeated_oof,
    selective_table,
    sensitivity_analysis,
    xy,
)
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="MorphoPreserve-DVI", page_icon="🧬", layout="wide")
st.title("MorphoPreserve-DVI")
st.caption("Information-preserving, uncertainty-aware supplementary biological profiling for DVI - not an identification system")

DATA = Path(__file__).parent / "data" / "ranchi_nasal_morphometry.csv"
source = load_records(DATA)

with st.sidebar:
    st.header("Analysis controls")
    width_min = st.number_input("Width plausibility minimum (mm)", 10.0, 40.0, 20.0, 0.5)
    width_max = st.number_input("Width plausibility maximum (mm)", 45.0, 100.0, 70.0, 0.5)
    height_min = st.number_input("Height plausibility minimum (mm)", 10.0, 40.0, 20.0, 0.5)
    height_max = st.number_input("Height plausibility maximum (mm)", 45.0, 120.0, 70.0, 0.5)
    repeats = st.slider("CV repeats", 1, 20, 5, 1)
    splits = st.slider("CV folds", 5, 10, 10, 1)
    bootstrap_n = st.slider("Bootstrap resamples", 200, 3000, 1000, 200)
    threshold = st.slider("Abstention threshold", 0.50, 0.90, 0.70, 0.01)

records = qc_records(source, (width_min, width_max), (height_min, height_max))
X, y = xy(records)

@st.cache_data(show_spinner=False)
def compute_analysis(width_min, width_max, height_min, height_max, repeats, splits, bootstrap_n):
    src = load_records(DATA)
    recs = qc_records(src, (width_min, width_max), (height_min, height_max))
    X0, y0 = xy(recs)
    base = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=5000, solver="liblinear", random_state=SEED))])
    feature_sets = {"NI only":[2], "NW only":[0], "NH only":[1], "NW + NH":[0,1], "NW + NH + NI":[0,1,2]}
    feature_rows, preds = [], {}
    for name, idx in feature_sets.items():
        p = repeated_oof(X0[:,idx], y0, base, repeats=repeats, splits=splits)
        preds[name] = p
        row = metric_row(name, y0, p)
        lo, hi = bootstrap_auc_ci(y0, p, n=bootstrap_n)
        row.update({"auc_95ci_low":lo, "auc_95ci_high":hi})
        feature_rows.append(row)
    acl = roc_auc_score(y0, preds["NW + NH"]) - roc_auc_score(y0, preds["NI only"])
    acl_ci = bootstrap_acl_ci(y0, preds["NW + NH"], preds["NI only"], n=max(bootstrap_n,500))
    return recs, X0, y0, feature_rows, preds, float(acl), acl_ci

with st.spinner("Running reproducible cross-validation..."):
    recs, X0, y0, feature_rows, preds, acl, acl_ci = compute_analysis(width_min, width_max, height_min, height_max, repeats, splits, bootstrap_n)

c1,c2,c3,c4 = st.columns(4)
c1.metric("Included records", len(recs))
c2.metric("Male / female", f"{int(y0.sum())} / {len(y0)-int(y0.sum())}")
c3.metric("ACL (AUC gap)", f"{acl:.3f}")
c4.metric("ACL 95% CI", f"{acl_ci[0]:.3f} to {acl_ci[1]:.3f}")

st.warning("All outputs are research decision-support results. They must not be used as a declaration of victim identity. Formal DVI identification requires appropriate reconciliation and primary identifiers.")

tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["Data audit","Representation ablation","Model explorer","Selective prediction","Sensitivity","Single-case explorer"])

with tab1:
    st.subheader("Source-data integrity audit")
    st.json(audit(source, (width_min,width_max), (height_min,height_max)))
    st.subheader("Sex-stratified descriptive statistics")
    st.dataframe(descriptives(recs), use_container_width=True)
    st.caption("NI is always recomputed from primary width and height; stored NI/type fields are audited, not trusted as modelling inputs.")

with tab2:
    st.subheader("Information-preserving representation experiment")
    st.dataframe(feature_rows, use_container_width=True)
    st.bar_chart({row["name"]: row["auc"] for row in feature_rows})
    st.info(f"Anthropometric Compression Loss = AUC(NW+NH) - AUC(NI) = {acl:.3f} (bootstrap 95% CI {acl_ci[0]:.3f}-{acl_ci[1]:.3f}).")

with tab3:
    st.subheader("Prespecified model explorer")
    model_name = st.selectbox("Model", ["Logistic regression","LDA","Linear SVM","Random forest","Gradient boosting"])
    models = {
        "Logistic regression": Pipeline([("scale",StandardScaler()),("clf",LogisticRegression(C=1,max_iter=5000,solver="liblinear",random_state=SEED))]),
        "LDA": LinearDiscriminantAnalysis(),
        "Linear SVM": Pipeline([("scale",StandardScaler()),("clf",SVC(kernel="linear",C=1,probability=True,random_state=SEED))]),
        "Random forest": RandomForestClassifier(n_estimators=200,max_depth=4,min_samples_leaf=3,class_weight="balanced",random_state=SEED),
        "Gradient boosting": GradientBoostingClassifier(n_estimators=150,learning_rate=.05,max_depth=1,random_state=SEED),
    }
    features = st.selectbox("Features", ["NW + NH","NI only","NW only","NH only","NW + NH + NI"])
    idx = {"NW + NH":[0,1],"NI only":[2],"NW only":[0],"NH only":[1],"NW + NH + NI":[0,1,2]}[features]
    p_model = repeated_oof(X0[:,idx], y0, models[model_name], repeats=repeats, splits=splits)
    st.json(metric_row(f"{model_name} | {features}", y0, p_model))
    st.caption("Changing models is exploratory. The manuscript's principal inference is the matched feature-representation ablation, not selection of the highest-scoring algorithm.")

with tab4:
    st.subheader("Explicit reject option")
    p = preds["NW + NH"]
    mask = (p>=threshold) | (p<=1-threshold)
    pred = (p[mask]>=.5).astype(int) if mask.any() else np.array([])
    coverage = float(mask.mean())
    accuracy = float((pred==y0[mask]).mean()) if mask.any() else float("nan")
    a,b,c = st.columns(3)
    a.metric("Coverage", f"{coverage:.1%}")
    b.metric("Abstention", f"{1-coverage:.1%}")
    c.metric("Accuracy among classified", f"{accuracy:.1%}" if mask.any() else "N/A")
    st.dataframe(selective_table(y0,p),use_container_width=True)
    st.subheader("Internal calibration")
    st.dataframe(calibration_summary(y0,p),use_container_width=True)

with tab5:
    st.subheader("Cleaning-rule sensitivity")
    st.dataframe(sensitivity_analysis(source,repeats=repeats,splits=splits,bootstrap=bootstrap_n),use_container_width=True)
    st.caption("The default manuscript analysis uses the broad technical plausibility screen; sensitivity analyses test whether the ACL conclusion depends on that exclusion decision.")

with tab6:
    st.subheader("Single-case research explorer")
    w = st.number_input("Nasal width (mm)", 20.0, 70.0, 36.0, 0.1, key="case_w")
    h = st.number_input("Nasal height (mm)", 20.0, 70.0, 48.0, 0.1, key="case_h")
    fitted = Pipeline([("scale",StandardScaler()),("clf",LogisticRegression(C=1,max_iter=5000,solver="liblinear",random_state=SEED))]).fit(X0[:,:2],y0)
    probability = float(fitted.predict_proba(np.array([[w,h]]))[:,1][0])
    ni = 100*w/h
    if probability >= threshold:
        decision = "Male-supporting"
    elif probability <= 1-threshold:
        decision = "Female-supporting"
    else:
        decision = "INDETERMINATE"
    x1,x2,x3 = st.columns(3)
    x1.metric("Recomputed NI",f"{ni:.2f}")
    x2.metric("P(male-supporting)",f"{probability:.3f}")
    x3.metric("Output",decision)
    st.error("This is not an identity determination and is not validated for operational casework. It is an interactive research demonstration of the manuscript's uncertainty-aware profiling rule.")
