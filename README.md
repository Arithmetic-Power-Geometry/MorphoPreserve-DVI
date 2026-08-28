# MorphoPreserve-DVI

**Copyright (C) 2026 Mohammad Amir Khusru Akhtar**  
Licensed under the Apache License, Version 2.0.

MorphoPreserve-DVI is the reproducible software companion to the manuscript **“Beyond the Nasal Index: Anthropometric Compression Loss and Uncertainty-Aware Sex Estimation for Disaster Victim Identification.”**

The framework tests whether deterministic compression of primary nasal width and height into the conventional nasal index (NI) removes sex-discriminative information that remains available in the original measurements. It also implements an explicit `INDETERMINATE` outcome for uncertainty-aware supplementary biological profiling in Disaster Victim Identification (DVI).

> **Scope:** This software is intended for research and supplementary biological-profile decision support only. It is **not** a victim-identification system and does not replace primary identification methods such as DNA analysis, fingerprints, forensic odontology, formal reconciliation procedures, or professional forensic judgment.

## GitHub workflow

The repository includes a GitHub Actions workflow for automated reproducibility.

After uploading the repository to GitHub:

1. Open **Actions**.
2. Select **One-click reproducibility**.
3. Click **Run workflow**.

The workflow installs the project, runs the automated tests, regenerates the analysis tables, figures, and summary outputs from the included de-identified analysis table, and uploads the complete `results/` directory as a workflow artifact.

## Local reproduction

Install the project and test dependency:

```bash
python -m pip install -e .
python -m pip install pytest
```

Run the tests:

```bash
pytest -q
```

Reproduce the principal analysis:

```bash
morphopreserve --data data/ranchi_nasal_morphometry.csv --out results --repeats 10 --splits 10 --bootstrap 3000
```

Alternatively, use the supplied Makefile:

```bash
make install
make test
make reproduce
```

## Interactive app

Launch the Streamlit application with:

```bash
streamlit run app.py
```

The interactive application allows researchers to vary:

- technical plausibility bounds;
- cross-validation folds and repeats;
- bootstrap size;
- feature representation;
- model family;
- abstention threshold; and
- individual nasal measurements for research demonstration.

The application provides dedicated views for:

- source-data auditing;
- descriptive statistics;
- representation ablation;
- model comparison;
- calibration assessment;
- selective prediction;
- cleaning-rule sensitivity; and
- single-case research demonstration.

## Reproducibility design

The software implements the analysis reported in the associated preprint using the following prespecified design:

- **Source cohort:** 200 de-identified records comprising 100 male and 100 female participants.
- **Nasal-index reconstruction:** NI is deterministically recomputed as `100 * width / height`.
- **Primary technical screen:** a broad 20–70 mm plausibility range is applied to the primary nasal dimensions. One technically implausible height entry is excluded, yielding a primary analysis sample of `n = 199`.
- **Sensitivity A:** all 200 records are retained while the estimator, cross-validation folds, repeats, and random seed remain identical to the primary analysis.
- **Sensitivity B:** only the specifically flagged record is removed without applying the general range rule; all other analytical settings remain identical to the primary analysis.
- **Principal comparison:** matched standardized logistic regression using NI alone versus nasal width plus nasal height (`NW+NH`).
- **Operational Anthropometric Compression Loss (ACL):** `AUC(NW+NH) - AUC(NI)`.
- **Validation:** repeated stratified 10-fold cross-validation with 10 repeats and fixed random seed `2026` for the principal representation ablation and cleaning-rule sensitivity analyses.
- **Model discipline:** prespecified model specifications are used without synthetic oversampling or opportunistic feature selection.
- **Uncertainty assessment:** bootstrap confidence intervals for AUC and ACL, Brier score, calibration summaries, and reject-option selective prediction are provided.

The principal analysis is designed to test whether preserving the primary anthropometric dimensions retains discriminative information that is attenuated when those dimensions are deterministically compressed into a ratio.

## Repository layout

```text
MorphoPreserve-DVI/
├── .github/
│   └── workflows/
│       └── reproduce.yml
├── data/
│   ├── README.md
│   └── ranchi_nasal_morphometry.csv
├── results/
│   ├── figures/
│   └── tables/
├── src/
│   └── morphopreserve/
│       ├── analysis.py
│       └── cli.py
├── tests/
├── app.py
├── CITATION.cff
├── LICENSE
├── NOTICE
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

### Key components

- `src/morphopreserve/analysis.py` — analysis pipeline, statistical evaluation, sensitivity analyses, and figure generation.
- `src/morphopreserve/cli.py` — command-line reproducibility interface.
- `app.py` — interactive Streamlit research application.
- `data/ranchi_nasal_morphometry.csv` — de-identified morphometric analysis table.
- `data/README.md` — dataset description and responsible-use information.
- `results/tables/` — regenerated tabular results.
- `results/figures/` — regenerated PDF and PNG figures.
- `tests/` — deterministic audit and analysis-consistency tests.
- `.github/workflows/reproduce.yml` — automated one-click reproducibility workflow.
- `CITATION.cff` — machine-readable citation metadata.

## Scientific interpretation

The central scientific claim concerns **representation loss**, not universal classification performance.

MorphoPreserve-DVI evaluates whether reducing multiple primary anthropometric measurements to a deterministic index can remove target-relevant discriminatory information. The operational ACL measure quantifies the cross-validated discrimination difference between the retained primary dimensions and their compressed representation.

The cohort-specific predictive models require independent, population- and acquisition-modality-aware external validation before any operational forensic use. Performance observed in the source cohort should not be assumed to transfer to other populations, acquisition methods, postmortem conditions, or forensic settings.

All probability estimates, thresholds, and selective-prediction outputs generated by the software should therefore be interpreted as exploratory supplementary biological-profile signals rather than individual-identification decisions.

## Data responsibility

The repository uses a de-identified morphometric analysis table containing generated subject identifiers and no names, contact information, or other direct personal identifiers.

Use and redistribution of human-participant data remain subject to the applicable institutional ethics, governance, legal, and data-sharing requirements. Users are responsible for ensuring that any reuse, redistribution, linkage, or external application of the data is consistent with those requirements.

## Citation

Akhtar, M. A. K., Mathew, M., & Xalxo, D. D. B. (2026). *Beyond the Nasal Index: Anthropometric Compression Loss and Uncertainty-Aware Sex Estimation for Disaster Victim Identification* [Preprint]. Zenodo. https://doi.org/10.5281/zenodo.22141702

## License

This software is distributed under the **Apache License, Version 2.0**.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar.

See `LICENSE` and `NOTICE` for details.
