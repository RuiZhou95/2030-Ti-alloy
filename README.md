# 2030 Ti Alloy

Reproducible code for modeling composition-processing-property relationships in
titanium and titanium-alloy plate, including grouped validation,
interpretability, symbolic regression, within-grade strength-ductility fronts,
and post-model trial validation.

## Data availability

The repository includes the analysis-ready data used in the study:

- `data/historical_78_material_states.csv`: 78 historical plate records used for
  model development and validation.
- `data/independent_trials_2.csv`: the two post-model trial records for Ti_9 and
  Ti_20.

The two datasets are separated before analysis. Trial responses are never used
for feature selection, hyperparameter tuning, candidate selection, or model
fitting. The preprocessing entry point validates row counts, sample separation,
required targets, and the physical requirement that yield strength does not
exceed ultimate tensile strength.

Dataset metadata, variables, provenance, licence, and checksums are documented
in `data/README.md`, `docs/input_schema.md`, and `data/SHA256SUMS`.

## Quick start

```bash
conda env create -f environment.yml
conda activate ti-alloy-ml
python scripts/00_prepare_data.py \
  --historical data/historical_78_material_states.csv \
  --trials data/independent_trials_2.csv \
  --expect-historical 78 --expect-trials 2
python scripts/01_build_features.py
python scripts/02_train_models.py
python scripts/03_model_validation.py
python scripts/04_interpret_models.py
python scripts/07_within_grade_fronts.py
python scripts/08_trial_validation.py
```

See `docs/input_schema.md`, `docs/workflow.md`, and
`docs/reproducibility.md`.

## Scope

The workflow supports relationships within represented alloy grades and
processing regions. Leave-grade and leave-material-family tests define transfer
limitations and do not establish zero-shot design of unseen grades.

## Licences

- Code: MIT License (`LICENSE`).
- Data: Creative Commons Attribution 4.0 International (`LICENSE-DATA.md`).
