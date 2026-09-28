# 2030 Ti Alloy

Reproducible code for modeling composition-processing-property relationships in
titanium and titanium-alloy plate, including grouped validation,
interpretability, symbolic regression, within-grade strength-ductility fronts,
and post-model trial validation.

## Data availability

No experimental records, derived tables, trained models, manuscripts, or figures
are included. Users provide two separate inputs: historical modeling records and
subsequent trial records generated after candidate and model freezing. Trial
responses are never used for candidate selection or model tuning.

## Quick start

```bash
conda env create -f environment.yml
conda activate ti-alloy-ml
python scripts/00_prepare_data.py   --historical /path/to/historical_records.xlsx   --trials /path/to/subsequent_trial_records.xlsx   --sheet metal_samples --header 1   --expect-historical 78 --expect-trials 2
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

## License

MIT License.
