# Input schema

The repository intentionally contains no experimental data.

Historical modeling records and subsequent trial records must be separate files.
A sample identifier may not occur in both. Trial responses must not be used for
feature selection, hyperparameter tuning, candidate selection, or model freezing.

Required columns are `sample_id`, `material_system`,
`alloy_or_coating_name`, `yield_strength_mpa`,
`ultimate_tensile_strength_mpa`, and `elongation_pct`.

The preparation script accepts the composition and processing columns listed in
`scripts/00_prepare_data.py`. Missing numeric entries should be blank/NA rather
than physical zero unless zero is the measured value. Exact
composition-processing duplicates receive one `replicate_group` identifier.

Optional categorical metadata that are absent are encoded as unknown; this value is a pipeline placeholder and is not interpreted as a measured microstructure.
