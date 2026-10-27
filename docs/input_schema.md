# Input schema

The released analysis inputs are `data/historical_78_material_states.csv` and
`data/independent_trials_2.csv`.

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

Every released record must contain finite YS, UTS, and EL values and satisfy
`yield_strength_mpa <= ultimate_tensile_strength_mpa`. The preparation script
terminates before writing outputs if this invariant is violated.

Column names include units where applicable. Composition fields use wt.%,
temperatures use degrees Celsius, time fields use hours, strength uses MPa, and
elongation and reduction fields use percent. Categorical fields retain the
recorded manufacturing, product, heat-treatment, microstructure, and specimen
orientation labels. See `data/README.md` for file-level provenance.
