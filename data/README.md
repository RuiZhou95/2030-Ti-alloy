# Titanium and titanium-alloy plate data

## Summary

This directory contains the analysis-ready data supporting the manuscript on
composition-processing-strength-ductility relationships in titanium and
titanium-alloy plate. Historical model-development records and post-model trial
records are released as separate files so the trial responses cannot enter
model selection or fitting.

## Files

- `historical_78_material_states.csv`: 78 historical records covering TA1,
  TA2, TC4, Ti6411, and Ti6321. These records are used for model development,
  grouped validation, interpretation, symbolic regression, and within-grade
  strength-ductility analysis.
- `independent_trials_2.csv`: two subsequently fabricated and tested candidates,
  Ti_9 and Ti_20. Each row contains one tensile-test result.
- `data_dictionary.csv`: field-level definitions, units, and data types for the
  released CSV files.
- `SHA256SUMS`: SHA-256 checksums for the two CSV files.

## Variables and units

Required identifiers and responses are defined in `docs/input_schema.md`.
Complete field-level metadata are provided in `data_dictionary.csv`.
Composition columns ending in `_wt_pct` are mass fractions in wt.%; temperature
columns ending in `_temp_c` are in degrees Celsius; time columns ending in
`_time_h` are in hours; strength columns ending in `_mpa` are in MPa; and
elongation or reduction columns ending in `_pct` are percentages. Empty cells
represent unavailable or inapplicable fields and are not physical zeros.

## Provenance and quality control

The historical file contains accumulated experimental and testing records for
titanium-alloy plates. The trial file contains the later Ti_9 and Ti_20
fabrication records. The released inputs satisfy
`yield_strength_mpa <= ultimate_tensile_strength_mpa`; the preprocessing script
enforces this condition and stops on any violation. Exact
composition-processing duplicates receive one `replicate_group` identifier and
remain together during grouped validation.

## Reproduction

Create the environment from `environment.yml`, then run the commands in the
repository `README.md`. Generated outputs are written under `outputs/`.
Run `python tools/validate_release_data.py` to verify row counts, file
separation, physical target consistency, excluded internal audit fields, and
checksums.

## Access and licence

The data are publicly accessible in this repository and licensed under the
Creative Commons Attribution 4.0 International licence; see
`LICENSE-DATA.md`. Code is separately licensed under the MIT License.

## Citation

Please cite the associated article and the versioned GitHub release. A preferred
software citation is provided in `CITATION.cff`.
