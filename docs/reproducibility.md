# Reproducibility notes

- Historical and post-model trial records are isolated before analysis.
- Exact composition-processing duplicates remain together in grouped validation.
- Validation preprocessing is fitted inside each training fold.
- Cross-grade and cross-family validation define the applicability boundary.
- SHAP values describe model associations, not metallurgical causality.
- Nondominated fronts are calculated within grade; no virtual cross-grade
  composition-processing candidates are generated.
- Model error reference bands do not represent experimental repeatability.
- Preprocessing fails if YS exceeds UTS or if a required mechanical target is
  missing; no automatic value swapping or silent correction is performed.
- The released CSV files are the sole canonical inputs for the published
  analysis. Checksums are provided in `data/SHA256SUMS`.
