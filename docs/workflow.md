# Workflow

1. Prepare separate historical and post-model trial inputs.
2. Build continuous composition-processing features.
3. Train and compare six model classes.
4. Run grouped, leave-grade, leave-system, and scrambling diagnostics.
5. Compute model feature contributions.
6. Run and merge symbolic-regression searches.
7. Compute observed nondominated fronts within each alloy grade.
8. Fit the selected model settings on the 78 historical records and apply them
   once to the separate post-model trial file.
9. Render manuscript figures from the physically consistent data and the
   corresponding model outputs.

Generated files are written under `outputs/` and are excluded from Git.
The random-split benchmark and grouped benchmark are written to
`model_summary.csv` and `grouped_model_summary.csv`, respectively; neither file
overwrites the other.
