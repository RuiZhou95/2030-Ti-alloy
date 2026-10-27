"""Validate the public data package and refresh deterministic checksums."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HISTORICAL = DATA / "historical_78_material_states.csv"
TRIALS = DATA / "independent_trials_2.csv"
FORBIDDEN_INTERNAL_FIELDS = {
    "ys_source",
    "uts_source",
    "ta1_field_transposition",
    "posthoc_case",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate() -> None:
    historical = pd.read_csv(HISTORICAL)
    trials = pd.read_csv(TRIALS)
    if len(historical) != 78 or len(trials) != 2:
        raise ValueError("Released data must contain 78 historical and 2 trial records")
    overlap = set(historical.sample_id.astype(str)) & set(trials.sample_id.astype(str))
    if overlap:
        raise ValueError(f"Historical/trial sample overlap: {sorted(overlap)}")
    for label, frame in (("historical", historical), ("trial", trials)):
        forbidden = sorted(FORBIDDEN_INTERNAL_FIELDS & set(frame.columns))
        if forbidden:
            raise ValueError(f"{label} data contain internal-only fields: {forbidden}")
        if frame[["yield_strength_mpa", "ultimate_tensile_strength_mpa", "elongation_pct"]].isna().any().any():
            raise ValueError(f"{label} data contain missing mechanical targets")
        if (frame.yield_strength_mpa > frame.ultimate_tensile_strength_mpa).any():
            raise ValueError(f"{label} data violate YS <= UTS")

    lines = [f"{sha256(path)}  {path.name}" for path in (HISTORICAL, TRIALS)]
    (DATA / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"historical_rows": len(historical), "trial_rows": len(trials), "overlap": 0, "ys_gt_uts": 0})


if __name__ == "__main__":
    validate()
