from pathlib import Path
import importlib.util
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("fronts",ROOT/"scripts"/"07_within_grade_fronts.py")
fronts=importlib.util.module_from_spec(spec);spec.loader.exec_module(fronts)
def test_nondominated_front():
    df=pd.DataFrame({"yield_strength_mpa":[100,110,105,90],"elongation_pct":[10,8,12,15]})
    assert set(fronts.nondominated(df,"yield_strength_mpa").index)=={1,2,3}

def test_prepare_rejects_ys_above_uts():
    prepare_spec=importlib.util.spec_from_file_location(
        "prepare",ROOT/"scripts"/"00_prepare_data.py"
    )
    prepare=importlib.util.module_from_spec(prepare_spec);prepare_spec.loader.exec_module(prepare)
    df=pd.DataFrame({
        "sample_id":["bad"],"material_system":["titanium"],
        "alloy_or_coating_name":["TA1"],"yield_strength_mpa":[320.0],
        "ultimate_tensile_strength_mpa":[220.0],"elongation_pct":[20.0],
    })
    try:
        prepare.clean(df)
    except ValueError as exc:
        assert "yield_strength_mpa exceeds ultimate_tensile_strength_mpa" in str(exc)
    else:
        raise AssertionError("nonphysical YS/UTS record was accepted")

def test_released_data_invariants():
    historical=ROOT/"data"/"historical_78_material_states.csv"
    trials=ROOT/"data"/"independent_trials_2.csv"
    if not historical.exists() or not trials.exists():
        return
    hist=pd.read_csv(historical);trial=pd.read_csv(trials)
    assert len(hist)==78 and len(trial)==2
    assert set(hist.sample_id.astype(str)).isdisjoint(set(trial.sample_id.astype(str)))
    assert (hist.yield_strength_mpa<=hist.ultimate_tensile_strength_mpa).all()
    assert (trial.yield_strength_mpa<=trial.ultimate_tensile_strength_mpa).all()
