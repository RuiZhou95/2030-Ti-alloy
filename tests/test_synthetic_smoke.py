from pathlib import Path
import importlib.util
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("fronts",ROOT/"scripts"/"07_within_grade_fronts.py")
fronts=importlib.util.module_from_spec(spec);spec.loader.exec_module(fronts)
def test_nondominated_front():
    df=pd.DataFrame({"yield_strength_mpa":[100,110,105,90],"elongation_pct":[10,8,12,15]})
    assert set(fronts.nondominated(df,"yield_strength_mpa").index)=={1,2,3}
