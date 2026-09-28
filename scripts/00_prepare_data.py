"""Prepare separate historical-modeling and post-model trial inputs."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

TARGETS=["yield_strength_mpa","ultimate_tensile_strength_mpa","elongation_pct"]
META=["sample_id","material_system","alloy_or_coating_name"]
COMPOSITION=["Al_wt_pct","Ti_wt_pct","Fe_wt_pct","Ni_wt_pct","Cr_wt_pct","Mn_wt_pct",
"Si_wt_pct","Cu_wt_pct","Zn_wt_pct","Mg_wt_pct","Mo_wt_pct","V_wt_pct","Nb_wt_pct",
"Zr_wt_pct","Sn_wt_pct","Co_wt_pct","C_wt_pct","N_wt_pct","O_wt_pct","B_wt_pct",
"P_wt_pct","S_wt_pct","H"]
PROCESS=["homogenization_temp_c","homogenization_time_h","solution_temp_c","solution_time_h",
"aging_temp_c","aging_time_h","annealing_temp_c","annealing_time_h","rolling_temp_c",
"rolling_reduction_pct","forging_temp_c","specimen_thickness_mm"]
MISSING={"无","不关注","无法给出","无法获取","/",""}
CATEGORICAL=["product_form","manufacturing_route","melting_method","heat_treatment_route","microstructure_class","sample_orientation"]

def read_table(path:Path,sheet:str|None,header:int):
    if path.suffix.lower() in {".xlsx",".xls"}:
        return pd.read_excel(path,sheet_name=sheet or 0,header=header)
    return pd.read_csv(path)

def clean(df:pd.DataFrame)->pd.DataFrame:
    df=df.copy()
    missing=[c for c in META+TARGETS if c not in df]
    if missing:raise ValueError(f"Missing required columns: {missing}")
    df=df[df.sample_id.notna()].copy()
    for c in COMPOSITION+PROCESS+TARGETS:
        if c not in df:df[c]=np.nan
        df[c]=pd.to_numeric(df[c].replace(list(MISSING),np.nan),errors="coerce")
    for c in CATEGORICAL:
        if c not in df:df[c]="unknown"
        df[c]=df[c].fillna("unknown").astype(str)
    hashes=pd.util.hash_pandas_object(df[COMPOSITION+PROCESS].fillna(-9.87654321e99),index=False)
    df["replicate_group"]=hashes.astype("uint64").astype(str)
    return df.reset_index(drop=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--historical",type=Path,required=True)
    ap.add_argument("--trials",type=Path,required=True)
    ap.add_argument("--sheet",default=None)
    ap.add_argument("--header",type=int,default=0)
    ap.add_argument("--output-dir",type=Path,default=Path("outputs/data"))
    ap.add_argument("--expect-historical",type=int)
    ap.add_argument("--expect-trials",type=int)
    a=ap.parse_args()
    hist=clean(read_table(a.historical,a.sheet,a.header))
    trials=clean(read_table(a.trials,a.sheet,a.header))
    overlap=set(hist.sample_id.astype(str))&set(trials.sample_id.astype(str))
    if overlap:raise ValueError(f"Historical/trial sample overlap: {sorted(overlap)}")
    if a.expect_historical is not None and len(hist)!=a.expect_historical:raise ValueError("Historical row count mismatch")
    if a.expect_trials is not None and len(trials)!=a.expect_trials:raise ValueError("Trial row count mismatch")
    a.output_dir.mkdir(parents=True,exist_ok=True)
    hist.to_csv(a.output_dir/"01_processed_data.csv",index=False)
    trials.to_csv(a.output_dir/"trial_records.csv",index=False)
    print({"historical_rows":len(hist),"historical_groups":hist.replicate_group.nunique(),"trial_rows":len(trials)})
if __name__=="__main__":main()
