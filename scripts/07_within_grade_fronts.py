"""Compute observed nondominated fronts separately within each alloy grade."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
GRADE="alloy_or_coating_name"
def nondominated(frame,strength,ductility="elongation_pct"):
    q=frame.dropna(subset=[strength,ductility]).copy()
    x=q[strength].to_numpy(float);y=q[ductility].to_numpy(float)
    keep=np.ones(len(q),dtype=bool)
    for i in range(len(q)):
        keep[i]=not np.any((x>=x[i])&(y>=y[i])&((x>x[i])|(y>y[i])))
    return q.loc[keep].sort_values(ductility)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,default=Path("outputs/data/01_processed_data.csv"))
    ap.add_argument("--output-dir",type=Path,default=Path("outputs/tables"))
    a=ap.parse_args();df=pd.read_csv(a.input)
    groups=df
    if "replicate_group" in df:
        numeric=list(df.select_dtypes(include="number").columns)
        agg={c:"mean" for c in numeric}
        for c in ["sample_id",GRADE,"material_system"]:agg[c]=lambda s:"/".join(map(str,s))
        groups=df.groupby("replicate_group",as_index=False).agg(agg)
    rows=[]
    for grade,g in groups.groupby(GRADE):
        for strength,label in [("yield_strength_mpa","YS_EL"),("ultimate_tensile_strength_mpa","UTS_EL")]:
            for _,r in nondominated(g,strength).iterrows():
                rows.append({"grade":grade,"front":label,"sample_id":r.get("sample_id",""),"x":r[strength],"y":r["elongation_pct"]})
    a.output_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(a.output_dir/"historical_local_pareto.csv",index=False)
    ys=nondominated(groups,"yield_strength_mpa")
    uts=nondominated(groups,"ultimate_tensile_strength_mpa")
    ys[["yield_strength_mpa","elongation_pct","ultimate_tensile_strength_mpa"]].to_csv(a.output_dir/"multiobj_front_ys.csv",index=False)
    uts[["ultimate_tensile_strength_mpa","elongation_pct","yield_strength_mpa"]].to_csv(a.output_dir/"multiobj_front_uts.csv",index=False)
if __name__=="__main__":main()
