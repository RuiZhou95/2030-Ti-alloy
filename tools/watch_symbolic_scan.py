from pathlib import Path
import os,time
ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get("TI_ALLOY_SR_SCAN_DIR",ROOT/"outputs"/"data"/"sr_scan"))
POPS=[20,40,60];PARS=["0.00032","0.0032","0.032"];MAXS=[8,12,16]
def expected():
    out=[(t,f"pop{p}_par{q}_max{m}") for t in ["yield_strength_mpa","ultimate_tensile_strength_mpa"] for p in POPS for q in PARS for m in MAXS]
    return out+[("elongation_pct","el_single")]
def main():
    while True:
        done=sum((DATA/t/s/"best.json").exists() for t,s in expected())
        print(f"{done}/{len(expected())} searches complete",flush=True)
        if done==len(expected()):return
        time.sleep(120)
if __name__=="__main__":main()
