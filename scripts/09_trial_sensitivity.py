"""Historical-only refit sensitivity; primary frozen predictions are preserved."""
from pathlib import Path
import importlib.util
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/"outputs"/"data"
spec=importlib.util.spec_from_file_location("diag",ROOT/"scripts"/"03_model_validation.py")
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
def main():
    hist=pd.read_csv(DATA/"01_processed_data.csv");trials=pd.read_csv(DATA/"trial_records.csv")
    frozen=pd.read_csv(DATA/"verification_predictions.csv");rows=[]
    for target in d.TARGETS:
        factory=(lambda:Ridge(alpha=10)) if target=="yield_strength_mpa" else (lambda:RandomForestRegressor(n_estimators=300,min_samples_leaf=2,max_features=.7,random_state=d.SEED,n_jobs=-1))
        oof=d.grouped_oof(hist,target,factory);rmse=mean_squared_error(hist[target],oof,squared=False)
        model=d.pipe(factory());model.fit(hist,hist[target]);pred=model.predict(trials)
        for sid,val,measured in zip(trials.sample_id,pred,trials[target]):
            rows.append({"sample_id":sid,"target":target,"measured":measured,"sensitivity_refit_prediction":val,"sensitivity_cv_rmse":rmse})
    pd.DataFrame(rows).to_csv(DATA/"trial_refit_sensitivity.csv",index=False)
if __name__=="__main__":main()
