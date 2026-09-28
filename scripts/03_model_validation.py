"""Recompute all SI model diagnostics using only the 78 historical states."""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
TABLES = ROOT / "outputs" / "tables"; DATA = ROOT / "outputs" / "data"
SEED = 20260827
TARGETS = ["yield_strength_mpa", "ultimate_tensile_strength_mpa", "elongation_pct"]
GRADE = "alloy_or_coating_name"
NUMERIC = [
    "Al_wt_pct","Fe_wt_pct","Cr_wt_pct","Mo_wt_pct","V_wt_pct","Nb_wt_pct","Zr_wt_pct",
    "Sn_wt_pct","O_wt_pct","N_wt_pct","C_wt_pct","solution_temp_c","solution_time_h",
    "aging_temp_c","aging_time_h","annealing_temp_c","annealing_time_h","rolling_temp_c",
    "rolling_reduction_pct","forging_temp_c","specimen_thickness_mm"]
CAT = [GRADE,"product_form","manufacturing_route","melting_method","heat_treatment_route",
       "microstructure_class","sample_orientation"]


def preprocessor(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),("scale", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ], sparse_threshold=0)


def pipe(model, num=NUMERIC, cat=CAT):
    return Pipeline([("prep", preprocessor(num, cat)), ("model", model)])


def metrics(y, pred):
    return r2_score(y,pred), mean_absolute_error(y,pred), mean_squared_error(y,pred,squared=False)


def grouped_oof(df, target, model, num=NUMERIC, cat=CAT):
    groups=df.replicate_group.astype(str).values; y=df[target].values; pred=np.full(len(df),np.nan)
    for tr,te in GroupKFold(5).split(df,y,groups):
        p=pipe(model(),num,cat);p.fit(df.iloc[tr],y[tr]);pred[te]=p.predict(df.iloc[te])
    return pred


def model_factories():
    return {
        "Ridge": lambda:Ridge(alpha=10),
        "XGBoost": lambda:HistGradientBoostingRegressor(max_iter=160,learning_rate=.05,max_leaf_nodes=15,random_state=SEED),
        "RandomForest": lambda:RandomForestRegressor(n_estimators=80,min_samples_leaf=2,max_features=.7,random_state=SEED,n_jobs=-1),
        "SVR": lambda:SVR(C=10,epsilon=.1,gamma="scale"),
        "GPR": lambda:GaussianProcessRegressor(kernel=RBF(1.0)+WhiteKernel(1.0),optimizer=None,normalize_y=True,random_state=SEED),
        "MLP": lambda:MLPRegressor(hidden_layer_sizes=(16,),alpha=.01,max_iter=350,random_state=SEED),
    }


def main():
    warnings.filterwarnings("ignore")
    TABLES.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(DATA/"01_processed_data.csv")
    for column in NUMERIC:
        df[column]=pd.to_numeric(df[column],errors="coerce")
    if df.sample_id.astype(str).isin(["Ti_9","Ti_20"]).any():
        raise ValueError("Historical input contains trial sample IDs; keep trial data separate.")

    # Six-model grouped benchmark.
    summaries=[]
    for target in TARGETS:
        y=df[target].values
        for name,factory in model_factories().items():
            pred=grouped_oof(df,target,factory)
            r2,mae,rmse=metrics(y,pred)
            summaries.append(dict(target=target,model=name,train_r2=np.nan,test_r2=np.nan,
                                  cv_r2_mean=r2,cv_r2_std=np.nan,test_rmse=rmse,best_params="78-only grouped CV"))
    pd.DataFrame(summaries).to_csv(DATA/"model_summary.csv",index=False)

    # Validation ladder.
    ladder=[]
    rf=lambda:RandomForestRegressor(n_estimators=60,min_samples_leaf=2,max_features=.7,random_state=SEED,n_jobs=-1)
    for target in TARGETS:
        y=df[target].values; groups=df.replicate_group.astype(str).values
        for rep,(tr,te) in enumerate(GroupShuffleSplit(8,test_size=.2,random_state=SEED).split(df,y,groups)):
            p=pipe(rf());p.fit(df.iloc[tr],y[tr]);pr=p.predict(df.iloc[te]);r2,mae,_=metrics(y[te],pr)
            ladder.append(dict(target=target,regime="Duplicate-safe grouped CV",repeat=rep,r2=r2,mae=mae))
        for rep,(tr,te) in enumerate(KFold(8,shuffle=True,random_state=SEED).split(df)):
            p=pipe(rf());p.fit(df.iloc[tr],y[tr]);pr=p.predict(df.iloc[te]);r2,mae,_=metrics(y[te],pr)
            ladder.append(dict(target=target,regime="Naive random holdout",repeat=rep,r2=r2,mae=mae))
        for regime,col in [("Leave-one-grade-out",GRADE),("Leave-one-system-out","material_system")]:
            for rep,held in enumerate(sorted(df[col].astype(str).unique())):
                te=np.where(df[col].astype(str).values==held)[0];tr=np.where(df[col].astype(str).values!=held)[0]
                if len(te)<2: continue
                p=pipe(rf());p.fit(df.iloc[tr],y[tr]);pr=p.predict(df.iloc[te]);r2,mae,_=metrics(y[te],pr)
                ladder.append(dict(target=target,regime=regime,repeat=rep,r2=r2,mae=mae))
    pd.DataFrame(ladder).to_csv(TABLES/"validation_ladder.csv",index=False)

    # Group-safe y-scrambling.
    ys=[]
    folds=list(GroupKFold(5).split(df,df[TARGETS[0]],df.replicate_group.astype(str)))
    rng=np.random.default_rng(SEED)
    for target in TARGETS:
        y=df[target].values
        obs=grouped_oof(df,target,rf);ys.append(dict(target=target,kind="Observed",iteration=0,r2=r2_score(y,obs)))
        for it in range(15):
            yp=rng.permutation(y);pred=np.full(len(df),np.nan)
            for tr,te in folds:
                p=pipe(rf());p.fit(df.iloc[tr],yp[tr]);pred[te]=p.predict(df.iloc[te])
            ys.append(dict(target=target,kind="Y-scrambled",iteration=it,r2=r2_score(yp,pred)))
    pd.DataFrame(ys).to_csv(TABLES/"y_scrambling.csv",index=False)

    # OOF errors and numeric-domain nearest-neighbour distances (78-only).
    num=SimpleImputer(strategy="median").fit_transform(df[NUMERIC]);z=StandardScaler().fit_transform(num)
    dist=np.sqrt(((z[:,None,:]-z[None,:,:])**2).mean(axis=2));np.fill_diagonal(dist,np.inf);nearest=dist.min(axis=1)
    err=[];oof=[]
    for target in TARGETS:
        pred=grouped_oof(df,target,rf);y=df[target].values
        for i,r in df.iterrows():
            err.append(dict(target=target,repeat=0,sample_id=r.sample_id,grade=r[GRADE],distance=nearest[i],abs_error=abs(y[i]-pred[i])))
            oof.append(dict(target=target,repeat=0,sample_id=r.sample_id,grade=r[GRADE],observed=y[i],predicted=pred[i]))
    pd.DataFrame(err).to_csv(TABLES/"ood_error_records.csv",index=False)
    pd.DataFrame(oof).to_csv(TABLES/"rf_grouped_oof_predictions.csv",index=False)

    # Factor increments under the same grouped CV.
    factor_defs=[("grade",[],[GRADE]),("grade + product",[],[GRADE,"product_form"]),
                 ("grade + route",[],[GRADE,"manufacturing_route"]),("grade + melting",[],[GRADE,"melting_method"]),
                 ("grade + heat",["solution_temp_c","aging_temp_c","annealing_temp_c"],[GRADE,"heat_treatment_route"]),
                 ("grade + structure",[],[GRADE,"microstructure_class"]),("grade + orientation",[],[GRADE,"sample_orientation"]),
                 ("grade + all metadata",NUMERIC,CAT)]
    inc=[]
    for target in TARGETS:
        base=grouped_oof(df,target,rf,num=[],cat=[GRADE]);base_mae=mean_absolute_error(df[target],base)
        for label,numcols,catcols in factor_defs:
            pred=grouped_oof(df,target,rf,num=numcols,cat=catcols);r2,mae,rmse=metrics(df[target],pred)
            inc.append(dict(target=target,repeat=0,model=label,r2=r2,mae=mae,rmse=rmse,grade_mae=base_mae,mae_improvement_pct=100*(base_mae-mae)/base_mae))
    pd.DataFrame(inc).to_csv(TABLES/"factor_increment_cv.csv",index=False)

    # Grade acquisition priorities computed only from historical states.
    out=pd.DataFrame(err); rows=[]
    for grade,g in df.groupby(GRADE):
        e=out[out.grade==grade].groupby("target").abs_error.mean().mean()
        scale=df[TARGETS].std().mean(); norm=e/scale
        d=nearest[g.index].mean(); sd=g[TARGETS].std().mean()
        rows.append(dict(grade=grade,n_samples=len(g),normalized_logo_error=norm,mean_ood_distance=d,within_grade_sd_mean=sd))
    pri=pd.DataFrame(rows)
    for c in ["normalized_logo_error","mean_ood_distance","within_grade_sd_mean"]:
        lo,hi=pri[c].min(),pri[c].max();pri[c+"_scaled"]=(pri[c]-lo)/(hi-lo if hi>lo else 1)
    pri["priority_score"]=.45*pri.normalized_logo_error_scaled+.35*pri.mean_ood_distance_scaled+.20*pri.within_grade_sd_mean_scaled
    pri["priority_rank"]=pri.priority_score.rank(method="first",ascending=False).astype(int)
    pri[["grade","n_samples","normalized_logo_error","mean_ood_distance","within_grade_sd_mean","priority_score","priority_rank"]].to_csv(TABLES/"data_acquisition_priority.csv",index=False)

    # Global versus within-grade correlations, recomputed on the isolated historical set.
    corr_global=[];corr_within=[]
    centered=df.copy()
    for c in NUMERIC+TARGETS:
        centered[c]=df[c]-df.groupby(GRADE)[c].transform("mean")
    for feature in NUMERIC:
        for target in TARGETS:
            corr_global.append(dict(feature=feature,target=target,correlation=df[[feature,target]].corr().iloc[0,1]))
            corr_within.append(dict(feature=feature,target=target,correlation=centered[[feature,target]].corr().iloc[0,1]))
    pd.DataFrame(corr_global).to_csv(TABLES/"global_feature_correlations.csv",index=False)
    pd.DataFrame(corr_within).to_csv(TABLES/"within_grade_correlations.csv",index=False)

    # Grade explainability and two-component multi-view PCA.
    grade_r2=[]
    for feature in NUMERIC:
        x=df[feature]; valid=x.notna(); total=((x[valid]-x[valid].mean())**2).sum()
        within=((x[valid]-df.loc[valid].groupby(GRADE)[feature].transform("mean"))**2).sum()
        grade_r2.append(dict(feature=feature,grade_r2=(1-within/total) if total>0 else 0))
    pd.DataFrame(grade_r2).to_csv(TABLES/"feature_grade_explainability.csv",index=False)
    views={"Composition":NUMERIC[:11],"Process":NUMERIC[11:],"Combined":NUMERIC,
           "Metadata":["rolling_temp_c","solution_temp_c","aging_temp_c","specimen_thickness_mm"]}
    pca_rows=[]
    for view,cols in views.items():
        xx=SimpleImputer(strategy="median").fit_transform(df[cols]);xx=StandardScaler().fit_transform(xx)
        ev=PCA(n_components=2,random_state=SEED).fit(xx).explained_variance_ratio_
        for i,val in enumerate(ev,1):pca_rows.append(dict(view=view,component=i,explained_variance=val))
    pd.DataFrame(pca_rows).to_csv(TABLES/"pca_explained_variance.csv",index=False)
    print("recomputed 78-only diagnostics",len(df))


if __name__=="__main__":main()
