from pathlib import Path
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import pairwise_distances, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBRegressor


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
ORIGINAL_DATA = ROOT / "outputs" / "data"
ORIGINAL_FIGURE = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"
DATA = ROOT / "outputs" / "data"
OUT = ROOT / "outputs" / "figures"
SI_OUT = ROOT / "outputs" / "si_figures"
SOURCE = ROOT / "outputs" / "source_data"
for directory in (OUT, SI_OUT, SOURCE):
    directory.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.2,
    "axes.labelsize": 7.4,
    "axes.titlesize": 8.0,
    "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6,
    "legend.fontsize": 6.4,
    "axes.linewidth": 0.75,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": None,
    "savefig.pad_inches": 0.035,
})

NEUTRAL = "#8E99A4"
DARK = "#27313B"
BLUE = "#356D8D"
TEAL = "#4A9A8A"
RED = "#B85855"
GOLD = "#D99A2B"
PALE = "#E8EDF1"
GRADE_COLORS = {
    "TA1": "#9CA7B1", "TA2": "#587B8D", "TC4": "#2C7A78",
    "Ti6411": "#9B5A55", "Ti6321": "#71658B",
}
GRADE_COLORS_SAT = {
    "TA1": "#4C78A8", "TA2": "#00A6A6", "TC4": "#F28E2B",
    "Ti6411": "#E15759", "Ti6321": "#7A4EAB",
}
GRADE_MARKERS = {"TA1":"o","TA2":"s","TC4":"^","Ti6411":"D","Ti6321":"P"}
FEATURE_LABELS = {
    "homogenization_temp_c": "Homogenization T",
    "homogenization_time_h": "Homogenization time",
    "solution_temp_c": "Solution T",
    "solution_time_h": "Solution time",
    "aging_temp_c": "Aging T",
    "aging_time_h": "Aging time",
    "annealing_temp_c": "Annealing T",
    "annealing_time_h": "Annealing time",
    "rolling_temp_c": "Rolling T",
    "rolling_reduction_pct": "Rolling reduction",
    "specimen_thickness_mm": "Thickness",
    "total_alloy_wt": "Total alloying",
    "Ti_x_Al": "Ti×Al",
    "Ti_x_V": "Ti×V",
}


def feature_label(name):
    return FEATURE_LABELS.get(str(name),str(name).replace("_wt_pct","").replace("_"," "))


def read(name, data=False):
    return pd.read_csv((DATA if data else TABLES) / name)


def read_original(name):
    return pd.read_csv(ORIGINAL_DATA / name, encoding="utf-8-sig")


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.8, pad=2)


def panel(ax, label):
    ax.text(-0.13, 1.05, label, transform=ax.transAxes, fontweight="bold",
            fontsize=8.3, va="bottom", ha="left", color=DARK)


def panel_title(ax, text):
    ax.set_title(text, loc="left", fontweight="bold", fontsize=8.0, pad=5)


def white_cell_text(ax, matrix, fontsize=5.2):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white", fontsize=fontsize, fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=.55,foreground="#27313B")])


def covariance_ellipse(ax, x, y, color, n_std=1.5, alpha=.12, lw=.9):
    xy=np.column_stack([np.asarray(x,float),np.asarray(y,float)])
    xy=xy[np.isfinite(xy).all(axis=1)]
    if len(xy)<3:return
    cov=np.cov(xy,rowvar=False)
    vals,vecs=np.linalg.eigh(cov);order=vals.argsort()[::-1];vals=vals[order];vecs=vecs[:,order]
    angle=np.degrees(np.arctan2(vecs[1,0],vecs[0,0]))
    width,height=2*n_std*np.sqrt(np.maximum(vals,0))
    center=np.median(xy,axis=0)
    ax.add_patch(Ellipse(center,width,height,angle=angle,facecolor=color,edgecolor=color,alpha=alpha,lw=lw))
    ax.scatter(center[0],center[1],s=44,marker="X",color=color,edgecolor="white",lw=.5,zorder=5)


def tight_crop(image, pad_fraction=.018):
    arr=np.asarray(image)
    rgb=arr[...,:3] if arr.ndim==3 else arr
    mask=np.any(rgb<.985,axis=2) if np.issubdtype(rgb.dtype,np.floating) else np.any(rgb<251,axis=2)
    ys,xs=np.where(mask)
    if not len(xs):return arr
    h,w=arr.shape[:2];pad_x=max(4,int(w*pad_fraction));pad_y=max(4,int(h*pad_fraction))
    x0=max(0,int(xs.min())-pad_x);x1=min(w,int(xs.max())+1+pad_x)
    y0=max(0,int(ys.min())-pad_y);y1=min(h,int(ys.max())+1+pad_y)
    return arr[y0:y1,x0:x1]


def nondominated_frame(frame, strength_col, ductility_col="elongation_pct"):
    q=frame.dropna(subset=[strength_col,ductility_col]).copy()
    x=q[strength_col].to_numpy(float);y=q[ductility_col].to_numpy(float)
    keep=np.ones(len(q),dtype=bool)
    for i in range(len(q)):
        dominates=(x>=x[i])&(y>=y[i])&((x>x[i])|(y>y[i]))
        if dominates.any():keep[i]=False
    return q.loc[keep].sort_values(ductility_col)


def save_bundle(fig, stem, source_frames, si=False):
    target = SI_OUT if si else OUT
    fig.savefig(target / f"{stem}.svg", facecolor="white", bbox_inches=None)
    fig.savefig(target / f"{stem}.pdf", facecolor="white", bbox_inches=None)
    fig.savefig(target / f"{stem}.png", dpi=600, facecolor="white", bbox_inches=None)
    fig.savefig(target / f"{stem}.tiff", dpi=600, facecolor="white", bbox_inches=None)
    frames = []
    for label, frame in source_frames:
        f = frame.copy()
        f.insert(0, "panel", label)
        frames.append(f)
    pd.concat(frames, ignore_index=True, sort=False).to_csv(SOURCE / f"{stem}_source_data.csv", index=False)
    plt.close(fig)


def fig1_data_landscape():
    h=read("historical_78_material_states.csv")
    comp_cols=["Al_wt_pct","V_wt_pct","Fe_wt_pct","Cr_wt_pct","Mo_wt_pct","Nb_wt_pct","Zr_wt_pct","O_wt_pct"]
    xx=h[comp_cols].apply(pd.to_numeric,errors="coerce")
    zz=StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(xx))
    candidates=[]
    for perplexity in [8,10,12,18]:
        emb_i=TSNE(n_components=2,perplexity=perplexity,learning_rate="auto",init="pca",random_state=20260827).fit_transform(zz)
        trust=trustworthiness(zz,emb_i,n_neighbors=7)
        distances=pairwise_distances(emb_i);np.fill_diagonal(distances,np.inf)
        span=float(np.linalg.norm(np.ptp(emb_i,axis=0)))
        dispersion=float(np.median(distances.min(axis=1))/(span+1e-12))
        candidates.append({"perplexity":perplexity,"trustworthiness":trust,"dispersion":dispersion,"embedding":emb_i})
    best_trust=max(x["trustworthiness"] for x in candidates)
    eligible=[x for x in candidates if x["trustworthiness"]>=best_trust-.02]
    chosen=max(eligible,key=lambda x:x["dispersion"]);emb=chosen["embedding"]
    pd.DataFrame([{k:v for k,v in x.items() if k!="embedding"} for x in candidates]).assign(
        selected=lambda d:d.perplexity.eq(chosen["perplexity"])).to_csv(SOURCE/"fig1_tsne_parameter_selection.csv",index=False)
    ts=pd.DataFrame({"sample_id":h.sample_id,"grade":h.alloy_or_coating_name,
                     "TSNE1":emb[:,0],"TSNE2":emb[:,1],"perplexity":chosen["perplexity"]})
    fig=plt.figure(figsize=(7.2,4.4));gs=fig.add_gridspec(3,2,width_ratios=[1.25,1],hspace=.52,wspace=.38)
    ax=fig.add_subplot(gs[:,0]);panel(ax,"a");clean(ax)
    for grade,gp in ts.groupby("grade"):
        color=GRADE_COLORS_SAT.get(grade,NEUTRAL)
        ax.scatter(gp.TSNE1,gp.TSNE2,s=18,c=color,marker=GRADE_MARKERS.get(grade,"o"),
                   label=f"{grade} (n={len(gp)})",alpha=.28,edgecolor="none")
        covariance_ellipse(ax,gp.TSNE1,gp.TSNE2,color,n_std=1.5,alpha=.14)
    ax.set_xlabel("t-SNE 1");ax.set_ylabel("t-SNE 2");panel_title(ax,"Composition space");ax.legend(frameon=False,fontsize=5.8)
    props=[("yield_strength_mpa","YS (MPa)"),("ultimate_tensile_strength_mpa","UTS (MPa)"),("elongation_pct","EL (%)")]
    frames=[]
    for i,(col,label) in enumerate(props):
        aa=fig.add_subplot(gs[i,1]);panel(aa,chr(ord('b')+i));clean(aa)
        data=[];pos=[];colors=[]
        for j,grade in enumerate(GRADE_COLORS):
            vals=pd.to_numeric(h.loc[h.alloy_or_coating_name==grade,col],errors="coerce").dropna().values
            data.append(vals);pos.append(j);colors.append(GRADE_COLORS[grade])
        bp=aa.boxplot(data,positions=pos,widths=.55,patch_artist=True,showfliers=False,medianprops=dict(color=DARK,lw=.9),whiskerprops=dict(color=NEUTRAL),capprops=dict(color=NEUTRAL))
        for patch,c in zip(bp["boxes"],colors):patch.set_facecolor(c);patch.set_alpha(.55);patch.set_edgecolor=DARK
        for j,vals in enumerate(data):aa.scatter(np.full(len(vals),j)+np.linspace(-.09,.09,len(vals)),vals,s=10,c=colors[j],alpha=.45,edgecolor="none")
        aa.set_xticks(pos,list(GRADE_COLORS.keys()));plt.setp(aa.get_xticklabels(),rotation=15,rotation_mode="anchor",ha="right");aa.set_ylabel(label)
        panel_title(aa,{"yield_strength_mpa":"YS distribution","ultimate_tensile_strength_mpa":"UTS distribution","elongation_pct":"EL distribution"}[col])
        frames.append(h[["sample_id","alloy_or_coating_name",col]].rename(columns={col:"value"}).assign(target=col))
    save_bundle(fig,"fig1_data_and_property_landscape",[("a",ts),("b-d",pd.concat(frames,ignore_index=True))])


def fig2_material_relationships():
    h=read("historical_78_material_states.csv")
    corr=read_original("correlation_matrix.csv");corr.columns=[str(c).lstrip("\ufeff") for c in corr.columns]
    pear=["yield_strength_mpa_pearson","ultimate_tensile_strength_mpa_pearson","elongation_pct_pearson"]
    top=corr.assign(score=corr[pear].abs().max(axis=1)).nlargest(12,"score")
    gl=read("global_feature_correlations.csv");wg=read("within_grade_correlations.csv");merged=gl.merge(wg,on=["feature","target"],suffixes=("_global","_within"))
    fig=plt.figure(figsize=(7.2,5.1));gs=fig.add_gridspec(2,3,width_ratios=[1.32,1,1],hspace=.50,wspace=.72,left=.09,right=.97,top=.93,bottom=.10)
    ax=fig.add_subplot(gs[:,0]);panel(ax,"a")
    mat=top[pear].to_numpy(float);im=ax.imshow(mat,aspect="auto",vmin=-1,vmax=1,cmap="coolwarm")
    ax.set_xticks(range(3),["YS","UTS","EL"]);ax.set_yticks(range(len(top)),[feature_label(x) for x in top.feature]);panel_title(ax,"Global Pearson correlation")
    plt.setp(ax.get_yticklabels(),rotation=20,rotation_mode="anchor",ha="right")
    white_cell_text(ax,mat,fontsize=5.1)
    fig.colorbar(im,ax=ax,fraction=.035,pad=.018)
    axb=fig.add_subplot(gs[0,1]);panel(axb,"b");clean(axb)
    for grade,g in h.groupby("alloy_or_coating_name"):
        axb.scatter(g.Al_eq,g.Tbeta_est_c,s=28,c=GRADE_COLORS.get(grade,NEUTRAL),alpha=.72,label=grade)
    axb.set_xlabel("Empirical Al equivalent (wt%)");axb.set_ylabel("Tβ,est (°C)",labelpad=7);panel_title(axb,"Alloy-stability descriptors")
    axc=fig.add_subplot(gs[0,2]);panel(axc,"c");clean(axc);hh=h.dropna(subset=["delta_T_roll_c"])
    for grade,g in hh.groupby("alloy_or_coating_name"):
        axc.scatter(g.delta_T_roll_c,g.elongation_pct,s=28,c=GRADE_COLORS.get(grade,NEUTRAL),alpha=.72)
    axc.axvline(0,c=DARK,lw=.7,ls="--");axc.set_xlabel("Troll − Tβ,est (°C)");axc.set_ylabel("EL (%)");panel_title(axc,"Relative rolling T")
    axd=fig.add_subplot(gs[1,1]);panel(axd,"d");clean(axd)
    for grade in ["TA1","TA2"]:
        g=h[h.alloy_or_coating_name==grade];axd.scatter(g.O_wt_pct,g.elongation_pct,s=32,c=GRADE_COLORS[grade],alpha=.75,label=grade)
        if len(g)>2:
            p=np.polyfit(g.O_wt_pct,g.elongation_pct,1);x=np.linspace(g.O_wt_pct.min(),g.O_wt_pct.max(),50);axd.plot(x,np.polyval(p,x),c=GRADE_COLORS[grade],lw=.9)
    axd.set_xlabel("O (wt%)");axd.set_ylabel("EL (%)");axd.legend(frameon=False);panel_title(axd,"Within-grade O–EL")
    axe=fig.add_subplot(gs[1,2]);panel(axe,"e");clean(axe)
    for tar,c in zip(merged.target.unique(),[BLUE,TEAL,RED]):
        z=merged[merged.target==tar];axe.scatter(z.correlation_global,z.correlation_within,s=22,c=c,alpha=.65,label={"yield_strength_mpa":"YS","ultimate_tensile_strength_mpa":"UTS","elongation_pct":"EL"}.get(tar,tar))
    axe.axhline(0,c=NEUTRAL,lw=.6);axe.axvline(0,c=NEUTRAL,lw=.6);axe.plot([-1,1],[-1,1],ls="--",c=DARK,lw=.7)
    axe.set_xlabel("Global correlation");axe.set_ylabel("Within-grade correlation");axe.legend(frameon=False,fontsize=5.7);panel_title(axe,"Correlation hierarchy")
    save_bundle(fig,"fig2_material_relationships",[("a",top),("b-d",h),("e",merged)])


def _results_json():
    with open(ORIGINAL_DATA/"results.json",encoding="utf-8") as f:return json.load(f)


_SHAP_CACHE=None


def recomputed_shap():
    global _SHAP_CACHE
    if _SHAP_CACHE is not None:
        return _SHAP_CACHE
    df=read_original("02_engineered_features.csv")
    targets=["yield_strength_mpa","ultimate_tensile_strength_mpa","elongation_pct"]
    feature_cols=[c for c in df.columns if c not in targets+["sample_id","material_system","alloy_or_coating_name"]]
    X_all=df[feature_cols].to_numpy(float);res=_results_json();payload={};checks=[]
    all_values=[]
    for target in targets:
        y=df[target].to_numpy(float)
        X_tr,X_te,y_tr,y_te=train_test_split(X_all,y,test_size=.2,random_state=42)
        scaler=StandardScaler().fit(X_tr);X_tr_s=scaler.transform(X_tr);X_te_s=scaler.transform(X_te)
        params=res[target]["XGBoost"]["best_params"]
        model=XGBRegressor(objective="reg:squarederror",random_state=42,n_jobs=4,**params)
        model.fit(X_tr_s,y_tr)
        pred=model.predict(X_te_s)
        reference=res[target]["XGBoost"]
        ref_y=np.asarray(reference["y_test"],float);ref_pred=np.asarray(reference["y_test_pred"],float)
        assert np.allclose(y_te,ref_y),f"XGBoost test rows differ for {target}"
        prediction_delta=float(np.max(np.abs(pred-ref_pred)))
        recomputed_r2=float(r2_score(y_te,pred));reference_r2=float(reference["test_r2"])
        r2_delta=abs(recomputed_r2-reference_r2)
        assert r2_delta<.005,(target,r2_delta)
        contrib=model.get_booster().predict(xgb.DMatrix(X_te_s),pred_contribs=True)
        sv=np.asarray(contrib[:,:-1],float);base=np.asarray(contrib[:,-1],float)
        reconstruction=float(np.max(np.abs((base+sv.sum(axis=1))-pred)))
        assert reconstruction<1e-3,(target,reconstruction)
        importance=np.abs(sv).mean(axis=0)
        payload[target]={"X_test":X_te,"y_test":y_te,"prediction":pred,"shap_values":sv,
                         "base_value":base,"importance":importance,"feature_cols":feature_cols}
        checks.append({"target":target,"prediction_max_abs_difference":prediction_delta,
                       "reference_test_r2":reference_r2,"recomputed_test_r2":recomputed_r2,
                       "test_r2_abs_difference":r2_delta,
                       "shap_reconstruction_max_abs_difference":reconstruction,
                       "test_rows":len(y_te),"features":len(feature_cols)})
        for row in range(len(y_te)):
            for j,feature in enumerate(feature_cols):
                all_values.append({"target":target,"test_row":row,"feature":feature,
                                   "feature_value":X_te[row,j],"shap_value":sv[row,j],
                                   "base_value":base[row],"prediction":pred[row]})
    pd.DataFrame(checks).to_csv(SOURCE/"recomputed_shap_validation_v15.csv",index=False)
    pd.DataFrame(all_values).to_csv(SOURCE/"recomputed_shap_values_v15.csv",index=False)
    _SHAP_CACHE=payload
    return payload


def fig3_model_validation():
    res=_results_json();ms=read_original("model_summary.csv");ladder=read("validation_ladder.csv");scr=read("y_scrambling.csv")
    fig=plt.figure(figsize=(7.2,5.0));gs=fig.add_gridspec(2,3,hspace=.45,wspace=.38)
    selected=[("yield_strength_mpa","Ridge","YS (MPa)"),("ultimate_tensile_strength_mpa","RandomForest","UTS (MPa)"),("elongation_pct","RandomForest","EL (%)")]
    src=[]
    for i,(target,model,label) in enumerate(selected):
        ax=fig.add_subplot(gs[0,i]);panel(ax,chr(ord('a')+i));clean(ax);d=res[target][model]
        ax.scatter(d["y_train"],d["y_train_pred"],s=17,c=NEUTRAL,alpha=.35,label="Training")
        ax.scatter(d["y_test"],d["y_test_pred"],s=30,c=[BLUE,TEAL,RED][i],alpha=.8,label="Test",edgecolor="white",lw=.35)
        vals=np.r_[d["y_train"],d["y_test"]];lo,hi=vals.min(),vals.max();ax.plot([lo,hi],[lo,hi],ls="--",c=DARK,lw=.8)
        ax.set_xlabel(f"Measured {label}");ax.set_ylabel(f"Predicted {label}");panel_title(ax,f"{label.split()[0]} | {model}")
        if i==0:ax.legend(frameon=False,fontsize=5.7)
        src.append(pd.DataFrame({"target":target,"model":model,"set":"test","measured":d["y_test"],"predicted":d["y_test_pred"]}))
    axd=fig.add_subplot(gs[1,0]);panel(axd,"d");clean(axd)
    piv=ms.pivot(index="model",columns="target",values="test_r2");piv.plot(kind="bar",ax=axd,color=[RED,TEAL,BLUE],width=.72);axd.set_ylabel("Test R²");axd.set_xlabel("");plt.setp(axd.get_xticklabels(),rotation=28,rotation_mode="anchor",ha="right")
    panel_title(axd,"Six-model performance")
    axd.legend(["EL","UTS","YS"],frameon=False,fontsize=5.4,ncol=3,loc="upper center",bbox_to_anchor=(.50,.96),borderaxespad=0,columnspacing=.8,handlelength=1.2)
    lo,hi=axd.get_ylim();axd.set_ylim(lo,hi+(hi-lo)*.38)
    axe=fig.add_subplot(gs[1,1]);panel(axe,"e");clean(axe)
    q=ladder.groupby(["target","regime"]).r2.agg(["mean","std"]).reset_index();reg=list(q.regime.unique());x=np.arange(len(reg));short={"Naive random holdout":"Random","Duplicate-safe grouped CV":"Grouped","Leave-one-grade-out":"Leave-grade","Leave-one-system-out":"Leave-system"}
    for i,(tar,c) in enumerate(zip(q.target.unique(),[BLUE,TEAL,RED])):
        z=q[q.target==tar].set_index("regime").reindex(reg);axe.errorbar(x+i*.07,z["mean"],yerr=z["std"],fmt="o-",ms=3,lw=.75,c=c,label={"yield_strength_mpa":"YS","ultimate_tensile_strength_mpa":"UTS","elongation_pct":"EL"}[tar])
    axe.set_xticks(x+.07,[short.get(v,v) for v in reg]);plt.setp(axe.get_xticklabels(),rotation=25,rotation_mode="anchor",ha="right");axe.set_ylabel("R²");axe.legend(frameon=False,fontsize=5.4);panel_title(axe,"Validation ladder")
    axf=fig.add_subplot(gs[1,2]);panel(axf,"f");clean(axf)
    kinds=list(scr.kind.unique())
    for i,(tar,c) in enumerate(zip(scr.target.unique(),[BLUE,TEAL,RED])):
        z=scr[scr.target==tar];vals=[z[z.kind==k].r2.values for k in kinds];pos=np.arange(len(kinds))+i*.22;bp=axf.boxplot(vals,positions=pos,widths=.17,showfliers=False,patch_artist=True,medianprops=dict(color=DARK));[p.set_facecolor(c) or p.set_alpha(.4) for p in bp["boxes"]]
    axf.set_xticks(np.arange(len(kinds))+.22,kinds);plt.setp(axf.get_xticklabels(),rotation=20,rotation_mode="anchor",ha="right");axf.set_ylabel("R²");panel_title(axf,"Label scrambling")
    save_bundle(fig,"fig3_model_performance_and_validation",[("a-c",pd.concat(src,ignore_index=True)),("d",ms),("e",ladder),("f",scr)])


def fig4_shap_interpretation():
    fig,axs=plt.subplots(2,3,figsize=(7.2,5.35))
    targets=["yield_strength_mpa","ultimate_tensile_strength_mpa","elongation_pct"]
    colors=[BLUE,TEAL,RED];labels=["YS","UTS","EL"];source_frames=[]
    for i,(target,label,signal) in enumerate(zip(targets,labels,colors)):
        importance=read_original(f"shap_importance_{target}.csv")
        importance.columns=[str(x).lstrip("\ufeff") for x in importance.columns]
        top=importance.nlargest(10,"mean_abs_shap").sort_values("mean_abs_shap")
        ax=axs[0,i];panel(ax,chr(ord("a")+i));clean(ax)
        ax.barh([feature_label(x) for x in top.feature],top.mean_abs_shap,color=signal,alpha=.82)
        ax.set_xlabel("Mean |SHAP value|");ax.tick_params(axis="y",labelsize=5.2)
        panel_title(ax,f"{label} importance")
        source_frames.append((chr(ord("a")+i),top.assign(target=target)))

    image=plt.imread(ORIGINAL_FIGURE/"fig5_shap_beeswarm.png")
    height,width=image.shape[:2];title_cut=int(height*.075)
    for i,(target,label) in enumerate(zip(targets,labels)):
        x0=round(i*width/3);x1=round((i+1)*width/3)
        crop=tight_crop(image[title_cut:,x0:x1],pad_fraction=.012)
        ax=axs[1,i];panel(ax,chr(ord("d")+i));ax.set_axis_off()
        ax.imshow(crop,aspect="auto",interpolation="lanczos")
        panel_title(ax,f"{label} SHAP distribution")
        source_frames.append((chr(ord("d")+i),pd.DataFrame({
            "target":[target],"source":["deterministic XGBoost SHAP distribution"]})))
    fig.subplots_adjust(left=.105,right=.975,top=.94,bottom=.08,wspace=.62,hspace=.45)
    save_bundle(fig,"fig4_shap_and_material_interpretation",source_frames)


def fig4_shap_importance():
    payload=recomputed_shap()
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.85))
    frames=[]
    specs=[("yield_strength_mpa","YS",BLUE),("ultimate_tensile_strength_mpa","UTS",TEAL),("elongation_pct","EL",RED)]
    for i,(target,label,signal) in enumerate(specs):
        d=payload[target];importance=pd.DataFrame({"feature":d["feature_cols"],"mean_abs_shap":d["importance"]})
        top=importance.nlargest(10,"mean_abs_shap").sort_values("mean_abs_shap")
        ax=axs[i];panel(ax,chr(ord("a")+i));clean(ax)
        ax.barh([feature_label(x) for x in top.feature],top.mean_abs_shap,color=signal,alpha=.82)
        ax.set_xlabel("Mean |SHAP value|");ax.tick_params(axis="y",labelsize=5.3)
        panel_title(ax,f"{label} importance");frames.append((chr(ord("a")+i),top.assign(target=target)))
    fig.subplots_adjust(left=.135,right=.985,top=.88,bottom=.19,wspace=.64)
    save_bundle(fig,"fig4_shap_importance",[x for x in frames])


def fig5_shap_distributions():
    payload=recomputed_shap()
    fig,axs=plt.subplots(1,3,figsize=(7.2,3.15));frames=[]
    specs=[("yield_strength_mpa","YS"),("ultimate_tensile_strength_mpa","UTS"),("elongation_pct","EL")]
    for i,(target,label) in enumerate(specs):
        d=payload[target];sv=d["shap_values"];X=d["X_test"];importance=d["importance"];features=d["feature_cols"]
        order=np.argsort(importance)[::-1][:8][::-1];ax=axs[i];panel(ax,chr(ord("a")+i));clean(ax);long=[]
        for row,j in enumerate(order):
            vals=sv[:,j];fvals=X[:,j];den=float(np.max(fvals)-np.min(fvals))
            norm=(fvals-np.min(fvals))/(den+1e-12)
            rank=np.argsort(vals);jitter=np.empty(len(vals),float);jitter[rank]=np.linspace(-.27,.27,len(vals))
            ax.scatter(vals,row+jitter,c=norm,cmap="coolwarm",vmin=0,vmax=1,
                       s=25,alpha=.84,edgecolor="white",lw=.18,zorder=3)
            long.append(pd.DataFrame({"target":target,"test_row":np.arange(len(vals)),
                "feature":features[j],"feature_value":fvals,"shap_value":vals}))
        ax.axvline(0,color=NEUTRAL,lw=.7,ls="--")
        ax.set_yticks(range(len(order)),[feature_label(features[j]) for j in order])
        ax.tick_params(axis="y",labelsize=5.3);ax.set_xlabel("SHAP value")
        panel_title(ax,f"{label} SHAP distribution")
        frames.append((chr(ord("a")+i),pd.concat(long,ignore_index=True)))
    fig.subplots_adjust(left=.12,right=.985,top=.88,bottom=.16,wspace=.66)
    save_bundle(fig,"fig5_shap_distributions",frames)


def fig5_symbolic_regression():
    sy=read_original("sr_yield_strength_mpa_global_front.csv")
    su=read_original("sr_ultimate_tensile_strength_mpa_global_front.csv")
    fig=plt.figure(figsize=(7.2,5.0));gs=fig.add_gridspec(2,2,hspace=.40,wspace=.34)
    ax=fig.add_subplot(gs[0,0]);panel(ax,"a");clean(ax)
    for frame,label,c in [(sy,"YS",BLUE),(su,"UTS",TEAL)]:
        q=frame.copy();q["loss"]=pd.to_numeric(q.loss,errors="coerce")
        q=q[np.isfinite(q.loss)&(q.loss>0)].sort_values("complexity")
        ax.plot(q.complexity,np.log10(q.loss),"o-",ms=3,lw=.85,color=c,label=label,alpha=.85)
    ax.set_xlabel("Formula complexity");ax.set_ylabel("log10(loss)")
    panel_title(ax,"Complexity–loss front");ax.legend(frameon=False)
    parity_sources=[];error_sets=[];res=_results_json()
    for j,(target,label,c) in enumerate([
        ("yield_strength_mpa","YS (MPa)",BLUE),
        ("ultimate_tensile_strength_mpa","UTS (MPa)",TEAL)]):
        axp=fig.add_subplot(gs[0,1] if j==0 else gs[1,0]);panel(axp,chr(ord("b")+j));clean(axp)
        ytr=np.loadtxt(ORIGINAL_DATA/f"sr_{target}_y_tr.csv",delimiter=",")
        ptr=np.loadtxt(ORIGINAL_DATA/f"sr_{target}_pred_tr.csv",delimiter=",")
        yte=np.loadtxt(ORIGINAL_DATA/f"sr_{target}_y_te.csv",delimiter=",")
        pte=np.loadtxt(ORIGINAL_DATA/f"sr_{target}_pred_te.csv",delimiter=",")
        axp.scatter(ytr,ptr,s=16,c=NEUTRAL,alpha=.35,label="Training")
        axp.scatter(yte,pte,s=30,c=c,alpha=.82,edgecolor="white",lw=.35,label="Test")
        lo=min(ytr.min(),yte.min(),ptr.min(),pte.min());hi=max(ytr.max(),yte.max(),ptr.max(),pte.max())
        axp.plot([lo,hi],[lo,hi],ls="--",c=DARK,lw=.8)
        axp.set_xlabel(f"Measured {label}");axp.set_ylabel(f"Predicted {label}")
        panel_title(axp,f"{label.split()[0]} empirical formula")
        if j==0:axp.legend(frameon=False,fontsize=5.7)
        parity_sources.append(pd.DataFrame({"target":target,"set":"train","measured":ytr,"predicted":ptr}))
        parity_sources.append(pd.DataFrame({"target":target,"set":"test","measured":yte,"predicted":pte}))
        selected="Ridge" if target=="yield_strength_mpa" else "RandomForest"
        d=res[target][selected]
        model_y=np.asarray(d["y_test"],dtype=float);model_pred=np.asarray(d["y_test_pred"],dtype=float)
        assert np.allclose(yte,model_y),f"SR and selected-model test rows differ for {target}"
        error_sets.extend([
            pd.DataFrame({"target":label.split()[0],"method":"SR","absolute_error":np.abs(pte-yte)}),
            pd.DataFrame({"target":label.split()[0],"method":"Selected model","absolute_error":np.abs(model_pred-model_y)})])
    axd=fig.add_subplot(gs[1,1]);panel(axd,"d");clean(axd)
    errors=pd.concat(error_sets,ignore_index=True)
    combos=[("YS","SR"),("YS","Selected model"),("UTS","SR"),("UTS","Selected model")]
    arrays=[errors[(errors.target==t)&(errors.method==m)].absolute_error.to_numpy() for t,m in combos]
    bp=axd.boxplot(arrays,positions=np.arange(4),widths=.52,patch_artist=True,showfliers=False,
                   medianprops=dict(color=DARK,lw=1),whiskerprops=dict(color=NEUTRAL),
                   capprops=dict(color=NEUTRAL))
    for patch,c in zip(bp["boxes"],[NEUTRAL,BLUE,NEUTRAL,TEAL]):
        patch.set_facecolor(c);patch.set_alpha(.58);patch.set_edgecolor(DARK)
    for i,(arr,c) in enumerate(zip(arrays,[NEUTRAL,BLUE,NEUTRAL,TEAL])):
        axd.scatter(i+np.linspace(-.10,.10,num=len(arr)),arr,s=16,color=c,alpha=.58,edgecolor="none")
    axd.set_xticks(range(4),["YS\nSR","YS\nRidge","UTS\nSR","UTS\nRF"])
    axd.set_ylabel("Test absolute error (MPa)");panel_title(axd,"Test-set absolute error")
    save_bundle(fig,"fig5_symbolic_regression",[("a",pd.concat([sy.assign(target="YS"),su.assign(target="UTS")],ignore_index=True)),("b-c",pd.concat(parity_sources,ignore_index=True)),("d",errors)])


def si_full_model_figures():
    res=_results_json()
    mapping=[("yield_strength_mpa","YS (MPa)","figS13_full_models_ys",BLUE),
             ("ultimate_tensile_strength_mpa","UTS (MPa)","figS14_full_models_uts",TEAL),
             ("elongation_pct","EL (%)","figS15_full_models_el",RED)]
    order=["Ridge","XGBoost","RandomForest","SVR","GPR","MLP"]
    for target,label,stem,signal in mapping:
        fig,axs=plt.subplots(2,3,figsize=(7.2,4.45));frames=[]
        for i,(ax,model) in enumerate(zip(axs.flat,order)):
            d=res[target][model];panel(ax,chr(ord("a")+i));clean(ax)
            ax.scatter(d["y_train"],d["y_train_pred"],s=14,c=NEUTRAL,alpha=.32,label="Training")
            ax.scatter(d["y_test"],d["y_test_pred"],s=27,c=signal,alpha=.82,edgecolor="white",lw=.3,label="Test")
            vals=np.r_[d["y_train"],d["y_test"],d["y_train_pred"],d["y_test_pred"]]
            lo,hi=float(vals.min()),float(vals.max());ax.plot([lo,hi],[lo,hi],ls="--",c=DARK,lw=.75)
            ax.set_xlabel(f"Measured {label}");ax.set_ylabel(f"Predicted {label}")
            ax.set_title(model,loc="left",fontweight="bold")
            if i==0:ax.legend(frameon=False,fontsize=5.4)
            frames.append(pd.DataFrame({"target":target,"model":model,"set":"train","measured":d["y_train"],"predicted":d["y_train_pred"]}))
            frames.append(pd.DataFrame({"target":target,"model":model,"set":"test","measured":d["y_test"],"predicted":d["y_test_pred"]}))
        fig.subplots_adjust(hspace=.44,wspace=.36)
        save_bundle(fig,stem,[("all",pd.concat(frames,ignore_index=True))],si=True)


def fig1():
    fig = plt.figure(figsize=(7.2, 4.25))
    gs = fig.add_gridspec(2, 12, height_ratios=[1.7, 1], hspace=0.32)
    ax = fig.add_subplot(gs[0, :]); ax.set_axis_off(); panel(ax, "a")
    steps = [
        ("78 accumulated\ntest records", "composition + process\n+ tensile response"),
        ("Material indicators", "Al_eq, Tβ,est,\nTroll − Tβ,est"),
        ("Grade-aware learning\n+ optimization", "search inside the\nmeasured data range"),
        ("Pre-test freeze", "Ti_9 and Ti_20\nroutes locked"),
        ("New manufacture", "remelt → process\n→ independent tensile test"),
        ("Experimental\nfeedback", "measured properties\nand process records"),
    ]
    x = np.linspace(0.055, 0.945, len(steps)); y = 0.58
    for i, ((title, sub), xi) in enumerate(zip(steps, x)):
        signal = i in (3, 4)
        box = FancyBboxPatch((xi-0.075, y-0.18), 0.15, 0.36,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            fc="#FFF5DF" if signal else "#F1F4F6", ec=GOLD if signal else NEUTRAL,
            lw=1.2 if signal else 0.8)
        ax.add_patch(box)
        ax.text(xi, y+0.055, title, ha="center", va="center", fontsize=7.4,
                fontweight="bold", color=DARK)
        ax.text(xi, y-0.09, sub, ha="center", va="center", fontsize=6.2, color="#56616B")
        if i < len(steps)-1:
            ax.add_patch(FancyArrowPatch((xi+0.078, y), (x[i+1]-0.078, y),
                         arrowstyle="-|>", mutation_scale=8, lw=0.85, color=BLUE))
    ax.add_patch(FancyArrowPatch((x[-1], y-0.2), (x[1], y-0.2),
                 connectionstyle="arc3,rad=-0.22", arrowstyle="-|>", mutation_scale=8,
                 lw=0.85, color=TEAL))
    ax.text(0.50, 0.08, "feedback informs the next experiment",
            ha="center", va="center", fontsize=6.3, color=TEAL)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    axb = fig.add_subplot(gs[1, :7]); panel(axb, "b")
    axb.set_axis_off()
    axb.text(0.00, 0.95, "Evidence chain", fontsize=8.0, fontweight="bold", color=DARK)
    levels = [("OBSERVED", "process–microstructure combinations"),
              ("SUPPORTED", "within-grade strength–ductility association"),
              ("MECHANISM QUESTION", "phase balance and slip compatibility"),
              ("NEXT TEST", "controlled T/Tβ and aging matrices")]
    for i, (head, body) in enumerate(levels):
        yy = 0.72 - i*0.19
        axb.text(0.0, yy, head, fontsize=6.2, fontweight="bold", color=[NEUTRAL, BLUE, RED, GOLD][i])
        axb.text(0.21, yy, body, fontsize=6.8, color=DARK)
        if i < 3: axb.plot([0.055, 0.055], [yy-0.145, yy-0.045], color=PALE, lw=2)
    axb.set_xlim(0, 1); axb.set_ylim(0, 1)

    axc = fig.add_subplot(gs[1, 8:]); panel(axc, "c")
    axc.set_axis_off(); axc.text(0.0, 0.95, "Independent trials", fontsize=8.0, fontweight="bold", color=DARK)
    for i, (sid, grade, route) in enumerate([
        ("Ti_9", "TC4", "forge–roll–anneal"), ("Ti_20", "Ti6411", "roll–solution–age")]):
        yy = 0.63 - i*0.34
        axc.scatter([0.08], [yy], s=95, marker="*", c=GOLD, edgecolor=DARK, linewidth=0.55)
        axc.text(0.18, yy+0.04, f"{sid}  |  {grade}", fontweight="bold", fontsize=7.4, color=DARK)
        axc.text(0.18, yy-0.08, route, fontsize=6.4, color="#56616B")
    axc.set_xlim(0, 1); axc.set_ylim(0, 1)
    src = pd.DataFrame({"sequence": np.arange(1, 7), "stage": [s[0].replace("\n", " ") for s in steps],
                        "detail": [s[1].replace("\n", " ") for s in steps]})
    save_bundle(fig, "fig1_closed_loop", [("a", src)])


def fig2():
    h = read("historical_78_material_states.csv"); t = read("independent_trials_2.csv")
    fig = plt.figure(figsize=(7.2, 5.2)); gs = fig.add_gridspec(2, 3, width_ratios=[1.45, 1, 1], hspace=0.42, wspace=0.38)
    ax = fig.add_subplot(gs[:, 0]); panel(ax, "a"); clean(ax)
    for grade, g in h.groupby("alloy_or_coating_name"):
        ax.scatter(g["Al_eq"], g["Tbeta_est_c"], s=31, alpha=.72, color=GRADE_COLORS.get(grade, NEUTRAL), label=grade,
                   edgecolors="white", linewidth=.35)
    for _, r in t.iterrows():
        ax.scatter(r.Al_eq, r.Tbeta_est_c, s=115, marker="*", c=GOLD, edgecolor=DARK, linewidth=.7, zorder=5)
        ax.annotate(r.sample_id, (r.Al_eq, r.Tbeta_est_c), xytext=(4,4), textcoords="offset points", fontsize=6.6, fontweight="bold")
    ax.set_xlabel("Empirical Al equivalent (wt%)"); ax.set_ylabel("Estimated β-transus (°C)")
    ax.legend(frameon=False, loc="lower right", ncol=1, handletextpad=.3, borderpad=.1)
    ax.set_title("Phase-stability map", loc="left", fontweight="bold")

    axb = fig.add_subplot(gs[0, 1:]); panel(axb, "b"); clean(axb)
    hh = h.dropna(subset=["delta_T_roll_c"])
    for grade, g in hh.groupby("alloy_or_coating_name"):
        axb.scatter(g["delta_T_roll_c"], g["elongation_pct"], s=30, alpha=.72,
                    color=GRADE_COLORS.get(grade, NEUTRAL), edgecolors="white", linewidth=.35)
    axb.axvline(0, color=DARK, lw=.8, ls="--")
    axb.text(0.02, .93, "sub-transus", transform=axb.transAxes, color=BLUE, fontsize=6.4)
    axb.text(.72, .93, "super-transus", transform=axb.transAxes, color=RED, fontsize=6.4)
    axb.set_xlabel("Troll − Tβ,est (°C)"); axb.set_ylabel("Elongation (%)")
    axb.set_title("Processing position relative to β-transus", loc="left", fontweight="bold")

    axc = fig.add_subplot(gs[1, 1]); panel(axc, "c"); clean(axc)
    for grade in ["TA1", "TA2"]:
        g = h[h.alloy_or_coating_name == grade]
        axc.scatter(g.O_wt_pct, g.elongation_pct, s=34, color=GRADE_COLORS[grade], alpha=.78, label=grade)
        if len(g) > 2:
            p=np.polyfit(g.O_wt_pct, g.elongation_pct, 1); xx=np.linspace(g.O_wt_pct.min(),g.O_wt_pct.max(),60)
            axc.plot(xx, np.polyval(p,xx), color=GRADE_COLORS[grade], lw=1)
    axc.set_xlabel("O (wt%)"); axc.set_ylabel("Elongation (%)"); axc.legend(frameon=False)
    axc.set_title("Interstitial–ductility", loc="left", fontweight="bold", fontsize=7.2)

    axd = fig.add_subplot(gs[1, 2]); panel(axd, "d"); clean(axd)
    for grade, g in h.groupby("alloy_or_coating_name"):
        axd.scatter(g.elongation_pct, g.yield_strength_mpa, s=27, alpha=.55,
                    color=GRADE_COLORS.get(grade, NEUTRAL), edgecolors="white", linewidth=.25)
    for _, r in t.iterrows():
        axd.scatter(r.elongation_pct, r.yield_strength_mpa, s=115, marker="*", c=GOLD, edgecolor=DARK, lw=.7, zorder=5)
        axd.annotate(r.sample_id, (r.elongation_pct,r.yield_strength_mpa), xytext=(3,4), textcoords="offset points", fontsize=6.3, fontweight="bold")
    axd.set_xlabel("Elongation (%)"); axd.set_ylabel("Yield strength (MPa)")
    axd.set_title("Strength–ductility map", loc="left", fontweight="bold", fontsize=7.2)
    save_bundle(fig, "fig2_physical_landscape", [("a", h[["sample_id","alloy_or_coating_name","Al_eq","Tbeta_est_c"]]),
        ("b", hh[["sample_id","alloy_or_coating_name","delta_T_roll_c","elongation_pct"]]),
        ("c", h[h.alloy_or_coating_name.isin(["TA1","TA2"])][["sample_id","alloy_or_coating_name","O_wt_pct","elongation_pct"]]),
        ("d", pd.concat([h.assign(dataset="accumulated_test_data"),t.assign(dataset="subsequent_trial")])[["dataset","sample_id","alloy_or_coating_name","yield_strength_mpa","elongation_pct"]])])


def fig3():
    g=read("tc4_historical_group_means.csv"); comp=read("table2_tc4_process_microstructure_packages.csv")
    g["package_short"] = g["package"].map({"A_1050_duplex_low_alpha":"A", "B_980_equiaxed_high_alpha":"B"})
    fig=plt.figure(figsize=(7.2,4.9)); gs=fig.add_gridspec(2,3,width_ratios=[1.5,1,1],hspace=.42,wspace=.42)
    ax=fig.add_subplot(gs[:,0]); panel(ax,"a"); clean(ax)
    for package, color, marker in [("A",BLUE,"o"),("B",TEAL,"s")]:
        q=g[g.package_short==package]
        ax.scatter(q.elongation_pct,q.yield_strength_mpa,s=30,c=color,marker=marker,alpha=.35,
                   edgecolor="white",lw=.45,label=f"Condition {package} (n={len(q)})")
        covariance_ellipse(ax,q.elongation_pct,q.yield_strength_mpa,color,n_std=1.35,alpha=.14)
    ax.set_xlabel("Elongation (%)"); ax.set_ylabel("Yield strength (MPa)"); ax.legend(frameon=False,loc="lower left")
    panel_title(ax,"TC4 group-level map")
    for a,b in [((10,850),(16,880)),((17,875),(20,900))]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="->",mutation_scale=8,lw=.7,color=NEUTRAL))

    metrics=[("yield_strength_mpa","YS (MPa)","Yield strength"),
             ("elongation_pct","EL (%)","Elongation"),
             ("yield_ratio","YS/UTS","Yield ratio"),
             ("hardening_reserve_mpa","UTS−YS (MPa)","Hardening reserve")]
    for i,(metric,label,title) in enumerate(metrics):
        aa=fig.add_subplot(gs[i//2,1+i%2]); panel(aa,chr(ord('b')+i)); clean(aa)
        vals=[g[g.package_short==p][metric].dropna().values for p in ["A","B"]]
        aa.boxplot(vals,positions=[0,1],widths=.45,patch_artist=True,showfliers=False,
                   boxprops=dict(facecolor=PALE,edgecolor=DARK,lw=.75),medianprops=dict(color=DARK,lw=1),
                   whiskerprops=dict(color=NEUTRAL),capprops=dict(color=NEUTRAL))
        for x0,(arr,c) in enumerate(zip(vals,[BLUE,TEAL])):
            jitter=np.linspace(-.07,.07,len(arr)); aa.scatter(x0+jitter,arr,s=32,c=c,edgecolor="white",lw=.35,zorder=3)
        row=comp[comp.metric==metric]
        lo,hi=aa.get_ylim();aa.set_ylim(lo,hi+(hi-lo)*.38)
        if len(row):
            r=row.iloc[0]; aa.text(.98,.965,f"Δ = {r.B_minus_A:.2f}  |  95% CI [{r.ci95_low:.2f}, {r.ci95_high:.2f}]",transform=aa.transAxes,ha="right",va="top",fontsize=5.45,color=DARK)
        aa.set_xticks([0,1],["A\n1050 °C","B\n980 °C"]); aa.set_ylabel(label)
        panel_title(aa,title)
    fig.subplots_adjust(left=.09,right=.98,top=.93,bottom=.12,wspace=.48,hspace=.52)
    save_bundle(fig,"fig3_tc4_packages",[("a",g),("b-e",comp)])


def fig4():
    v=read("table3_trial_predictions.csv"); t=read("independent_trials_2.csv"); m=read("trial_matched_controls.csv")
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.1));axs=axs.ravel()
    targets=[("yield_strength_mpa","YS (MPa)"),("ultimate_tensile_strength_mpa","UTS (MPa)"),("elongation_pct","EL (%)")]
    for i,(tar,label) in enumerate(targets):
        aa=axs[i];panel(aa,chr(ord('a')+i));clean(aa)
        q=v[v.target==tar].set_index("sample_id").loc[["Ti_9","Ti_20"]]
        x=np.arange(2); w=.32
        aa.bar(x-w/2,q.selected_prediction,w,color=BLUE,label="Predicted")
        aa.bar(x+w/2,q.measured,w,color=GOLD,label="Measured")
        aa.errorbar(x-w/2,q.selected_prediction,yerr=1.96*q.cv_rmse,fmt="none",ecolor=DARK,capsize=2,lw=.7)
        aa.set_xticks(x,["Ti_9","Ti_20"]); aa.set_ylabel(label)
        panel_title(aa,{"yield_strength_mpa":"Yield strength","ultimate_tensile_strength_mpa":"Ultimate tensile strength","elongation_pct":"Elongation"}[tar])
        ymax=max((q.selected_prediction+1.96*q.cv_rmse).max(),q.measured.max())
        ymin=min((q.selected_prediction-1.96*q.cv_rmse).min(),q.measured.min())
        span=max(1.0,ymax-ymin);aa.set_ylim(max(0,ymin-.10*span),ymax+.22*span)
        for xx,val in zip(x,q.measured):
            aa.text(xx,val+.025*span,f"{val:.0f}" if tar!="elongation_pct" else f"{val:.0f}%",
                    ha="center",va="bottom",fontsize=5.8,fontweight="bold")

    axe=axs[3];panel(axe,"d");clean(axe)
    for grade,color in [("Ti6411",RED),("TC4",TEAL)]:
        q=m[m.alloy_or_coating_name==grade]
        axe.scatter(q.elongation_pct,q.yield_strength_mpa,s=18,c=color,alpha=.22,label=f"{grade} controls")
        covariance_ellipse(axe,q.elongation_pct,q.yield_strength_mpa,color,n_std=1.5,alpha=.12)
    markers={"Ti_9":"*","Ti_20":"D"}
    for _,r in t.iterrows():
        axe.scatter(r.elongation_pct,r.yield_strength_mpa,s=115 if r.sample_id=="Ti_9" else 58,
                    c=GOLD,marker=markers[r.sample_id],edgecolor=DARK,lw=.7,label=r.sample_id,zorder=5)
    axe.set_xlabel("Elongation (%)");axe.set_ylabel("YS (MPa)")
    axe.legend(frameon=False,fontsize=5.5,ncol=2,loc="best")
    panel_title(axe,"Within-grade trial position")
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,ncol=2,loc="upper center",bbox_to_anchor=(.5,.985),
               fontsize=5.8,columnspacing=1.1)
    fig.subplots_adjust(left=.10,right=.97,top=.89,bottom=.10,wspace=.34,hspace=.42)
    save_bundle(fig,"fig4_independent_trials",[("a-c",v),("d",pd.concat([
        m[["sample_id","alloy_or_coating_name","yield_strength_mpa","elongation_pct"]].assign(dataset="controls"),
        t[["sample_id","alloy_or_coating_name","yield_strength_mpa","elongation_pct"]].assign(dataset="trial")
    ],ignore_index=True))])


def fig6():
    """Grade-specific fronts, observed cross-grade envelope and locked candidates."""
    pareto=read("historical_local_pareto.csv")
    h=read("historical_78_material_states.csv")
    t=read("independent_trials_2.csv")
    fig=plt.figure(figsize=(7.2,5.60))
    gs=fig.add_gridspec(2,4,height_ratios=[1.0,.85],hspace=.32,wspace=.52)
    trial_markers={"Ti_9":"*","Ti_20":"D"}

    ax=fig.add_subplot(gs[0,0:2]);panel(ax,"a");clean(ax)
    ax.scatter(h.elongation_pct,h.yield_strength_mpa,s=11,c=NEUTRAL,alpha=.18,label="Data")
    for grade,gp in pareto[pareto.front=="YS_EL"].groupby("grade"):
        gp=gp.sort_values("y");ax.plot(gp.y,gp.x,"o-",ms=3,lw=.9,color=GRADE_COLORS.get(grade,NEUTRAL),label=grade)
    for _,r in t.iterrows():
        ax.scatter(r.elongation_pct,r.yield_strength_mpa,s=90 if r.sample_id=="Ti_9" else 46,
                   marker=trial_markers[r.sample_id],c=GOLD,edgecolor=DARK,lw=.65,zorder=5,label=r.sample_id)
    ax.set_xlabel("Elongation, EL (%)");ax.set_ylabel("Yield strength, YS (MPa)")
    panel_title(ax,"Within-grade YS–EL fronts")

    axb=fig.add_subplot(gs[0,2:4]);panel(axb,"b");clean(axb)
    axb.scatter(h.elongation_pct,h.ultimate_tensile_strength_mpa,s=11,c=NEUTRAL,alpha=.18)
    for grade,gp in pareto[pareto.front=="UTS_EL"].groupby("grade"):
        gp=gp.sort_values("y");axb.plot(gp.y,gp.x,"o-",ms=3,lw=.9,color=GRADE_COLORS.get(grade,NEUTRAL))
    for _,r in t.iterrows():
        axb.scatter(r.elongation_pct,r.ultimate_tensile_strength_mpa,s=90 if r.sample_id=="Ti_9" else 46,
                    marker=trial_markers[r.sample_id],c=GOLD,edgecolor=DARK,lw=.65,zorder=5)
    axb.set_xlabel("Elongation, EL (%)");axb.set_ylabel("Ultimate tensile strength, UTS (MPa)")
    panel_title(axb,"Within-grade UTS–EL fronts")

    axc=fig.add_subplot(gs[1,0:2]);panel(axc,"c");clean(axc)
    front_ys=nondominated_frame(h,"yield_strength_mpa").assign(front="YS–EL")
    front_uts=nondominated_frame(h,"ultimate_tensile_strength_mpa").assign(front="UTS–EL")
    axc.plot(front_ys.elongation_pct,front_ys.yield_strength_mpa,"-",color=BLUE,lw=1.15,label="Observed YS–EL envelope")
    axc.plot(front_uts.elongation_pct,front_uts.ultimate_tensile_strength_mpa,"--",color=TEAL,lw=1.15,label="Observed UTS–EL envelope")
    for _,r in front_ys.iterrows():
        axc.scatter(r.elongation_pct,r.yield_strength_mpa,s=28,color=GRADE_COLORS.get(r.alloy_or_coating_name,NEUTRAL),edgecolor="white",lw=.35,zorder=4)
    for _,r in front_uts.iterrows():
        axc.scatter(r.elongation_pct,r.ultimate_tensile_strength_mpa,s=28,marker="s",color=GRADE_COLORS.get(r.alloy_or_coating_name,NEUTRAL),edgecolor="white",lw=.35,zorder=4)
    axc.set_xlabel("Elongation, EL (%)");axc.set_ylabel("Strength (MPa)")
    axc.legend(frameon=False,fontsize=5.4,loc="best");panel_title(axc,"Observed cross-grade envelope")

    axd=fig.add_subplot(gs[1,2]);panel(axd,"d");axd.set_axis_off();panel_title(axd,"Locked candidates")
    cards=[("Ti_9 | TC4","VAR → forge/roll → anneal",TEAL),("Ti_20 | Ti6411","EB → roll → solution/age",RED)]
    for i,(head,body,c) in enumerate(cards):
        y=.67-i*.38;axd.add_patch(FancyBboxPatch((.03,y-.12),.92,.25,boxstyle="round,pad=.012",fc="#F4F6F7",ec=c,lw=.9))
        axd.text(.08,y+.04,head,fontweight="bold",fontsize=6.5,color=DARK)
        axd.text(.08,y-.055,body,fontsize=5.4,color="#56616B")
    axd.set_xlim(0,1);axd.set_ylim(0,1)

    axe=fig.add_subplot(gs[1,3]);panel(axe,"e");axe.set_axis_off();panel_title(axe,"Design workflow")
    steps=[("Search",BLUE),("Lock",TEAL),("Make",GOLD),("Test",RED)]
    for i,(txt,c) in enumerate(steps):
        y=.78-i*.21
        axe.add_patch(FancyBboxPatch((.20,y-.07),.62,.13,boxstyle="round,pad=.012",fc="#F4F6F7",ec=c,lw=.9))
        axe.text(.51,y,txt,ha="center",va="center",fontsize=6.4,fontweight="bold",color=DARK)
        if i<3:axe.annotate("",(.51,y-.16),(.51,y-.08),arrowprops=dict(arrowstyle="->",lw=.8,color=NEUTRAL))
    axe.set_xlim(0,1);axe.set_ylim(0,1)

    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.50,.985),ncol=4,
               frameon=False,fontsize=5.2,columnspacing=.8,handletextpad=.35)
    fig.subplots_adjust(top=.88,bottom=.10,left=.09,right=.98,wspace=.52,hspace=.32)
    combined=pd.concat([
        front_ys[["sample_id","alloy_or_coating_name","elongation_pct","yield_strength_mpa","front"]],
        front_uts[["sample_id","alloy_or_coating_name","elongation_pct","ultimate_tensile_strength_mpa","front"]]
    ],ignore_index=True,sort=False)
    save_bundle(fig,"fig6_optimization_and_trial_fronts",[("a-b",pareto),("c",combined),("d",t),("e",pd.DataFrame({"stage":["Search","Lock","Make","Test"]}))])


def fig6_v13():
    """V11 layout with observed cross-grade envelopes overlaid in panels a and b."""
    pareto=read("historical_local_pareto.csv")
    h=read("historical_78_material_states.csv")
    t=read("independent_trials_2.csv")
    front_ys=nondominated_frame(h,"yield_strength_mpa").assign(front="Global YS–EL")
    front_uts=nondominated_frame(h,"ultimate_tensile_strength_mpa").assign(front="Global UTS–EL")
    fig=plt.figure(figsize=(7.2,4.00))
    gs=fig.add_gridspec(2,3,width_ratios=[1.15,1.15,.9],hspace=.42,wspace=.42)
    trial_markers={"Ti_9":"*","Ti_20":"D"}

    ax=fig.add_subplot(gs[:,0]);panel(ax,"a");clean(ax)
    ax.scatter(h.elongation_pct,h.yield_strength_mpa,s=15,c=NEUTRAL,alpha=.28,label="Data")
    for grade,gp in pareto[pareto.front=="YS_EL"].groupby("grade"):
        gp=gp.sort_values("y");ax.plot(gp.y,gp.x,"o-",ms=3,lw=.85,color=GRADE_COLORS.get(grade,NEUTRAL),label=grade)
    ax.plot(front_ys.elongation_pct,front_ys.yield_strength_mpa,"o-",ms=4.1,lw=1.35,color=RED,label="Observed global envelope",zorder=5)
    for _,r in t.iterrows():
        ax.scatter(r.elongation_pct,r.yield_strength_mpa,s=95 if r.sample_id=="Ti_9" else 50,
                   marker=trial_markers[r.sample_id],c=GOLD,edgecolor=DARK,lw=.65,zorder=6,label=r.sample_id)
    ax.set_xlabel("Elongation, EL (%)");ax.set_ylabel("Yield strength, YS (MPa)")
    panel_title(ax,"YS–EL fronts")

    axb=fig.add_subplot(gs[:,1]);panel(axb,"b");clean(axb)
    axb.scatter(h.elongation_pct,h.ultimate_tensile_strength_mpa,s=15,c=NEUTRAL,alpha=.28)
    for grade,gp in pareto[pareto.front=="UTS_EL"].groupby("grade"):
        gp=gp.sort_values("y");axb.plot(gp.y,gp.x,"o-",ms=3,lw=.85,color=GRADE_COLORS.get(grade,NEUTRAL))
    axb.plot(front_uts.elongation_pct,front_uts.ultimate_tensile_strength_mpa,"o-",ms=4.1,lw=1.35,color=RED,zorder=5)
    for _,r in t.iterrows():
        axb.scatter(r.elongation_pct,r.ultimate_tensile_strength_mpa,s=95 if r.sample_id=="Ti_9" else 50,
                    marker=trial_markers[r.sample_id],c=GOLD,edgecolor=DARK,lw=.65,zorder=6)
    axb.set_xlabel("Elongation, EL (%)");axb.set_ylabel("Ultimate tensile strength, UTS (MPa)")
    panel_title(axb,"UTS–EL fronts")

    axc=fig.add_subplot(gs[0,2]);panel(axc,"c");axc.set_axis_off();panel_title(axc,"Locked candidates")
    cards=[("Ti_9 | TC4","VAR → forge/roll → anneal",TEAL),("Ti_20 | Ti6411","EB → roll → solution/age",RED)]
    for i,(head,body,c) in enumerate(cards):
        y=.68-i*.38;axc.add_patch(FancyBboxPatch((.03,y-.12),.92,.25,boxstyle="round,pad=.012",fc="#F4F6F7",ec=c,lw=.9))
        axc.text(.08,y+.04,head,fontweight="bold",fontsize=6.7,color=DARK)
        axc.text(.08,y-.055,body,fontsize=5.4,color="#56616B")
    axc.set_xlim(0,1);axc.set_ylim(0,1)

    axd=fig.add_subplot(gs[1,2]);panel(axd,"d");axd.set_axis_off();panel_title(axd,"Design workflow")
    steps=[("Search",BLUE),("Lock",TEAL),("Make",GOLD),("Test",RED)]
    for i,(txt,c) in enumerate(steps):
        y=.78-i*.21;axd.add_patch(FancyBboxPatch((.20,y-.07),.62,.13,boxstyle="round,pad=.012",fc="#F4F6F7",ec=c,lw=.9))
        axd.text(.51,y,txt,ha="center",va="center",fontsize=6.5,fontweight="bold",color=DARK)
        if i<3:axd.annotate("",(.51,y-.16),(.51,y-.08),arrowprops=dict(arrowstyle="->",lw=.8,color=NEUTRAL))
    axd.set_xlim(0,1);axd.set_ylim(0,1)

    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.50,.99),ncol=5,
               frameon=False,fontsize=5.0,columnspacing=.65,handletextpad=.3)
    fig.subplots_adjust(top=.85,bottom=.13,left=.09,right=.98,wspace=.44,hspace=.45)
    global_source=pd.concat([
        front_ys[["sample_id","alloy_or_coating_name","elongation_pct","yield_strength_mpa","front"]],
        front_uts[["sample_id","alloy_or_coating_name","elongation_pct","ultimate_tensile_strength_mpa","front"]]
    ],ignore_index=True,sort=False)
    save_bundle(fig,"fig6_optimization_and_trial_fronts",[("a-b_local",pareto),("a-b_global",global_source),("c",t),("d",pd.DataFrame({"stage":["Search","Lock","Make","Test"]}))])


def si_figures():
    h=read("historical_78_material_states.csv"); v=read("table3_trial_predictions.csv")
    # S1 record structure and isolation
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.55));
    counts=pd.DataFrame({"set":["Historical records","Trial records","Condition groups"],"n":[len(h),2,h.replicate_group.nunique()]})
    axs[0].bar(counts["set"],counts.n,color=[BLUE,GOLD,TEAL]);axs[0].set_ylabel("Count");panel(axs[0],"a");clean(axs[0]);plt.setp(axs[0].get_xticklabels(),rotation=22,rotation_mode="anchor",ha="right")
    panel_title(axs[0],"Record structure")
    dup=h.groupby("replicate_group").size().value_counts().sort_index();axs[1].bar(dup.index.astype(str),dup.values,color=NEUTRAL);axs[1].set_xlabel("Records per replicate group");axs[1].set_ylabel("Groups");panel(axs[1],"b");clean(axs[1])
    panel_title(axs[1],"Repeated-input groups")
    ys=h.yield_strength_mpa;uts=h.ultimate_tensile_strength_mpa;axs[2].scatter(ys,uts,s=25,c=GRADE_COLORS.get("TA1"),alpha=.55);lo=min(ys.min(),uts.min());hi=max(ys.max(),uts.max());axs[2].plot([lo,hi],[lo,hi],ls="--",c=DARK,lw=.8);axs[2].set_xlabel("YS (MPa)");axs[2].set_ylabel("UTS (MPa)");panel(axs[2],"c");clean(axs[2])
    panel_title(axs[2],"YS–UTS consistency")
    dup_frame = dup.rename_axis("records_per_group").rename("group_count").reset_index()
    fig.subplots_adjust(left=.10,right=.96,bottom=.22,top=.90,wspace=.38)
    save_bundle(fig,"figS1_data_audit",[("a",counts),("b",dup_frame),("c",h[["sample_id","yield_strength_mpa","ultimate_tensile_strength_mpa"]])],si=True)

    # S2 model benchmark
    ms=read("grouped_model_summary.csv",data=True); fi=read("factor_increment_cv.csv")
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.0));
    piv=ms.pivot(index="model",columns="target",values="cv_r2_mean");piv.plot(kind="bar",ax=axs[0],color=[BLUE,TEAL,RED],width=.75);plt.setp(axs[0].get_xticklabels(), rotation_mode="anchor", ha="right");axs[0].set_ylabel("CV R²");axs[0].legend(frameon=False,fontsize=5.7);panel(axs[0],"a");clean(axs[0])
    panel_title(axs[0],"Six-model CV performance")
    q=fi.groupby(["model","target"]).mae_improvement_pct.mean().reset_index();
    factor_order=list(fi.model.unique()); factor_labels={x:("Grade" if x=="grade" else "+"+x.replace("grade + ","")) for x in factor_order}
    for target,c in zip(q.target.unique(),[BLUE,TEAL,RED]):
        z=q[q.target==target].set_index("model").reindex(factor_order);axs[1].plot(np.arange(len(z)),z.mae_improvement_pct,"o-",label=target,color=c,lw=1,ms=4)
    axs[1].set_xticks(np.arange(len(factor_order)),[factor_labels[x] for x in factor_order]);plt.setp(axs[1].get_xticklabels(),rotation=28,rotation_mode="anchor",ha="right");axs[1].axhline(0,c=DARK,lw=.7);axs[1].set_ylabel("MAE improvement (%)");axs[1].legend(frameon=False,fontsize=5.7);panel(axs[1],"b");clean(axs[1])
    panel_title(axs[1],"Incremental descriptor value")
    fig.subplots_adjust(left=.10,right=.96,bottom=.25,top=.90,wspace=.36)
    save_bundle(fig,"figS2_model_benchmark",[("a",ms),("b",q)],si=True)

    # S3 validation ladder and scrambling
    ladder=read("validation_ladder.csv"); scr=read("y_scrambling.csv")
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.0));
    q=ladder.groupby(["target","regime"]).r2.agg(["mean","std"]).reset_index();reg=list(q.regime.unique());x=np.arange(len(reg)); short_reg={"Naive random holdout":"Random","Duplicate-safe grouped CV":"Grouped","Leave-one-grade-out":"Leave-grade","Leave-one-system-out":"Leave-system"}
    for i,(tar,c) in enumerate(zip(q.target.unique(),[BLUE,TEAL,RED])):
        z=q[q.target==tar].set_index("regime").reindex(reg);axs[0].errorbar(x+i*.07,z["mean"],yerr=z["std"],fmt="o-",ms=3,lw=.8,color=c,label=tar)
    axs[0].set_xticks(x+.07,[short_reg.get(a,a) for a in reg]);plt.setp(axs[0].get_xticklabels(),rotation=25,rotation_mode="anchor",ha="right");axs[0].set_ylabel("R²");axs[0].legend(frameon=False,fontsize=5.7);panel(axs[0],"a");clean(axs[0])
    panel_title(axs[0],"Validation ladder")
    kinds=scr.kind.unique();
    for i,(tar,c) in enumerate(zip(scr.target.unique(),[BLUE,TEAL,RED])):
        z=scr[scr.target==tar];vals=[z[z.kind==k].r2.values for k in kinds];pos=np.arange(len(kinds))+i*.22;axs[1].boxplot(vals,positions=pos,widths=.17,showfliers=False,patch_artist=True,boxprops=dict(facecolor=c,alpha=.45),medianprops=dict(color=DARK))
    axs[1].set_xticks(np.arange(len(kinds))+.22,kinds);plt.setp(axs[1].get_xticklabels(),rotation=20,rotation_mode="anchor",ha="right");axs[1].set_ylabel("R²");panel(axs[1],"b");clean(axs[1])
    panel_title(axs[1],"Label scrambling")
    fig.subplots_adjust(left=.10,right=.96,bottom=.25,top=.90,wspace=.36)
    save_bundle(fig,"figS3_validation_and_scrambling",[("a",ladder),("b",scr)],si=True)

    # S4 OOD and acquisition
    ood=read("ood_error_records.csv"); pri=read("data_acquisition_priority.csv")
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.0));
    # Preserve every OOD record; rasterized marks keep the full cloud legible and compact.
    sample=ood
    for tar,c in zip(sample.target.unique(),[BLUE,TEAL,RED]):
        z=sample[sample.target==tar];axs[0].scatter(z.distance,z.abs_error,s=9,alpha=.25,c=c,label=tar,rasterized=True)
    axs[0].set_xlabel("OOD distance");axs[0].set_ylabel("Absolute error");axs[0].legend(frameon=False,fontsize=5.7);panel(axs[0],"a");clean(axs[0])
    panel_title(axs[0],"OOD distance and error")
    p=pri.sort_values("priority_score");axs[1].barh(p.grade,p.priority_score,color=[GRADE_COLORS.get(x,NEUTRAL) for x in p.grade]);axs[1].set_xlabel("Acquisition priority score");panel(axs[1],"b");clean(axs[1])
    panel_title(axs[1],"Data-acquisition priority")
    save_bundle(fig,"figS4_ood_uncertainty",[("a",ood),("b",pri)],si=True)

    # S5 correlation contrast
    gl=read("global_feature_correlations.csv");wg=read("within_grade_correlations.csv")
    merged=gl.merge(wg,on=["feature","target"],suffixes=("_global","_within"));
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.0));
    for tar,c in zip(merged.target.unique(),[BLUE,TEAL,RED]):
        z=merged[merged.target==tar];axs[0].scatter(z.correlation_global,z.correlation_within,s=24,c=c,label=tar,alpha=.75)
    axs[0].axhline(0,c=NEUTRAL,lw=.6);axs[0].axvline(0,c=NEUTRAL,lw=.6);axs[0].plot([-1,1],[-1,1],ls="--",c=DARK,lw=.7);axs[0].set_xlabel("Global correlation");axs[0].set_ylabel("Within-grade correlation");axs[0].legend(frameon=False,fontsize=5.7);panel(axs[0],"a");clean(axs[0])
    panel_title(axs[0],"Global and within-grade")
    merged["shift"]=abs(merged.correlation_global-merged.correlation_within);top=merged.nlargest(12,"shift").sort_values("shift").copy(); target_short={"yield_strength_mpa":"YS","ultimate_tensile_strength_mpa":"UTS","elongation_pct":"EL"}; feature_short={"rolling_temp_c":"roll T","rolling_reduction_pct":"reduction","specimen_thickness_mm":"thickness","Al_wt_pct":"Al","V_wt_pct":"V","Fe_wt_pct":"Fe"}; top["short_label"]=[feature_short.get(f,f.replace("_wt_pct",""))+" | "+target_short.get(t,t) for f,t in zip(top.feature,top.target)];axs[1].barh(top.short_label,top["shift"],color=RED,alpha=.75);axs[1].set_xlabel("|global − within-grade| correlation");axs[1].tick_params(axis="y",labelsize=5.6);panel(axs[1],"b");clean(axs[1])
    panel_title(axs[1],"Largest correlation shifts")
    save_bundle(fig,"figS5_global_within_grade",[("a-b",merged)],si=True)

    # S6 symbolic regression and local fronts
    pareto=read("historical_local_pareto.csv")
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.0));
    q=v.copy();q["sr_abs_error"]=abs(q.pred_SR-q.measured);q["model_abs_error"]=abs(q.selected_prediction-q.measured)
    target_short={"yield_strength_mpa":"YS","ultimate_tensile_strength_mpa":"UTS","elongation_pct":"EL"};x=np.arange(len(q));axs[0].bar(x-.18,q.sr_abs_error,.36,color=NEUTRAL,label="Symbolic regression");axs[0].bar(x+.18,q.model_abs_error,.36,color=BLUE,label="Selected model");axs[0].set_xticks(x,[f"{a.replace('Ti_','T')}–{target_short.get(b,b)}" for a,b in zip(q.sample_id,q.target)]);plt.setp(axs[0].get_xticklabels(),rotation=35,rotation_mode="anchor",ha="right");axs[0].set_ylabel("Absolute error");axs[0].legend(frameon=False,fontsize=5.7);panel(axs[0],"a");clean(axs[0])
    panel_title(axs[0],"Trial prediction errors")
    for grade,gp in pareto.groupby("grade"):
        axs[1].plot(gp.x,gp.y,"o-",ms=3,lw=.8,label=grade,color=GRADE_COLORS.get(grade,NEUTRAL))
    axs[1].set_xlabel("Strength objective");axs[1].set_ylabel("Ductility objective");axs[1].legend(frameon=False,fontsize=5.5);panel(axs[1],"b");clean(axs[1])
    panel_title(axs[1],"Grade-specific fronts")
    fig.subplots_adjust(left=.10,right=.96,bottom=.28,top=.90,wspace=.36)
    save_bundle(fig,"figS6_symbolic_and_fronts",[("a",q),("b",pareto)],si=True)

    # S7 all six model predictions
    models=["pred_Ridge","pred_XGBoost","pred_RandomForest","pred_SVR","pred_GPR","pred_MLP"]
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.65));
    for ax,(tar,label) in zip(axs,[("yield_strength_mpa","YS"),("ultimate_tensile_strength_mpa","UTS"),("elongation_pct","EL")]):
        q=v[v.target==tar].set_index("sample_id").loc[["Ti_9","Ti_20"]];x=np.arange(2)
        for j,mn in enumerate(models):ax.scatter(x+(j-2.5)*.055,q[mn],s=18,label=mn.replace("pred_","") if tar=="yield_strength_mpa" else None,alpha=.8)
        ax.scatter(x,q.measured,s=72,marker="*",c=GOLD,edgecolor=DARK,lw=.6,zorder=5,label="Measured" if tar=="yield_strength_mpa" else None)
        ax.set_xticks(x,["Ti_9","Ti_20"]);ax.set_ylabel(label);panel(ax,chr(ord('a')+list(axs).index(ax)));clean(ax)
        panel_title(ax,{"yield_strength_mpa":"YS predictions","ultimate_tensile_strength_mpa":"UTS predictions","elongation_pct":"EL predictions"}[tar])
    axs[0].legend(frameon=False,fontsize=5.0,ncol=2,loc="best")
    save_bundle(fig,"figS7_all_model_trial_predictions",[("a-c",v)],si=True)

    # S8 PCA variance and grade explainability
    pca=read("pca_explained_variance.csv"); ge=read("feature_grade_explainability.csv")
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.0));
    for view,c in zip(pca.view.unique(),[BLUE,TEAL,RED,GOLD]):
        q=pca[pca.view==view];axs[0].plot(q.component,q.explained_variance,"o-",label=view,color=c,ms=3)
    axs[0].set_xlabel("Principal component");axs[0].set_ylabel("Explained variance");axs[0].legend(frameon=False,fontsize=5.5);panel(axs[0],"a");clean(axs[0])
    panel_title(axs[0],"PCA explained variance")
    z=ge.sort_values("grade_r2").tail(15);axs[1].barh([feature_label(x) for x in z.feature],z.grade_r2,color=NEUTRAL);axs[1].set_xlabel("Variance explained by grade (R²)");axs[1].tick_params(axis="y",labelsize=5.7);panel(axs[1],"b");clean(axs[1])
    panel_title(axs[1],"Grade-coded descriptors")
    save_bundle(fig,"figS8_multiview_structure",[("a",pca),("b",ge)],si=True)

    # S9 composition projection and property distributions from the 78 accumulated records.
    comp_cols=["Al_wt_pct","V_wt_pct","Fe_wt_pct","Cr_wt_pct","Mo_wt_pct","Nb_wt_pct","Zr_wt_pct","O_wt_pct"]
    xx=h[comp_cols].apply(pd.to_numeric,errors="coerce")
    zz=StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(xx))
    emb=TSNE(n_components=2,perplexity=18,learning_rate="auto",init="pca",random_state=20260827).fit_transform(zz)
    ts=pd.DataFrame({"sample_id":h.sample_id,"grade":h.alloy_or_coating_name,"TSNE1":emb[:,0],"TSNE2":emb[:,1]})
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),gridspec_kw={"width_ratios":[1.1,1.4]})
    for grade,gp in ts.groupby("grade"):
        axs[0].scatter(gp.TSNE1,gp.TSNE2,s=28,c=GRADE_COLORS.get(grade,NEUTRAL),label=grade,alpha=.75)
    axs[0].set_xlabel("t-SNE 1");axs[0].set_ylabel("t-SNE 2");axs[0].legend(frameon=False,fontsize=5.7);panel(axs[0],"a");clean(axs[0])
    panel_title(axs[0],"Composition space")
    long=h[["sample_id","alloy_or_coating_name","yield_strength_mpa","ultimate_tensile_strength_mpa","elongation_pct"]].melt(id_vars=["sample_id","alloy_or_coating_name"],var_name="target",value_name="value")
    long["normalized_value"]=long.groupby("target").value.transform(lambda s:(s-s.min())/(s.max()-s.min()))
    targets=["yield_strength_mpa","ultimate_tensile_strength_mpa","elongation_pct"]
    positions=[];data=[];colors=[];labels=[];pos=0
    for target in targets:
        for grade in GRADE_COLORS:
            vals=long[(long.target==target)&(long.alloy_or_coating_name==grade)].normalized_value.dropna().values
            data.append(vals);positions.append(pos);colors.append(GRADE_COLORS[grade]);labels.append(grade if target==targets[0] else "");pos+=1
        pos+=1
    bp=axs[1].boxplot(data,positions=positions,widths=.62,patch_artist=True,showfliers=False,
        medianprops=dict(color=DARK,lw=.8),whiskerprops=dict(color=NEUTRAL),capprops=dict(color=NEUTRAL))
    for patch,c in zip(bp["boxes"],colors):patch.set_facecolor(c);patch.set_alpha(.55);patch.set_edgecolor(DARK)
    centers=[np.mean(positions[i*5:(i+1)*5]) for i in range(3)]
    axs[1].set_xticks(centers,["YS","UTS","EL"]);axs[1].set_ylabel("Within-property normalized value");panel(axs[1],"b");clean(axs[1])
    panel_title(axs[1],"Property distributions")
    save_bundle(fig,"figS9_composition_and_property_distribution",[("a",ts),("b",long)],si=True)

    # S10 Pearson, Spearman and distance-correlation maps retained from the original analysis.
    corr=read_original("correlation_matrix.csv")
    corr.columns=[str(c).lstrip("\ufeff") for c in corr.columns]
    target_names=["yield_strength_mpa","ultimate_tensile_strength_mpa","elongation_pct"]
    pearson_cols=[f"{t}_pearson" for t in target_names]
    order=corr.assign(rank_strength=corr[pearson_cols].abs().max(axis=1)).nlargest(14,"rank_strength")
    fig=plt.figure(figsize=(7.2,4.0))
    gs=fig.add_gridspec(1,5,width_ratios=[1,1,.055,1,.055],wspace=.52,
                        left=.17,right=.96,top=.91,bottom=.10)
    axs=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[0,3])]
    cax_corr=fig.add_subplot(gs[0,2]);cax_dist=fig.add_subplot(gs[0,4]);ims=[]
    blocks=[("Pearson","pearson",-1,1,"coolwarm"),("Spearman","spearman",-1,1,"coolwarm"),("Distance correlation","distance",0,1,"Blues")]
    for i,(title,suffix,vmin,vmax,cmap) in enumerate(blocks):
        cols=[f"{t}_{suffix}" for t in target_names];mat=order[cols].to_numpy(float)
        im=axs[i].imshow(mat,aspect="auto",vmin=vmin,vmax=vmax,cmap=cmap);ims.append(im)
        axs[i].set_xticks(range(3),["YS","UTS","EL"]);panel_title(axs[i],title)
        if i==0:axs[i].set_yticks(range(len(order)),[feature_label(x) for x in order.feature])
        else:
            axs[i].set_yticks(range(len(order)));axs[i].tick_params(axis="y",left=False,labelleft=False)
        white_cell_text(axs[i],mat,fontsize=5.0)
        panel(axs[i],chr(ord('a')+i))
    fig.colorbar(ims[1],cax=cax_corr);cax_corr.tick_params(labelsize=5.5)
    fig.colorbar(ims[2],cax=cax_dist);cax_dist.tick_params(labelsize=5.5)
    save_bundle(fig,"figS10_correlation_heatmaps",[("a-c",order)],si=True)

    # S11 SHAP summaries, interpreted only as model contribution rankings.
    fig,axs=plt.subplots(1,3,figsize=(7.2,3.4))
    shap_frames=[];payload=recomputed_shap()
    for i,(target,title,c) in enumerate([("yield_strength_mpa","YS",BLUE),("ultimate_tensile_strength_mpa","UTS",TEAL),("elongation_pct","EL",RED)]):
        d=payload[target];s=pd.DataFrame({"feature":d["feature_cols"],"mean_abs_shap":d["importance"]})
        s=s.nlargest(10,"mean_abs_shap").sort_values("mean_abs_shap");shap_frames.append(s.assign(target=target))
        axs[i].barh([feature_label(x) for x in s.feature],s.mean_abs_shap,color=c,alpha=.8);axs[i].set_xlabel("Mean |SHAP value|");panel_title(axs[i],f"{title} importance");axs[i].tick_params(axis="y",labelsize=5.4);panel(axs[i],chr(ord('a')+i));clean(axs[i])
    save_bundle(fig,"figS11_shap_importance",[("a-c",pd.concat(shap_frames,ignore_index=True))],si=True)

    # S12 symbolic-regression complexity–loss fronts for the strength targets.
    sy=read_original("sr_yield_strength_mpa_global_front.csv");su=read_original("sr_ultimate_tensile_strength_mpa_global_front.csv")
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.0))
    for ax,frame,title,c,lab in [(axs[0],sy,"Yield strength",BLUE,"a"),(axs[1],su,"Ultimate tensile strength",TEAL,"b")]:
        loss=pd.to_numeric(frame.loss,errors="coerce");assert (loss.dropna()>0).all()
        ax.plot(frame.complexity,np.log10(loss),"o-",ms=3,lw=.9,color=c)
        ax.set_xlabel("Formula complexity");ax.set_ylabel("log10(loss)");panel_title(ax,f"{title} front");panel(ax,lab);clean(ax)
    save_bundle(fig,"figS12_symbolic_regression_fronts",[("a",sy),("b",su)],si=True)


def write_contract():
    text = """# Figure contract — titanium manuscript v11 for JCIM

## Central claim
Accumulated composition–processing–property data support within-grade titanium plate modeling, TC4 process–microstructure analysis, and the subsequent Ti_9/Ti_20 single-test trials.

## Evidence hierarchy
Observed measurements, model interpretation and subsequent single-test trials occupy distinct evidence layers. All reported model outputs are generated from the released physically consistent historical dataset.

## Visual system
7.2-inch journal width; Python/matplotlib; editable SVG text; PDF, 600-dpi TIFF and 600-dpi PNG; 5-point minimum final text. Neutral gray encodes accumulated test data, blue/teal material branches, muted red hypotheses, and gold subsequent trials.

## Reviewer risks addressed
- TC4 is interpreted as a coupled process–microstructure comparison.
- Ti_9/Ti_20 chronology is explicit.
- Global and within-grade correlations are shown together.
- Prospective characterization is described in the manuscript text.
"""
    (ROOT / "figure_contract.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    write_contract()
    fig1_data_landscape(); fig2_material_relationships(); fig3_model_validation(); fig4_shap_importance(); fig5_shap_distributions(); fig5_symbolic_regression()
    fig6_v13(); fig3(); fig4(); si_figures(); si_full_model_figures()
    print("generated figure bundles", len(list(OUT.glob("*.svg"))), len(list(SI_OUT.glob("*.svg"))))
