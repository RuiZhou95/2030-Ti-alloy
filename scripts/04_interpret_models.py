# -*- coding: utf-8 -*-
"""
03_shap_analysis.py
基于 XGBoost（各目标最优超参）的 SHAP 特征重要性分析。
输出：data/shap_importance_{target}.csv、figure/fig4_shap_importance.png、figure/fig5_shap_beeswarm.png
环境：LubLLM
"""
import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
# SHAP releases prior to 0.42 still reference NumPy's removed ``np.int`` alias.
# Keep the analysis runnable on archived environments without changing results.
if not hasattr(np, "int"):
    np.int = int
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get('TI_ALLOY_DATA_DIR', os.path.join(BASE, 'outputs', 'data'))
FIG_DIR = os.environ.get('TI_ALLOY_FIGURE_DIR', os.path.join(BASE, 'outputs', 'figures'))
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.9,
    'legend.frameon': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})

TARGETS = ['yield_strength_mpa', 'ultimate_tensile_strength_mpa', 'elongation_pct']
TARGET_EN = {'yield_strength_mpa': 'YS',
             'ultimate_tensile_strength_mpa': 'UTS',
             'elongation_pct': 'EL'}

FEATURE_EN = {
    'Al_wt_pct': 'Al (wt%)', 'Ti_wt_pct': 'Ti (wt%)', 'Fe_wt_pct': 'Fe (wt%)',
    'Cr_wt_pct': 'Cr (wt%)', 'Mo_wt_pct': 'Mo (wt%)', 'V_wt_pct': 'V (wt%)',
    'Nb_wt_pct': 'Nb (wt%)', 'Zr_wt_pct': 'Zr (wt%)', 'C_wt_pct': 'C (wt%)',
    'N_wt_pct': 'N (wt%)', 'O_wt_pct': 'O (wt%)', 'H': 'H (wt%)',
    'homogenization_temp_c': 'Homogenization temp.', 'homogenization_time_h': 'Homogenization time',
    'solution_temp_c': 'Solution temp.', 'solution_time_h': 'Solution time',
    'aging_temp_c': 'Aging temp.', 'aging_time_h': 'Aging time',
    'annealing_temp_c': 'Annealing temp.', 'annealing_time_h': 'Annealing time',
    'rolling_temp_c': 'Rolling temp.', 'rolling_reduction_pct': 'Rolling reduction',
    'specimen_thickness_mm': 'Thickness', 'Ti_x_Al': 'Ti×Al', 'Ti_x_V': 'Ti×V',
    'total_alloy_wt': 'Total alloying',
}


def _plot_beeswarm(ax, shap_values, X, feature_names, max_display=8):
    """SHAP beeswarm：横轴 SHAP 值，纵轴特征，颜色=特征值高低（红高蓝低）。"""
    order = np.argsort(np.abs(shap_values).mean(axis=0))[::-1][:max_display]
    sv = shap_values[:, order][:, ::-1]
    Xd = X[:, order][:, ::-1]
    names = [FEATURE_EN.get(feature_names[i], feature_names[i]) for i in order][::-1]
    rng = np.random.RandomState(0)
    cmap = plt.cm.RdBu_r
    for i in range(len(names)):
        vals = sv[:, i]
        fvals = Xd[:, i]
        fmax, fmin = fvals.max(), fvals.min()
        norm = (fvals - fmin) / (fmax - fmin + 1e-9)
        colors = cmap(norm)
        jitter = rng.random(len(vals)) * 0.7 - 0.35
        ax.scatter(vals, i + jitter, c=colors, s=52, alpha=0.9, edgecolors='none')
    ax.axvline(0, color='#767676', lw=1.0, ls='--')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=12)
    ax.set_xlabel('SHAP value', fontsize=13)


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, '02_engineered_features.csv'), encoding='utf-8-sig')
    feature_cols = [c for c in df.columns
                    if c not in TARGETS + ['sample_id', 'material_system', 'alloy_or_coating_name']]
    X_all = df[feature_cols].values.astype(float)

    with open(os.path.join(DATA_DIR, 'results.json'), encoding='utf-8') as f:
        results = json.load(f)

    fig4, axes4 = plt.subplots(1, 3, figsize=(15.5, 6.2))
    fig5, axes5 = plt.subplots(1, 3, figsize=(15.5, 5.8))
    for k, t in enumerate(TARGETS):
        params = results[t]['XGBoost']['best_params']
        y = df[t].values.astype(float)
        X_tr, X_te, y_tr, y_te = train_test_split(X_all, y, test_size=0.2, random_state=42)
        model = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=4, **params)
        model.fit(X_tr, y_tr)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_te)

        imp = pd.DataFrame({'feature': feature_cols,
                            'mean_abs_shap': np.abs(shap_values).mean(axis=0)})
        imp = imp.sort_values('mean_abs_shap', ascending=False)
        imp.to_csv(os.path.join(DATA_DIR, f'shap_importance_{t}.csv'),
                   index=False, encoding='utf-8-sig')
        print(f'--- {t} Top10 ---')
        print(imp.head(10).to_string(index=False))

        # 图6：重要性条形图
        ax = axes4[k]
        top = imp.head(12).iloc[::-1]
        names = [FEATURE_EN.get(f, f) for f in top['feature']]
        ax.barh(range(len(top)), top['mean_abs_shap'], color='#0F4D92')
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(names, fontsize=12)
        ax.text(0.02, 1.0, '(' + 'abc'[k] + ') ' + TARGET_EN[t],
                transform=ax.transAxes, fontsize=15, fontweight='bold', va='bottom')
        ax.tick_params(labelsize=12)
        ax.set_xlabel('mean |SHAP value|', fontsize=13)

        # 图7：beeswarm
        ax5 = axes5[k]
        _plot_beeswarm(ax5, shap_values, X_te, feature_cols, max_display=8)
        ax5.text(0.02, 1.0, '(' + 'abc'[k] + ') ' + TARGET_EN[t],
                 transform=ax5.transAxes, fontsize=15, fontweight='bold', va='bottom')
        ax5.tick_params(labelsize=12)

    fig4.tight_layout()
    fig4.savefig(os.path.join(FIG_DIR, 'fig4_shap_importance.png'), dpi=600, bbox_inches='tight')
    plt.close(fig4)
    print('saved fig4_shap_importance.png')
    fig5.tight_layout()
    fig5.savefig(os.path.join(FIG_DIR, 'fig5_shap_beeswarm.png'), dpi=600, bbox_inches='tight')
    plt.close(fig5)
    print('saved fig5_shap_beeswarm.png')
    print('03 done')


if __name__ == '__main__':
    main()
