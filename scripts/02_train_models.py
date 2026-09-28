# -*- coding: utf-8 -*-
"""
02_train_models.py
6 模型 × 3 目标训练：Ridge / XGBoost / RandomForest / SVR / GPR / MLP。
超参搜索（5 折 CV）+ 8:2 划分（random_state=42）+ StandardScaler（仅在训练集拟合）。
输出：data/model_summary.csv、data/results.json、figure/fig3_diag_{target}.png
环境：LubLLM
"""
import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.metrics import r2_score, mean_squared_error
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
TARGET_EN = {'yield_strength_mpa': 'YS (MPa)',
             'ultimate_tensile_strength_mpa': 'UTS (MPa)',
             'elongation_pct': 'EL (%)'}
MODEL_NAMES = {'ridge': 'Ridge', 'xgb': 'XGBoost', 'rf': 'RandomForest',
               'svr': 'SVR', 'gpr': 'GPR', 'mlp': 'MLP'}


def get_model(mname):
    if mname == 'ridge':
        return Ridge(random_state=42), {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    if mname == 'xgb':
        return XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=4), {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.03, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]}
    if mname == 'rf':
        return RandomForestRegressor(random_state=42, n_jobs=4), {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 4, 8, 12],
            'min_samples_split': [2, 5, 10]}
    if mname == 'svr':
        return SVR(), {
            'C': [0.1, 1.0, 10.0, 100.0, 1000.0],
            'gamma': [0.001, 0.01, 0.1, 'scale', 'auto'],
            'epsilon': [0.01, 0.05, 0.1, 0.2]}
    if mname == 'gpr':
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01)
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                        n_restarts_optimizer=3, random_state=42), {
            'alpha': [1e-8, 1e-6, 1e-4]}
    if mname == 'mlp':
        return MLPRegressor(random_state=42, max_iter=2000, early_stopping=True), {
            'hidden_layer_sizes': [(16,), (32,), (16, 8), (32, 16)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.0005, 0.001, 0.005]}
    raise ValueError(mname)


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, '02_engineered_features.csv'), encoding='utf-8-sig')
    feature_cols = [c for c in df.columns
                    if c not in TARGETS + ['sample_id', 'material_system', 'alloy_or_coating_name']]
    print('特征数:', len(feature_cols))

    summary_rows = []
    results = {}
    for t in TARGETS:
        X = df[feature_cols].values.astype(float)
        y = df[t].values.astype(float)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)
        results[t] = {}

        for mname in ['ridge', 'xgb', 'rf', 'svr', 'gpr', 'mlp']:
            base, grid = get_model(mname)
            if mname == 'gpr':
                search = GridSearchCV(base, grid, cv=5, scoring='r2', n_jobs=1)
            else:
                search = RandomizedSearchCV(base, grid, n_iter=30, cv=5, scoring='r2',
                                            random_state=42, n_jobs=4)
            search.fit(X_tr_s, y_tr)
            best = search.best_estimator_
            y_tr_pred = best.predict(X_tr_s)
            y_te_pred = best.predict(X_te_s)
            tr_r2 = r2_score(y_tr, y_tr_pred)
            te_r2 = r2_score(y_te, y_te_pred)
            te_rmse = float(np.sqrt(mean_squared_error(y_te, y_te_pred)))
            # 10 折 CV
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            cv_r2 = []
            for i_tr, i_val in kf.split(X_tr_s):
                m, _ = get_model(mname)
                m.set_params(**search.best_params_)
                m.fit(X_tr_s[i_tr], y_tr[i_tr])
                cv_r2.append(r2_score(y_tr[i_val], m.predict(X_tr_s[i_val])))
            summary_rows.append({
                'target': t, 'model': MODEL_NAMES[mname],
                'train_r2': round(tr_r2, 4), 'test_r2': round(te_r2, 4),
                'cv_r2_mean': round(float(np.mean(cv_r2)), 4), 'cv_r2_std': round(float(np.std(cv_r2)), 4),
                'test_rmse': round(te_rmse, 2), 'best_params': str(search.best_params_)})
            results[t][MODEL_NAMES[mname]] = {
                'best_params': search.best_params_,
                'train_r2': tr_r2, 'test_r2': te_r2, 'test_rmse': te_rmse,
                'cv_r2_mean': float(np.mean(cv_r2)),
                'y_test': y_te.tolist(), 'y_test_pred': y_te_pred.tolist(),
                'y_train': y_tr.tolist(), 'y_train_pred': y_tr_pred.tolist()}
            print(f'{t:35s} {MODEL_NAMES[mname]:13s} train={tr_r2:.4f} test={te_r2:.4f} '
                  f'cv={np.mean(cv_r2):.4f}±{np.std(cv_r2):.4f} rmse={te_rmse:.1f}')

        diag_figure(t, y_tr, y_te, results[t])

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(DATA_DIR, 'model_summary.csv'), index=False, encoding='utf-8-sig')
    with open(os.path.join(DATA_DIR, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('saved data/model_summary.csv, data/results.json')
    print('02 done')


def diag_figure(target, y_tr, y_te, results):
    """图3/4/5：每目标 2×3 六模型对角散点图（英文、无总标题、nature 配色）。"""
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.6))
    axes = axes.ravel()
    for i, name in enumerate(['Ridge', 'XGBoost', 'RandomForest', 'SVR', 'GPR', 'MLP']):
        ax = axes[i]
        r = results[name]
        ytr_pred = np.array(r['y_train_pred'])
        yte_pred = np.array(r['y_test_pred'])
        lim = [min(min(y_tr), min(y_te), min(ytr_pred), min(yte_pred)) * 0.95,
               max(max(y_tr), max(y_te), max(ytr_pred), max(yte_pred)) * 1.05]
        ax.scatter(y_tr, ytr_pred, s=90, c='#0F4D92', alpha=0.8, edgecolors='white',
                   linewidths=0.6, label='Training')
        ax.scatter(y_te, yte_pred, s=150, c='#B64342', alpha=0.9, edgecolors='white',
                   linewidths=0.8, label='Test')
        ax.plot(lim, lim, 'k--', lw=1.2)
        ax.set_xlim(lim), ax.set_ylim(lim)
        ax.text(0.04, 0.93, f'$R^2$ train = {r["train_r2"]:.3f}\n$R^2$ test = {r["test_r2"]:.3f}',
                transform=ax.transAxes, fontsize=12.5, va='top')
        ax.text(0.03, 1.0, '(' + 'abcdef'[i] + ') ' + name,
                transform=ax.transAxes, fontsize=15, fontweight='bold', va='bottom')
        ax.tick_params(labelsize=12)
        if i in (0, 3):
            ax.set_ylabel('Predicted ' + TARGET_EN[target], fontsize=14)
        if i >= 3:
            ax.set_xlabel('Measured ' + TARGET_EN[target], fontsize=14)
        if i == 0:
            ax.legend(fontsize=12, loc='lower right')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f'fig3_diag_{target}.png')
    fig.savefig(out, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print('saved', out)


if __name__ == '__main__':
    main()
