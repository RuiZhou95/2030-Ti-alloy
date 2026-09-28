# -*- coding: utf-8 -*-
"""
04_symbolic_regression.py
PySR 符号回归（YS、UTS 两目标）的 Python 侧：
1. 用各目标 SHAP 前 8 特征准备输入（与 02 相同的 8:2 划分、训练集标准化）；
2. 调用 julia（SymbolicRegression.jl 0.24.5，pySR 环境对应 julia_env）执行搜索；
3. 绘制图6：帕累托前沿 + 公式验证散点。
输出：data/sr_pareto_{target}.csv、data/sr_best_{target}.json、figure/fig6_sr_{target}.png
环境：LubLLM（本脚本） + julia_env 的 julia（搜索）
"""
import os
import json
import subprocess
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

JULIA_BIN = os.environ.get('JULIA_BIN', 'julia')
JULIA_PROJECT = os.environ.get('JULIA_PROJECT', os.path.join(BASE, 'julia'))
TARGETS = ['yield_strength_mpa', 'ultimate_tensile_strength_mpa']
PREP_TARGETS = TARGETS + ['elongation_pct']
TARGET_EN = {'yield_strength_mpa': 'YS', 'ultimate_tensile_strength_mpa': 'UTS'}
FEATURE_CN = {
    'Al_wt_pct': 'Al', 'Ti_wt_pct': 'Ti', 'Fe_wt_pct': 'Fe', 'Cr_wt_pct': 'Cr',
    'Mo_wt_pct': 'Mo', 'V_wt_pct': 'V', 'Nb_wt_pct': 'Nb', 'Zr_wt_pct': 'Zr',
    'C_wt_pct': 'C', 'N_wt_pct': 'N', 'O_wt_pct': 'O', 'H': 'H',
    'homogenization_temp_c': 'T_hom', 'homogenization_time_h': 't_hom',
    'solution_temp_c': 'T_sol', 'solution_time_h': 't_sol',
    'aging_temp_c': 'T_age', 'aging_time_h': 't_age',
    'annealing_temp_c': 'T_ann', 'annealing_time_h': 't_ann',
    'rolling_temp_c': 'T_roll', 'rolling_reduction_pct': 'red_pct',
    'specimen_thickness_mm': 'thick', 'Ti_x_Al': 'TiAl', 'Ti_x_V': 'TiV',
    'total_alloy_wt': 'alloy_tot',
}


def prepare_inputs(target):
    df = pd.read_csv(os.path.join(DATA_DIR, '02_engineered_features.csv'), encoding='utf-8-sig')
    shap_imp = pd.read_csv(os.path.join(DATA_DIR, f'shap_importance_{target}.csv'), encoding='utf-8-sig')
    top8 = shap_imp['feature'].head(8).tolist()
    X = df[top8].values.astype(float)
    y = df[target].values.astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)
    prefix = os.path.join(DATA_DIR, f'sr_{target}')
    np.savetxt(prefix + '_X_tr.csv', X_tr_s, delimiter=',')
    np.savetxt(prefix + '_X_te.csv', X_te_s, delimiter=',')
    np.savetxt(prefix + '_y_tr.csv', y_tr, delimiter=',')
    np.savetxt(prefix + '_y_te.csv', y_te, delimiter=',')
    with open(prefix + '_features.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(top8))
    return top8


def run_julia():
    env = dict(os.environ)
    env['JULIA_PKG_OFFLINE'] = 'true'
    script = os.path.join(BASE, 'julia', 'symbolic_regression_search.jl')
    procs = []
    for target in TARGETS:
        cmd = [JULIA_BIN, '--project=' + JULIA_PROJECT, '--threads=8', '--startup-file=no',
               script, target, DATA_DIR]
        print('running julia for', target, '...')
        procs.append((target, subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE, text=True)))
    for target, proc in procs:
        out, err = proc.communicate(timeout=3600)
        print(f'--- {target} julia exit={proc.returncode} ---')
        print(out[-800:])
        if proc.returncode != 0:
            print('STDERR:', err[-1500:])
            raise RuntimeError('julia SR search failed for ' + target)


def draw_figures():
    """图8：两目标帕累托前沿 + 公式验证散点（2×2，英文，nature 配色）。"""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
    for col, t in enumerate(TARGETS):
        pareto = pd.read_csv(os.path.join(DATA_DIR, f'sr_{t}_pareto.csv'), encoding='utf-8-sig')
        best = json.load(open(os.path.join(DATA_DIR, f'sr_{t}_best.json'), encoding='utf-8'))
        # (a)(b) 帕累托前沿
        ax = axes[0, col]
        ax.scatter(pareto['complexity'], pareto['loss'], s=70, c='#767676', alpha=0.7,
                   edgecolors='white', linewidths=0.6)
        ax.scatter(best['complexity'], best['loss'], s=320, c='#E53935', marker='*',
                   edgecolors='#4D4D4D', linewidths=0.8, zorder=5, label='Selected')
        ax.set_xlabel('Complexity', fontsize=14)
        ax.set_ylabel('Loss (MSE)', fontsize=14)
        ax.text(0.03, 1.0, '(' + 'ab'[col] + ') ' + TARGET_EN[t] + ' Pareto front',
                transform=ax.transAxes, fontsize=15, fontweight='bold', va='bottom')
        ax.legend(fontsize=12, loc='upper right')
        ax.tick_params(labelsize=12)
        # (c)(d) 公式验证散点
        ax2 = axes[1, col]
        y_tr = np.loadtxt(os.path.join(DATA_DIR, f'sr_{t}_y_tr.csv'))
        y_te = np.loadtxt(os.path.join(DATA_DIR, f'sr_{t}_y_te.csv'))
        p_tr = np.loadtxt(os.path.join(DATA_DIR, f'sr_{t}_pred_tr.csv'))
        p_te = np.loadtxt(os.path.join(DATA_DIR, f'sr_{t}_pred_te.csv'))
        lim = [min(y_tr.min(), y_te.min(), p_tr.min(), p_te.min()) * 0.95,
               max(y_tr.max(), y_te.max(), p_tr.max(), p_te.max()) * 1.05]
        ax2.scatter(y_tr, p_tr, s=45, c='#0F4D92', alpha=0.8, edgecolors='white',
                    linewidths=0.6, label='Training')
        ax2.scatter(y_te, p_te, s=75, c='#B64342', alpha=0.9, edgecolors='white',
                    linewidths=0.8, label='Test')
        ax2.plot(lim, lim, 'k--', lw=1.2)
        ax2.set_xlim(lim), ax2.set_ylim(lim)
        ax2.text(0.04, 0.9, f'$R^2$ train = {best["r2_train"]:.3f}\n$R^2$ test = {best["r2_test"]:.3f}',
                 transform=ax2.transAxes, fontsize=12.5, va='top')
        ax2.set_xlabel('Measured ' + TARGET_EN[t], fontsize=14)
        ax2.set_ylabel('Formula predicted ' + TARGET_EN[t], fontsize=14)
        ax2.text(0.03, 1.0, '(' + 'cd'[col] + ') ' + TARGET_EN[t] + ' formula validation',
                 transform=ax2.transAxes, fontsize=15, fontweight='bold', va='bottom')
        ax2.legend(fontsize=12, loc='lower right')
        ax2.tick_params(labelsize=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig6_sr_formula.png')
    fig.savefig(out, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print('saved', out)


def main():
    import sys
    if '--prepare-only' in sys.argv:
        for t in PREP_TARGETS:
            prepare_inputs(t)
            print(f'{t}: 特征见 sr_{t}_features.txt')
        print('04 prepare done')
        return
    if '--no-search' not in sys.argv:
        for t in PREP_TARGETS:
            prepare_inputs(t)
            print(f'{t}: 特征见 sr_{t}_features.txt')
        run_julia()
    for t in TARGETS:
        best = json.load(open(os.path.join(DATA_DIR, f'sr_{t}_best.json'), encoding='utf-8'))
        print(f'  选定公式: {best["formula_text"]}')
        print(f'  复杂度={best["complexity"]}, 训练R²={best["r2_train"]:.4f}, 测试R²={best["r2_test"]:.4f}')
    draw_figures()
    print('04 done')


if __name__ == '__main__':
    main()
