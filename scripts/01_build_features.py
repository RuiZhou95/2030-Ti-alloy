# -*- coding: utf-8 -*-
"""
01_feature_screen.py
特征定义与工程化（37 特征 = 23 成分 + 11 工艺 + 3 工程特征；类别编码不纳入特征集），
成分/工艺-性能相关性分析。
输出：data/02_engineered_features.csv、data/correlation_matrix.csv、figure/fig2_corr_heatmap.png
环境：LubLLM
"""
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

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

# 37 特征 = 23 成分 + 11 工艺 + 3 工程特征
COMP_COLS = ['Al_wt_pct', 'Ti_wt_pct', 'Fe_wt_pct', 'Ni_wt_pct', 'Cr_wt_pct',
             'Mn_wt_pct', 'Si_wt_pct', 'Cu_wt_pct', 'Zn_wt_pct', 'Mg_wt_pct',
             'Mo_wt_pct', 'V_wt_pct', 'Nb_wt_pct', 'Zr_wt_pct', 'Sn_wt_pct',
             'Co_wt_pct', 'C_wt_pct', 'N_wt_pct', 'O_wt_pct', 'B_wt_pct',
             'P_wt_pct', 'S_wt_pct', 'H']
PROC_COLS = ['homogenization_temp_c', 'homogenization_time_h',
             'solution_temp_c', 'solution_time_h',
             'aging_temp_c', 'aging_time_h',
             'annealing_temp_c', 'annealing_time_h',
             'rolling_temp_c', 'rolling_reduction_pct', 'specimen_thickness_mm']

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


def build_features():
    df = pd.read_csv(os.path.join(DATA_DIR, '01_processed_data.csv'), encoding='utf-8-sig')
    # 成分/工艺列数值化（'无'/'不关注'→0，NaN→0）
    for c in COMP_COLS + PROC_COLS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].apply(lambda v: 0 if v in ('无', '不关注', '/', '', None) else v)
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    # 工程特征
    df['Ti_x_Al'] = df['Ti_wt_pct'] * df['Al_wt_pct']
    df['Ti_x_V'] = df['Ti_wt_pct'] * df['V_wt_pct']
    df['total_alloy_wt'] = df[COMP_COLS].drop(columns=['Ti_wt_pct']).sum(axis=1)

    feature_cols = COMP_COLS + PROC_COLS + ['Ti_x_Al', 'Ti_x_V', 'total_alloy_wt']
    # 剔除常量列
    nonconst = [c for c in feature_cols if df[c].nunique() > 1]
    print('特征数（剔除常量后）:', len(nonconst))
    df[['sample_id', 'material_system', 'alloy_or_coating_name'] + nonconst + TARGETS].to_csv(
        os.path.join(DATA_DIR, '02_engineered_features.csv'), index=False, encoding='utf-8-sig')
    return df, nonconst


def _distcorr(x, y):
    """距离相关系数（distance correlation）。"""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    dx = np.abs(x[:, None] - x[None, :])
    dy = np.abs(y[:, None] - y[None, :])
    dx_row, dx_col = dx.mean(axis=1, keepdims=True), dx.mean(axis=0, keepdims=True)
    dy_row, dy_col = dy.mean(axis=1, keepdims=True), dy.mean(axis=0, keepdims=True)
    A = dx - dx_row - dx_col + dx.mean()
    B = dy - dy_row - dy_col + dy.mean()
    dcov2 = (A * B).mean()
    dvar_x = (A * A).mean()
    dvar_y = (B * B).mean()
    denom = np.sqrt(dvar_x * dvar_y)
    return np.sqrt(dcov2 / denom) if denom > 0 else 0.0


def correlation_analysis(df, feature_cols):
    """Pearson/Spearman/距离相关矩阵（特征 × 3 目标）。"""
    rows = []
    for c in feature_cols:
        row = {'feature': c}
        for t in TARGETS:
            valid = df[[c, t]].dropna()
            rp, _ = pearsonr(valid[c], valid[t])
            rs, _ = spearmanr(valid[c], valid[t])
            rd = _distcorr(valid[c].values, valid[t].values)
            row[t + '_pearson'] = rp
            row[t + '_spearman'] = rs
            row[t + '_distance'] = rd
        rows.append(row)
    corr = pd.DataFrame(rows).set_index('feature')
    corr.to_csv(os.path.join(DATA_DIR, 'correlation_matrix.csv'), encoding='utf-8-sig')

    # 按 Pearson 绝对值平均排序取前 12
    corr['_score'] = corr[[t + '_pearson' for t in TARGETS]].abs().mean(axis=1)
    top = corr.sort_values('_score', ascending=False).head(12)

    # 合并式三面板：共享纵轴（特征名只显示一次）、顶部大括弧标注三种相关系数
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.4), sharey=True)
    fig.subplots_adjust(left=0.17, right=0.84, top=0.84, bottom=0.12, wspace=0.05)
    specs = [('pearson', 'Pearson', (-1, 1)), ('spearman', 'Spearman', (-1, 1)),
             ('distance', 'Distance', (0, 1))]

    def top_bracket(ax, label):
        y_top, y_tick = 1.035, 1.008
        kw = dict(transform=ax.transAxes, color='#4D4D4D', lw=1.4, clip_on=False)
        ax.plot([0.08, 0.92], [y_top, y_top], **kw)
        ax.plot([0.08, 0.08], [y_top, y_tick], **kw)
        ax.plot([0.92, 0.92], [y_top, y_tick], **kw)
        ax.text(0.5, 1.075, label, transform=ax.transAxes, ha='center',
                va='bottom', fontsize=15, fontweight='bold')
        ax.text(0.02, 1.075, '(' + 'abc'[list(axes).index(ax)] + ')',
                transform=ax.transAxes, fontsize=15, fontweight='bold', va='bottom')

    for ax, (method, title, (vmin, vmax)) in zip(axes, specs):
        mat = top[[t + '_' + method for t in TARGETS]]
        im = ax.imshow(mat.values, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(3))
        ax.set_xticklabels(['YS', 'UTS', 'EL'], fontsize=14)
        ax.tick_params(labelsize=12)
        ax.set_yticks(range(len(mat)))
        if ax is axes[0]:
            ax.set_yticklabels([FEATURE_EN.get(f, f) for f in mat.index], fontsize=12)
        else:
            ax.tick_params(left=False, labelleft=False)
        top_bracket(ax, title)
        for i in range(len(mat)):
            for j in range(3):
                ax.text(j, i, f'{mat.values[i, j]:.2f}', ha='center', va='center',
                        fontsize=10.5, color='black')
    # 双色条：Pearson/Spearman 共用 −1~1；Distance 单独 0~1
    cbar_ps = fig.add_axes([0.865, 0.42, 0.014, 0.30])
    cb1 = fig.colorbar(axes[0].images[0], cax=cbar_ps)
    cb1.set_label('Correlation\n(Pearson / Spearman)', fontsize=12)
    cb1.ax.tick_params(labelsize=11)
    cbar_d = fig.add_axes([0.865, 0.14, 0.014, 0.22])
    cb2 = fig.colorbar(axes[2].images[0], cax=cbar_d)
    cb2.set_label('Distance\ncorrelation', fontsize=12)
    cb2.ax.tick_params(labelsize=11)
    out = os.path.join(FIG_DIR, 'fig2_corr_heatmap.png')
    fig.savefig(out, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print('saved', out)
    return corr


def main():
    df, feature_cols = build_features()
    print('feature_cols:', feature_cols)
    corr = correlation_analysis(df, feature_cols)
    print(corr[['yield_strength_mpa_pearson', 'ultimate_tensile_strength_mpa_pearson',
                'elongation_pct_pearson']].head(12).round(3).to_string())
    print('01 done')


if __name__ == '__main__':
    main()
