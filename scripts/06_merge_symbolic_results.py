# -*- coding: utf-8 -*-
"""
05_merge_sr_scan.py
汇总 54 组符号回归参数扫描结果：
1. 合并全部 pareto.csv 点（按复杂度-MSE），计算跨参数设置的全局帕累托前沿
2. 从合并前沿按综合得分选最终公式（复杂度 8~16 均可选，score = -MSE - parsimony*complexity，
   选取时对每个复杂度档位保留最优 MSE 的公式），将最终公式的预测文件写回 data/sr_{target}_*（供图8与论文用）
3. 重绘图8（图6_sr_formula.png）：(a)(b) 帕累托前沿——全部扫描点灰点 + 全局前沿线 + 选中红星；
   (c)(d) 公式验证散点
环境：LubLLM
"""
import os
import json
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

TARGETS = ['yield_strength_mpa', 'ultimate_tensile_strength_mpa']
TARGET_EN = {'yield_strength_mpa': 'YS', 'ultimate_tensile_strength_mpa': 'UTS'}


def global_pareto(points):
    """points: (complexity, mse) 数组；返回全局帕累托前沿行索引（按复杂度升序）。"""
    order = np.argsort(points[:, 0])
    c = points[order, 0]
    m = points[order, 1]
    front = []
    best = np.inf
    for i in range(len(c)):
        if m[i] < best:
            front.append(order[i])
            best = m[i]
    return np.array(front)


def main():
    for t in TARGETS:
        # ---------- 合并所有扫描点 ----------
        all_rows = []
        for p in sorted(glob.glob(os.path.join(DATA_DIR, 'sr_scan', t, '*/pareto.csv'))):
            sub = pd.read_csv(p, encoding='utf-8-sig')
            sub['setting'] = os.path.basename(os.path.dirname(p))
            all_rows.append(sub)
        merged = pd.concat(all_rows, ignore_index=True)
        merged.to_csv(os.path.join(DATA_DIR, f'sr_{t}_scan_all.csv'), index=False, encoding='utf-8-sig')
        print(f'{t}: 扫描点总数 {len(merged)}, 来自 {merged["setting"].nunique()} 组参数设置')

        # ---------- 全局帕累托前沿 ----------
        pts = merged[['complexity', 'loss']].values
        f_idx = global_pareto(pts)
        front = merged.iloc[f_idx].sort_values('complexity')
        front.to_csv(os.path.join(DATA_DIR, f'sr_{t}_global_front.csv'), index=False,
                     encoding='utf-8-sig')
        print(f'{t}: 全局帕累托前沿点数 {len(front)}, 复杂度范围 '
              f'{front["complexity"].min()}~{front["complexity"].max()}')

        # ---------- 最终公式选择：比较各参数设置自身的最优公式（best.json） ----------
        # score = -MSE - 0.0032*complexity（与搜索一致），复杂度限定 8~16
        best_setting, best_score, best = None, -np.inf, None
        for sub_dir in glob.glob(os.path.join(DATA_DIR, 'sr_scan', t, '*/')):
            bj = os.path.join(sub_dir, 'best.json')
            if not os.path.exists(bj):
                continue
            with open(bj, encoding='utf-8') as f:
                d = json.load(f)
            if not (8 <= d['complexity'] <= 16):
                continue
            s = -d['loss'] - 0.0032 * d['complexity']
            if s > best_score:
                best_score, best_setting, best = s, os.path.basename(sub_dir.rstrip('/')), d
        if best is None:
            raise RuntimeError(f'{t}: 无符合复杂度 8~16 的结果')
        best_dir = os.path.join(DATA_DIR, 'sr_scan', t, best_setting)
        print(f'{t}: 最终公式来自 {best_setting}, 复杂度 {best["complexity"]}, '
              f'训练R²={best["r2_train"]:.4f}, 测试R²={best["r2_test"]:.4f}')
        print(f'   formula: {best["formula_text"][:160]}')

        # 将最终公式的预测文件与 best.json 写回标准位置（供论文生成脚本使用）
        import shutil
        shutil.copy(os.path.join(best_dir, 'best.json'),
                    os.path.join(DATA_DIR, f'sr_{t}_best.json'))
        shutil.copy(os.path.join(best_dir, 'pred_tr.csv'),
                    os.path.join(DATA_DIR, f'sr_{t}_pred_tr.csv'))
        shutil.copy(os.path.join(best_dir, 'pred_te.csv'),
                    os.path.join(DATA_DIR, f'sr_{t}_pred_te.csv'))
        shutil.copy(os.path.join(best_dir, 'pareto.csv'),
                    os.path.join(DATA_DIR, f'sr_{t}_pareto.csv'))

    # ---------- 图8 重绘 ----------
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
    for col, t in enumerate(TARGETS):
        merged = pd.read_csv(os.path.join(DATA_DIR, f'sr_{t}_scan_all.csv'), encoding='utf-8-sig')
        front = pd.read_csv(os.path.join(DATA_DIR, f'sr_{t}_global_front.csv'), encoding='utf-8-sig')
        best = json.load(open(os.path.join(DATA_DIR, f'sr_{t}_best.json'), encoding='utf-8'))

        ax = axes[0, col]
        ax.scatter(merged['complexity'], merged['loss'], s=45, c='#CFCECE', alpha=0.8,
                   edgecolors='none', label='Scan points (27 settings)')
        ax.plot(front['complexity'], front['loss'], '-', c='#767676', lw=2, zorder=4,
                label='Global Pareto front')
        ax.scatter(front['complexity'], front['loss'], s=55, c='#767676', alpha=0.9,
                   edgecolors='white', linewidths=0.5, zorder=5)
        ax.scatter(best['complexity'], best['loss'], s=340, c='#E53935', marker='*',
                   edgecolors='#4D4D4D', linewidths=0.8, zorder=6, label='Selected')
        ax.set_xlabel('Complexity', fontsize=14)
        ax.set_ylabel('Loss (MSE)', fontsize=14)
        ax.set_yscale('log')
        ax.text(0.03, 1.0, '(' + 'ab'[col] + ') ' + TARGET_EN[t] + ' Pareto front',
                transform=ax.transAxes, fontsize=15, fontweight='bold', va='bottom')
        ax.legend(fontsize=11, loc='upper right')
        ax.tick_params(labelsize=12)

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
    print('05_merge done')


if __name__ == '__main__':
    main()
