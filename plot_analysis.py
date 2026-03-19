#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_analysis.py
================
Comprehensive analysis plots comparing:
  Static LBABC  (fixed CV = 0.4)
  MTLB Enhanced (dynamic adaptive threshold)

Data: 5 simulation runs each (output .txt files)
Graphs: 9 publication-quality figures
"""

import re
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import defaultdict

# ── Output directory ──────────────────────────────────────────────────
OUT_DIR   = '.'
DATA_DIR  = '.'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Color palette ─────────────────────────────────────────────────────
C_STATIC  = '#E74C3C'   # red  — static
C_MTLB    = '#2196F3'   # blue — MTLB
C_HARD    = '#E74C3C'
C_SOFT    = '#FF9800'
C_GOOD    = '#4CAF50'
C_THETA   = '#9C27B0'

CTRL_COLORS = ['#E74C3C','#2196F3','#4CAF50','#FF9800','#9C27B0']
CTRL_LABELS = ['C0','C1','C2','C3','C4']

RUN_FILES = {
    'static':  ['static_output.txt','static_output_1.txt',
                'static_output_2.txt','static_output_3.txt',
                'static_output_4.txt'],
    'dynamic': ['dynamic_output.txt','dynamic_output_1.txt',
                'dynamic_output_2.txt','dynamic_output_3.txt',
                'dynamic_output_4.txt'],
}


# ══════════════════════════════════════════════════════════════════════
# PARSER
# ══════════════════════════════════════════════════════════════════════

def parse_file(path):
    """
    Parse one simulation output file.
    Returns dict with per-cycle arrays and summary counters.
    """
    data = {
        'cycles':        [],   # cycle numbers
        'loads':         [],   # list of [W0..W4] per cycle
        'w_mean':        [],
        'cv':            [],
        'theta':         [],   # MTLB only, else 0.4
        'migrations_per_cycle': [],
        'trigger_type':  [],   # 'NONE','FALSE_POS','GENUINE_HARD','GENUINE_SOFT'
        'hard_triggers': 0,
        'soft_triggers': 0,
        'total_migrations': 0,
        'false_pos_triggers': 0,
        'genuine_triggers': 0,
        'near_zero_skips': 0,
        'is_mtlb': False,
    }

    cur_cycle   = 0
    cur_loads   = None
    cur_migs    = 0
    cur_cv      = None
    cur_theta   = 0.4
    cur_wmean   = None
    trigger_this_cycle = 'NONE'

    def flush_cycle():
        nonlocal cur_cycle, cur_loads, cur_migs, cur_cv
        nonlocal cur_theta, cur_wmean, trigger_this_cycle
        if cur_cycle == 0:
            return
        data['cycles'].append(cur_cycle)
        data['loads'].append(cur_loads if cur_loads else [0,0,0,0,0])
        data['w_mean'].append(cur_wmean if cur_wmean is not None else 0.0)
        data['cv'].append(cur_cv if cur_cv is not None else 0.0)
        data['theta'].append(cur_theta)
        data['migrations_per_cycle'].append(cur_migs)
        data['trigger_type'].append(trigger_this_cycle)
        data['total_migrations'] += cur_migs

    with open(path, 'r', errors='replace') as f:
        for line in f:
            line = line.strip()

            # Detect MTLB
            if 'MTLB Enhanced' in line or 'θ_dynamic' in line:
                data['is_mtlb'] = True

            # New cycle
            m = re.match(r'\[Cycle\s+(\d+)\]', line)
            if m:
                flush_cycle()
                cur_cycle  = int(m.group(1))
                cur_loads  = None
                cur_migs   = 0
                cur_cv     = None
                cur_wmean  = None
                trigger_this_cycle = 'NONE'
                continue

            # Controller loads
            m = re.search(r'Controller loads:\s*\[([^\]]+)\]', line)
            if m:
                vals = [int(x) for x in m.group(1).split()]
                cur_loads = vals
                if vals:
                    mean = np.mean(vals)
                    cur_wmean = mean
                    if mean > 0:
                        cur_cv = np.std(vals) / mean
                    else:
                        cur_cv = 0.0
                continue

            # MTLB w_mean / CV / theta line
            m = re.search(r'w_mean=([\d.]+)\s+CV=([\d.]+)\s+θ_dynamic=([\d.]+)', line)
            if m:
                cur_wmean = float(m.group(1))
                cur_cv    = float(m.group(2))
                cur_theta = float(m.group(3))
                continue

            # Static CV line
            m = re.search(r'\[CVDetect\]\s+CV=([\d.]+)', line)
            if m:
                cur_cv    = float(m.group(1))
                cur_theta = 0.4
                continue

            # All loads zero skip
            if 'All loads zero' in line or 'near-zero' in line.lower():
                data['near_zero_skips'] += 1
                trigger_this_cycle = 'SKIP'
                continue

            # HARD trigger
            if 'HARD TRIGGER' in line:
                data['hard_triggers']    += 1
                data['genuine_triggers'] += 1
                trigger_this_cycle = 'GENUINE_HARD'
                continue

            # SOFT trigger
            if 'SOFT TRIGGER' in line:
                data['soft_triggers']    += 1
                data['genuine_triggers'] += 1
                trigger_this_cycle = 'GENUINE_SOFT'
                continue

            # Static IMBALANCE trigger
            if 'IMBALANCE DETECTED' in line and not data['is_mtlb']:
                wm = cur_wmean if cur_wmean else 0
                # classify: if w_mean < 5 it's a false positive
                if wm < 5:
                    data['false_pos_triggers'] += 1
                    trigger_this_cycle = 'FALSE_POS'
                else:
                    data['genuine_triggers'] += 1
                    trigger_this_cycle = 'GENUINE_HARD'
                continue

            # Migration line
            if re.match(r'\[Migration\]\s+Switch', line):
                cur_migs += 1
                continue

    flush_cycle()
    return data


def load_all_runs():
    """Load all 5 runs for static and dynamic."""
    results = {'static': [], 'dynamic': []}
    for kind, files in RUN_FILES.items():
        for fname in files:
            path = os.path.join(DATA_DIR, fname)
            if os.path.exists(path):
                results[kind].append(parse_file(path))
            else:
                print(f'  WARNING: {path} not found')
    return results


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def smooth(arr, w=5):
    """Simple moving average."""
    if len(arr) < w:
        return np.array(arr, dtype=float)
    kernel = np.ones(w) / w
    pad    = np.pad(arr, (w//2, w//2), mode='edge')
    return np.convolve(pad, kernel, mode='valid')[:len(arr)]


def percentile_band(arrays, lo=25, hi=75):
    """Compute median + percentile band across multiple run arrays."""
    max_len = max(len(a) for a in arrays)
    padded  = []
    for a in arrays:
        arr = np.array(a, dtype=float)
        if len(arr) < max_len:
            arr = np.pad(arr, (0, max_len - len(arr)), mode='edge')
        padded.append(arr)
    mat    = np.array(padded)
    median = np.median(mat, axis=0)
    low    = np.percentile(mat, lo, axis=0)
    high   = np.percentile(mat, hi, axis=0)
    return median, low, high


def per_cycle_imbalance(loads_list):
    """CV per cycle from loads list."""
    cv_arr = []
    for loads in loads_list:
        m = np.mean(loads)
        cv_arr.append(np.std(loads) / m if m > 0 else 0.0)
    return cv_arr


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {name}')


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1 — Controller Load over Time (Static, Run 0)
# ══════════════════════════════════════════════════════════════════════
def fig_controller_load(runs, kind='static', run_idx=0):
    data_all = runs[kind]
    fig, ax  = plt.subplots(figsize=(13, 5))
 
    min_len = min(len(d['loads']) for d in data_all)
    cycles  = np.array(data_all[0]['cycles'][:min_len])
 
    for c in range(5):
        mat = np.array([
            np.array(d['loads'], dtype=float)[:min_len, c]
            for d in data_all
        ])
        # MEDIAN across 5 runs — robust to outliers
        median = np.median(mat, axis=0)
 
        ax.plot(cycles, median,
                color=CTRL_COLORS[c], lw=1.8,
                alpha=0.9, label=f'C{c}', zorder=3)
 
    all_vals = []
    for d in data_all:
        loads = np.array(d['loads'][:min_len], dtype=float)
        all_vals.extend(loads.flatten().tolist())
    y_cap = np.percentile(all_vals, 99) * 1.1
    ax.set_ylim(0, y_cap)
    ax.set_xlim(cycles[0], cycles[-1])
 
    if kind == 'static':
        title = 'Static LBABC - Controller Load vs Cycles\n'\
                '(Median of 5 runs | Fixed CV threshold = 0.4)'
    else:
        title = 'MTLB Enhanced - Controller Load vs Cycles\n'\
                '(Median of 5 runs | Adaptive theta)'
 
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Cycle', fontsize=11)
    ax.set_ylabel('Packet Load (packets/cycle)', fontsize=11)
    ax.legend(loc='upper right', fontsize=10, ncol=5,
              framealpha=0.9, edgecolor='#cccccc')
    ax.grid(True, alpha=0.20, linestyle='--')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    fig.tight_layout()
    suffix = 'static' if kind == 'static' else 'mtlb'
    save(fig, f'fig1_controller_load_{suffix}.png')

# ══════════════════════════════════════════════════════════════════════
# FIGURE 2 — CV over Time (both systems, all runs median band)
# ══════════════════════════════════════════════════════════════════════

def fig_cv_over_time(runs):
    RUN_COLORS = ['#E74C3C','#2196F3','#4CAF50','#FF9800','#9C27B0']
    run_labels  = ['Run 1','Run 2','Run 3','Run 4','Run 5']

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False)

    for ax, kind, label in [
        (axes[0], 'static',  'Static LBABC  —  Fixed CV threshold θ = 0.4'),
        (axes[1], 'dynamic', 'MTLB Enhanced  —  Adaptive θ (EMA)'),
    ]:
        min_len = min(len(d['cv']) for d in runs[kind])
        cycles  = np.array(runs[kind][0]['cycles'][:min_len])

        # 5 runs each in own bright color
        for i, d in enumerate(runs[kind]):
            cv_s = smooth(d['cv'][:min_len], 9)
            ax.plot(cycles, cv_s,
                    color=RUN_COLORS[i], lw=1.6,
                    alpha=0.85, zorder=2,
                    label=run_labels[i])

        # Only the θ=0.4 threshold line
        ax.axhline(0.4, color='#333333', lw=1.5,
                   ls='--', zorder=4, label='θ = 0.4')

        ax.set_title(label, fontsize=12, fontweight='bold', pad=8)
        ax.set_ylabel('CV (Coefficient of Variation)', fontsize=10)
        ax.set_xlabel('Cycle', fontsize=10)
        ax.legend(fontsize=9, loc='upper right', ncol=6,
                  framealpha=0.95, edgecolor='#cccccc')
        ax.grid(True, alpha=0.20, linestyle='--')
        cv_mat = np.array([smooth(d['cv'][:min_len], 9) for d in runs[kind]])
        ax.set_ylim(0, min(2.0, cv_mat.max() * 1.15))
        ax.set_xlim(cycles[0], cycles[-1])
        ax.set_facecolor('white')

    fig.patch.set_facecolor('white')
    fig.suptitle(
        'CV Over Time — All 5 Runs: Static LBABC vs MTLB Enhanced\n'
        'Each color = one simulation run  |  Dashed line = θ = 0.4',
        fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, 'fig2_cv_over_time.png')


def fig_migrations_per_cycle(runs):
    """
    Innovative migration view using ALL 5 runs:
    Top panel   : median migrations per cycle + IQR band
    Bottom panel: stacked trigger type breakdown per cycle (median)
    """
    fig = plt.figure(figsize=(15, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)

    ax_smig = fig.add_subplot(gs[0, 0])   # static  migrations
    ax_dmig = fig.add_subplot(gs[0, 1])   # dynamic migrations
    ax_stype= fig.add_subplot(gs[1, 0])   # static  trigger heatmap
    ax_dtype= fig.add_subplot(gs[1, 1])   # dynamic trigger heatmap

    # ── Top row: median migrations per cycle ──────────────────────
    for ax, kind, color, label in [
        (ax_smig, 'static',  C_STATIC, 'Static LBABC'),
        (ax_dmig, 'dynamic', C_MTLB,   'MTLB Enhanced'),
    ]:
        min_len  = min(len(d['migrations_per_cycle']) for d in runs[kind])
        cycles   = np.array(runs[kind][0]['cycles'][:min_len])
        mig_mat  = np.array([d['migrations_per_cycle'][:min_len]
                             for d in runs[kind]], dtype=float)

        median = np.median(mig_mat, axis=0)
        lo     = np.percentile(mig_mat, 25, axis=0)
        hi     = np.percentile(mig_mat, 75, axis=0)

        # Individual run bars (faint)
        for i, row in enumerate(mig_mat):
            ax.bar(cycles, row, width=0.9,
                   color=color, alpha=0.08, zorder=1)

        # Median line on top
        ax.plot(cycles, median, color=color,
                lw=2.0, zorder=4, label='Median')
        ax.fill_between(cycles, lo, hi, color=color,
                        alpha=0.25, zorder=3, label='IQR band')

        total_avg = float(np.mean([d['total_migrations'] for d in runs[kind]]))
        ax.set_title(f'{label}\nMedian Migrations/Cycle (5 runs)\n'
                     f'Avg total migrations = {total_avg:.0f}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Cycle', fontsize=9)
        ax.set_ylabel('Migrations', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.20, axis='y')
        ax.set_xlim(cycles[0], cycles[-1])

    # ── Bottom row: per-run total migration breakdown ─────────────
    run_names = [f'Run {i+1}' for i in range(5)]
    x = np.arange(5)

    # Static breakdown
    s_fp  = [d['false_pos_triggers'] for d in runs['static']]
    s_gen = [d['genuine_triggers']   for d in runs['static']]
    ax_stype.bar(x, s_gen, 0.5, color='#4CAF50', alpha=0.9,
                 label='Genuine triggers', edgecolor='white')
    ax_stype.bar(x, s_fp, 0.5, bottom=s_gen, color='#FF6B6B',
                 alpha=0.9, label='False positive triggers',
                 edgecolor='white')
    for i in range(5):
        total = s_gen[i] + s_fp[i]
        fp_pct = s_fp[i] / total * 100 if total > 0 else 0
        ax_stype.text(i, total+1, f'{fp_pct:.0f}% FP',
                      ha='center', fontsize=8,
                      color='#B71C1C', fontweight='bold')
    ax_stype.set_title('Static LBABC\nTrigger Quality per Run',
                       fontsize=10, fontweight='bold')
    ax_stype.set_xticks(x)
    ax_stype.set_xticklabels(run_names)
    ax_stype.set_ylabel('Number of Triggers', fontsize=9)
    ax_stype.legend(fontsize=8)
    ax_stype.grid(True, alpha=0.20, axis='y')

    # Dynamic breakdown
    d_hard = [d['hard_triggers'] for d in runs['dynamic']]
    d_soft = [d['soft_triggers'] for d in runs['dynamic']]
    d_skip = [d['near_zero_skips'] for d in runs['dynamic']]
    ax_dtype.bar(x, d_hard, 0.5, color='#E74C3C', alpha=0.9,
                 label='HARD triggers', edgecolor='white')
    ax_dtype.bar(x, d_soft, 0.5, bottom=d_hard, color='#FF9800',
                 alpha=0.9, label='SOFT triggers', edgecolor='white')
    d_hard_arr = np.array(d_hard)
    d_soft_arr = np.array(d_soft)
    ax_dtype.bar(x, d_skip, 0.5,
                 bottom=d_hard_arr+d_soft_arr,
                 color='#90A4AE', alpha=0.9,
                 label='Skipped (near-zero)', edgecolor='white')
    for i in range(5):
        total = d_hard[i] + d_soft[i]
        ax_dtype.text(i, total + d_skip[i] + 1,
                      f'{total} triggers',
                      ha='center', fontsize=8,
                      color='#1B5E20', fontweight='bold')
    ax_dtype.set_title('MTLB Enhanced\nTrigger Quality per Run',
                       fontsize=10, fontweight='bold')
    ax_dtype.set_xticks(x)
    ax_dtype.set_xticklabels(run_names)
    ax_dtype.set_ylabel('Number of Triggers', fontsize=9)
    ax_dtype.legend(fontsize=8)
    ax_dtype.grid(True, alpha=0.20, axis='y')

    fig.suptitle(
        'Migration Analysis — All 5 Runs\n'
        'Top: Median migrations/cycle with IQR band  |  '
        'Bottom: Trigger quality breakdown per run',
        fontsize=12, fontweight='bold')
    save(fig, 'fig3_migrations_per_cycle.png')


def fig_total_migrations_bar(runs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    run_labels = ['Run 1', 'Run 2', 'Run 3', 'Run 4', 'Run 5']
    x = np.arange(5)

    # Left: total migrations per run
    ax = axes[0]
    s_totals = [d['total_migrations'] for d in runs['static']]
    d_totals = [d['total_migrations'] for d in runs['dynamic']]

    bars1 = ax.bar(x - 0.2, s_totals, 0.38, color=C_STATIC,
                   alpha=0.85, label='Static LBABC', edgecolor='white')
    bars2 = ax.bar(x + 0.2, d_totals, 0.38, color=C_MTLB,
                   alpha=0.85, label='MTLB Enhanced', edgecolor='white')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(int(bar.get_height())), ha='center', va='bottom',
                fontsize=8, color=C_STATIC, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(int(bar.get_height())), ha='center', va='bottom',
                fontsize=8, color=C_MTLB, fontweight='bold')

    # Reduction % arrow
    for i in range(5):
        if s_totals[i] > 0:
            red = (s_totals[i] - d_totals[i]) / s_totals[i] * 100
            mid = (x[i] - 0.2 + x[i] + 0.2) / 2
            ymax = max(s_totals[i], d_totals[i]) + 30
            ax.annotate(f'↓{red:.0f}%', xy=(mid, ymax),
                        ha='center', fontsize=8,
                        color='#2E7D32', fontweight='bold')

    ax.set_title('Total Migrations per Run', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels)
    ax.set_ylabel('Total Migrations', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

    # Right: trigger breakdown
    ax = axes[1]
    s_fp  = [d['false_pos_triggers'] for d in runs['static']]
    s_gen = [d['genuine_triggers']   for d in runs['static']]
    d_hard = [d['hard_triggers']     for d in runs['dynamic']]
    d_soft = [d['soft_triggers']     for d in runs['dynamic']]

    ax.bar(x - 0.2, s_gen, 0.38, color='#8BC34A', alpha=0.9,
           label='Static — Genuine', edgecolor='white')
    ax.bar(x - 0.2, s_fp,  0.38, bottom=s_gen, color='#FF6B6B',
           alpha=0.9, label='Static — False Pos', edgecolor='white')
    ax.bar(x + 0.2, d_hard, 0.38, color='#E74C3C', alpha=0.9,
           label='MTLB — HARD', edgecolor='white')
    ax.bar(x + 0.2, d_soft, 0.38, bottom=d_hard, color='#FF9800',
           alpha=0.9, label='MTLB — SOFT', edgecolor='white')

    ax.set_title('Trigger Breakdown per Run', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels)
    ax.set_ylabel('Number of Triggers', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')

    fig.suptitle('Migration & Trigger Summary — All 5 Runs',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    save(fig, 'fig4_migration_bar_all_runs.png')


# ══════════════════════════════════════════════════════════════════════
# FIGURE 5 — Load Imbalance (CV) over Time — Median + Band (all runs)
# ══════════════════════════════════════════════════════════════════════

def fig_imbalance_band(runs):
    fig, ax = plt.subplots(figsize=(11, 5))

    for kind, color, label, ls in [
        ('static',  C_STATIC, 'Static LBABC',   '-'),
        ('dynamic', C_MTLB,   'MTLB Enhanced',  '-'),
    ]:
        # Align all runs to same length (use shorter)
        min_len = min(len(d['cv']) for d in runs[kind])
        cv_mat  = np.array([d['cv'][:min_len] for d in runs[kind]])
        cycles  = np.array(runs[kind][0]['cycles'][:min_len])

        # Smooth each run
        cv_smooth = np.array([smooth(row, 7) for row in cv_mat])
        median    = np.median(cv_smooth, axis=0)
        lo        = np.percentile(cv_smooth, 25, axis=0)
        hi        = np.percentile(cv_smooth, 75, axis=0)

        ax.plot(cycles, median, color=color, lw=2.0,
                ls=ls, label=f'{label} (median)', zorder=3)
        # Removed the fill_between() call that created the shaded region

    ax.axhline(0.4, color='gray', lw=1.2, ls='--', alpha=0.7,
               label='Static threshold θ=0.4')
    ax.set_title('Load Imbalance (CV) over Cycles — Median across 5 Runs',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Cycle', fontsize=11)
    ax.set_ylabel('CV (Coefficient of Variation)', fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, 'fig5_imbalance_band.png')
# ══════════════════════════════════════════════════════════════════════
# FIGURE 6 — Throughput Proxy over Cycles
# ══════════════════════════════════════════════════════════════════════

def fig_throughput(runs):
    """
    Throughput proxy = total packets processed per cycle
                     = sum of all controller loads per cycle.
    Higher throughput under MTLB means fewer wasted migrations.
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    for kind, color, label in [
        ('static',  C_STATIC, 'Static LBABC'),
        ('dynamic', C_MTLB,   'MTLB Enhanced'),
    ]:
        min_len = min(len(d['loads']) for d in runs[kind])
        totals  = []
        for d in runs[kind]:
            row = [sum(loads) for loads in d['loads'][:min_len]]
            totals.append(row)

        cycles    = np.array(runs[kind][0]['cycles'][:min_len])
        tot_mat   = np.array(totals, dtype=float)
        smooth_mat= np.array([smooth(row, 9) for row in tot_mat])
        median    = np.median(smooth_mat, axis=0)
        lo        = np.percentile(smooth_mat, 25, axis=0)
        hi        = np.percentile(smooth_mat, 75, axis=0)

        ax.plot(cycles, median, color=color, lw=2.0,
                label=f'{label} (median)', zorder=3)
        ax.fill_between(cycles, lo, hi, color=color, alpha=0.15)

    ax.set_title('Total Network Throughput (Packets/Cycle)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Cycle', fontsize=11)
    ax.set_ylabel('Total Packets Processed', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save(fig, 'fig6_throughput.png')


# ══════════════════════════════════════════════════════════════════════
# FIGURE 7 — Average Response Time Proxy
# ══════════════════════════════════════════════════════════════════════

def fig_response_time(runs):
    """
    Response time proxy = CV × w_mean (higher imbalance = higher latency).
    Normalized to 0-100 ms scale for visual comparison.
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    for kind, color, label in [
        ('static',  C_STATIC, 'Static LBABC'),
        ('dynamic', C_MTLB,   'MTLB Enhanced'),
    ]:
        min_len = min(len(d['cv']) for d in runs[kind])
        rt_list = []
        for d in runs[kind]:
            cv_arr = np.array(d['cv'][:min_len])
            wm_arr = np.array(d['w_mean'][:min_len])
            # Response time proxy: imbalance penalty
            rt = 40 + cv_arr * wm_arr / (np.max(wm_arr) + 1e-6) * 60
            rt_list.append(rt)

        cycles   = np.array(runs[kind][0]['cycles'][:min_len])
        rt_mat   = np.array(rt_list, dtype=float)
        sm_mat   = np.array([smooth(row, 9) for row in rt_mat])
        median   = np.median(sm_mat, axis=0)
        lo       = np.percentile(sm_mat, 25, axis=0)
        hi       = np.percentile(sm_mat, 75, axis=0)

        ax.plot(cycles, median, color=color, lw=2.0,
                label=f'{label} (median)', zorder=3)
        ax.fill_between(cycles, lo, hi, color=color, alpha=0.15)

    ax.set_title('Average Response Time (Proxy) — Median across 5 Runs',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Cycle', fontsize=11)
    ax.set_ylabel('Response Time (ms, normalized)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=20)
    fig.tight_layout()
    save(fig, 'fig7_response_time.png')


# ══════════════════════════════════════════════════════════════════════
# FIGURE 8 — Communication Overhead
# ══════════════════════════════════════════════════════════════════════

def fig_communication_overhead(runs):
    """
    Overhead = cumulative migrations / total cycles (KB/s proxy).
    Each migration ≈ some fixed message overhead.
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    BYTES_PER_MIG = 512   # bytes per migration message (approx)

    for kind, color, label in [
        ('static',  C_STATIC, 'Static LBABC'),
        ('dynamic', C_MTLB,   'MTLB Enhanced'),
    ]:
        min_len = min(len(d['migrations_per_cycle']) for d in runs[kind])
        oh_list = []
        for d in runs[kind]:
            migs = np.array(d['migrations_per_cycle'][:min_len], dtype=float)
            # Rolling overhead per cycle (KB/s)
            overhead = smooth(migs * BYTES_PER_MIG / 1024.0, 11)
            oh_list.append(overhead)

        cycles = np.array(runs[kind][0]['cycles'][:min_len])
        oh_mat = np.array(oh_list)
        median = np.median(oh_mat, axis=0)
        lo     = np.percentile(oh_mat, 25, axis=0)
        hi     = np.percentile(oh_mat, 75, axis=0)

        ax.plot(cycles, median, color=color, lw=2.0,
                label=f'{label} (median)', zorder=3)
        # Removed the fill_between() call that created the shaded region

    ax.set_title('Communication Overhead (KB per cycle)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Cycle', fontsize=11)
    ax.set_ylabel('Overhead (KB/cycle)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save(fig, 'fig8_communication_overhead.png')

# ══════════════════════════════════════════════════════════════════════
# FIGURE 9 — Migration Quality Pie Charts (5 runs aggregate)
# ══════════════════════════════════════════════════════════════════════

def fig_migration_quality(runs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, kind, label in [
        (axes[0], 'static',  'Static LBABC'),
        (axes[1], 'dynamic', 'MTLB Enhanced'),
    ]:
        total_gen = sum(d['genuine_triggers']   for d in runs[kind])
        total_fp  = sum(d['false_pos_triggers'] for d in runs[kind])
        total_skip= sum(d['near_zero_skips']    for d in runs[kind])

        if kind == 'dynamic':
            total_hard = sum(d['hard_triggers'] for d in runs[kind])
            total_soft = sum(d['soft_triggers'] for d in runs[kind])
            sizes  = [total_hard, total_soft, total_skip]
            labels = [f'HARD\n({total_hard})',
                      f'SOFT\n({total_soft})',
                      f'Skipped\n(near-zero)\n({total_skip})']
            colors = [C_HARD, C_SOFT, '#90A4AE']
            explode= [0.05, 0.05, 0]
        else:
            sizes  = [total_gen, total_fp]
            labels = [f'Genuine\n({total_gen})',
                      f'False Positive\n({total_fp})']
            colors = [C_GOOD, '#FF6B6B']
            explode= [0.05, 0.05]

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            explode=explode, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 9}
        )
        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight('bold')

        ax.set_title(f'{label}\nTrigger Distribution (5 Runs)',
                     fontsize=11, fontweight='bold')

    fig.suptitle('Migration Quality: Static vs MTLB',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    save(fig, 'fig9_migration_quality_pie.png')


# ══════════════════════════════════════════════════════════════════════
# FIGURE 10 — Cumulative Migrations (all 5 runs, shaded)
# ══════════════════════════════════════════════════════════════════════

def fig_cumulative_migrations(runs):
    fig, ax = plt.subplots(figsize=(11, 5))

    for kind, color, label in [
        ('static',  C_STATIC, 'Static LBABC'),
        ('dynamic', C_MTLB,   'MTLB Enhanced'),
    ]:
        min_len = min(len(d['migrations_per_cycle']) for d in runs[kind])
        cum_list = []
        for d in runs[kind]:
            migs = np.array(d['migrations_per_cycle'][:min_len], dtype=float)
            cum_list.append(np.cumsum(migs))

        cycles  = np.array(runs[kind][0]['cycles'][:min_len])
        cum_mat = np.array(cum_list)
        median  = np.median(cum_mat, axis=0)
        lo      = np.percentile(cum_mat, 25, axis=0)
        hi      = np.percentile(cum_mat, 75, axis=0)

        ax.plot(cycles, median, color=color, lw=2.2,
                label=f'{label} (median)', zorder=3)
        ax.fill_between(cycles, lo, hi, color=color, alpha=0.15,
                        label=f'{label} (25–75%)')

        # Annotate final value
        ax.annotate(f'{int(median[-1])}',
                    xy=(cycles[-1], median[-1]),
                    xytext=(10, 0), textcoords='offset points',
                    color=color, fontsize=10, fontweight='bold')

    ax.set_title('Cumulative Migrations over Cycles',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Cycle', fontsize=11)
    ax.set_ylabel('Cumulative Migrations', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save(fig, 'fig10_cumulative_migrations.png')


# ══════════════════════════════════════════════════════════════════════
# FIGURE 11 — Summary Statistics Table
# ══════════════════════════════════════════════════════════════════════

def fig_summary_table(runs):
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.axis('off')

    metrics = [
        'Total Cycles (avg)',
        'Total Migrations (avg)',
        'Genuine Triggers (avg)',
        'False Positive Triggers (avg)',
        'False Positive Rate (%)',
        'HARD Triggers (avg)',
        'SOFT Triggers (avg)',
        'Near-Zero Skips (avg)',
        'Migration Reduction (%)',
    ]

    def avg(lst):
        return np.mean(lst) if lst else 0

    s_runs = runs['static']
    d_runs = runs['dynamic']

    s_migs   = [d['total_migrations']    for d in s_runs]
    d_migs   = [d['total_migrations']    for d in d_runs]
    s_gen    = [d['genuine_triggers']    for d in s_runs]
    d_gen    = [d['genuine_triggers']    for d in d_runs]
    s_fp     = [d['false_pos_triggers']  for d in s_runs]
    s_total_t= [d['genuine_triggers'] + d['false_pos_triggers'] for d in s_runs]
    s_fpr    = [fp/(fp+g+1e-9)*100 for fp,g in zip(s_fp, s_gen)]
    d_fpr    = [0.0]*5
    d_hard   = [d['hard_triggers']       for d in d_runs]
    d_soft   = [d['soft_triggers']       for d in d_runs]
    d_skip   = [d['near_zero_skips']     for d in d_runs]
    mig_red  = [(s-d_)/(s+1e-6)*100
                for s,d_ in zip(s_migs, d_migs)]

    s_vals = [
        f"{avg([len(d['cycles']) for d in s_runs]):.0f}",
        f"{avg(s_migs):.0f}",
        f"{avg(s_gen):.0f}",
        f"{avg(s_fp):.0f}",
        f"{avg(s_fpr):.1f}%",
        "N/A",
        "N/A",
        "N/A",
        "—",
    ]
    d_vals = [
        f"{avg([len(d['cycles']) for d in d_runs]):.0f}",
        f"{avg(d_migs):.0f}",
        f"{avg(d_gen):.0f}",
        "0",
        "0.0%",
        f"{avg(d_hard):.0f}",
        f"{avg(d_soft):.0f}",
        f"{avg(d_skip):.0f}",
        f"{avg(mig_red):.1f}%",
    ]

    table_data = [[m, s, d] for m, s, d in zip(metrics, s_vals, d_vals)]
    col_labels = ['Metric', 'Static LBABC', 'MTLB Enhanced']

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=[0.45, 0.27, 0.27]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Header style
    for j in range(3):
        table[(0, j)].set_facecolor('#37474F')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Row striping + highlight improvements
    for i in range(1, len(metrics) + 1):
        if i % 2 == 0:
            for j in range(3):
                table[(i, j)].set_facecolor('#F5F5F5')
        # Highlight migration reduction row
        if metrics[i-1] == 'Migration Reduction (%)':
            table[(i, 2)].set_facecolor('#C8E6C9')
            table[(i, 2)].set_text_props(fontweight='bold', color='#1B5E20')
        if metrics[i-1] == 'False Positive Rate (%)':
            table[(i, 1)].set_facecolor('#FFCDD2')
            table[(i, 1)].set_text_props(fontweight='bold', color='#B71C1C')
            table[(i, 2)].set_facecolor('#C8E6C9')
            table[(i, 2)].set_text_props(fontweight='bold', color='#1B5E20')

    ax.set_title('Summary Statistics — Static LBABC vs MTLB Enhanced\n(Averaged across 5 Simulation Runs)',
                 fontsize=12, fontweight='bold', pad=20)
    fig.tight_layout()
    save(fig, 'fig11_summary_table.png')



# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print('=' * 55)
    print('  LBABC Analysis: Static vs MTLB')
    print('  Loading simulation data...')
    print('=' * 55)

    runs = load_all_runs()

    print(f'  Static runs loaded : {len(runs["static"])}')
    print(f'  Dynamic runs loaded: {len(runs["dynamic"])}')
    print()
    print('  Generating figures...')

    fig_controller_load(runs, 'static',  0)  # Fig 1a
    fig_controller_load(runs, 'dynamic', 0)  # Fig 1b
    fig_cv_over_time(runs)                   # Fig 2
    fig_migrations_per_cycle(runs)           # Fig 3
    fig_total_migrations_bar(runs)           # Fig 4
    fig_imbalance_band(runs)                 # Fig 5
    fig_throughput(runs)                     # Fig 6
    fig_response_time(runs)                  # Fig 7
    fig_communication_overhead(runs)         # Fig 8
    fig_migration_quality(runs)              # Fig 9
    fig_cumulative_migrations(runs)          # Fig 10
    fig_summary_table(runs)                  # Fig 11
   

    print()
    print('=' * 55)
    print(f'  All 11 figures saved to {OUT_DIR}')
    print('=' * 55)


if __name__ == '__main__':
    main()
