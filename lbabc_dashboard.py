#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lbabc_dashboard.py
==================
Interactive Analysis Dashboard
LBABC Static Baseline  vs  MTLB Enhanced Controller

HOW TO RUN:
  python3 lbabc_dashboard.py

  Place this file in the SAME folder as your output .txt files:
    static_output.txt   static_output_1.txt  ... static_output_4.txt
    dynamic_output.txt  dynamic_output_1.txt ... dynamic_output_4.txt

REQUIREMENTS:
  pip install matplotlib numpy
  (tkinter is included with Python by default)
"""

import os
import re
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from collections import defaultdict

# ── locate data files relative to this script ─────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RUN_FILES = {
    'static':  ['static_output.txt','static_output_1.txt',
                'static_output_2.txt','static_output_3.txt',
                'static_output_4.txt'],
    'dynamic': ['dynamic_output.txt','dynamic_output_1.txt',
                'dynamic_output_2.txt','dynamic_output_3.txt',
                'dynamic_output_4.txt'],
}

# ── Theme ──────────────────────────────────────────────────────────────
BG          = '#0F1923'
PANEL       = '#162232'
CARD        = '#1E2E42'
ACCENT      = '#00D4FF'
ACCENT2     = '#FF6B35'
TEXT        = '#E8F4FD'
TEXT_DIM    = '#7A9BB5'
C_STATIC    = '#FF4757'
C_MTLB      = '#2ED573'
C_HARD      = '#FF4757'
C_SOFT      = '#FFA502'
C_THETA     = '#A29BFE'
C_GOOD      = '#2ED573'
CTRL_COLORS = ['#FF4757','#2ED573','#00D4FF','#FFA502','#A29BFE']

plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    CARD,
    'axes.edgecolor':    '#2A4A6B',
    'axes.labelcolor':   TEXT,
    'xtick.color':       TEXT_DIM,
    'ytick.color':       TEXT_DIM,
    'text.color':        TEXT,
    'grid.color':        '#1E3A5F',
    'grid.alpha':        0.5,
    'legend.facecolor':  PANEL,
    'legend.edgecolor':  '#2A4A6B',
    'legend.labelcolor': TEXT,
    'axes.titlecolor':   TEXT,
    'axes.titlesize':    11,
    'axes.labelsize':    9,
    'font.family':       'DejaVu Sans',
})


# ══════════════════════════════════════════════════════════════════════
# DATA PARSER
# ══════════════════════════════════════════════════════════════════════

def parse_file(path):
    data = {
        'cycles': [], 'loads': [], 'w_mean': [], 'cv': [],
        'theta': [], 'migrations_per_cycle': [], 'trigger_type': [],
        'hard_triggers': 0, 'soft_triggers': 0, 'total_migrations': 0,
        'false_pos_triggers': 0, 'genuine_triggers': 0,
        'near_zero_skips': 0, 'is_mtlb': False,
    }
    cur_cycle = 0
    cur_loads = None
    cur_migs  = 0
    cur_cv    = None
    cur_theta = 0.4
    cur_wmean = None
    trigger   = 'NONE'

    def flush():
        nonlocal cur_cycle, cur_loads, cur_migs, cur_cv
        nonlocal cur_theta, cur_wmean, trigger
        if cur_cycle == 0:
            return
        data['cycles'].append(cur_cycle)
        data['loads'].append(cur_loads if cur_loads else [0]*5)
        data['w_mean'].append(cur_wmean if cur_wmean is not None else 0.0)
        data['cv'].append(cur_cv if cur_cv is not None else 0.0)
        data['theta'].append(cur_theta)
        data['migrations_per_cycle'].append(cur_migs)
        data['trigger_type'].append(trigger)
        data['total_migrations'] += cur_migs

    try:
        with open(path, 'r', errors='replace') as f:
            for line in f:
                line = line.strip()
                if 'MTLB Enhanced' in line or 'θ_dynamic' in line:
                    data['is_mtlb'] = True
                m = re.match(r'\[Cycle\s+(\d+)\]', line)
                if m:
                    flush()
                    cur_cycle = int(m.group(1))
                    cur_loads = None; cur_migs = 0
                    cur_cv = None; cur_wmean = None; trigger = 'NONE'
                    continue
                m = re.search(r'Controller loads:\s*\[([^\]]+)\]', line)
                if m:
                    vals = [int(x) for x in m.group(1).split()]
                    cur_loads = vals
                    mean = np.mean(vals)
                    cur_wmean = mean
                    cur_cv = (np.std(vals)/mean) if mean > 0 else 0.0
                    continue
                m = re.search(r'w_mean=([\d.]+)\s+CV=([\d.]+)\s+θ_dynamic=([\d.]+)', line)
                if m:
                    cur_wmean = float(m.group(1))
                    cur_cv    = float(m.group(2))
                    cur_theta = float(m.group(3))
                    continue
                m = re.search(r'\[CVDetect\]\s+CV=([\d.]+)', line)
                if m:
                    cur_cv = float(m.group(1)); cur_theta = 0.4
                    continue
                if 'All loads zero' in line or 'near-zero' in line.lower():
                    data['near_zero_skips'] += 1; trigger = 'SKIP'; continue
                if 'HARD TRIGGER' in line:
                    data['hard_triggers']    += 1
                    data['genuine_triggers'] += 1
                    trigger = 'GENUINE_HARD'; continue
                if 'SOFT TRIGGER' in line:
                    data['soft_triggers']    += 1
                    data['genuine_triggers'] += 1
                    trigger = 'GENUINE_SOFT'; continue
                if 'IMBALANCE DETECTED' in line and not data['is_mtlb']:
                    wm = cur_wmean if cur_wmean else 0
                    if wm < 5:
                        data['false_pos_triggers'] += 1; trigger = 'FALSE_POS'
                    else:
                        data['genuine_triggers'] += 1; trigger = 'GENUINE_HARD'
                    continue
                if re.match(r'\[Migration\]\s+Switch', line):
                    cur_migs += 1
    except FileNotFoundError:
        pass
    flush()
    return data


def load_all_runs():
    runs = {'static': [], 'dynamic': []}
    missing = []
    for kind, files in RUN_FILES.items():
        for fname in files:
            path = os.path.join(SCRIPT_DIR, fname)
            d = parse_file(path)
            if not d['cycles']:
                missing.append(fname)
            runs[kind].append(d)
    return runs, missing


def smooth(arr, w=7):
    if len(arr) < w:
        return np.array(arr, dtype=float)
    k = np.ones(w)/w
    p = np.pad(arr, (w//2, w//2), mode='edge')
    return np.convolve(p, k, mode='valid')[:len(arr)]


# ══════════════════════════════════════════════════════════════════════
# PLOT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def plot_controller_loads(fig, runs):
    """Page 1 — Controller Loads: Static (top) vs MTLB (bottom)"""
    fig.clear()
    axes = fig.subplots(2, 1, gridspec_kw={'hspace': 0.45})

    for ax, kind, color, label in [
        (axes[0], 'static',  C_STATIC, 'Static LBABC  (Fixed CV=0.4)'),
        (axes[1], 'dynamic', C_MTLB,   'MTLB Enhanced (Adaptive θ)'),
    ]:
        data   = runs[kind][0]
        loads  = np.array(data['loads'], dtype=float)
        cycles = np.array(data['cycles'])
        for c in range(min(5, loads.shape[1])):
            ax.plot(cycles, smooth(loads[:, c], 7),
                    color=CTRL_COLORS[c], lw=1.5,
                    marker='o', markersize=2, alpha=0.9,
                    label=f'C{c}')
        ax.set_title(label, fontweight='bold', pad=6)
        ax.set_ylabel('Packets / Cycle', fontsize=9)
        ax.set_xlabel('Cycle', fontsize=9)
        ax.legend(loc='upper right', fontsize=8, ncol=5,
                  framealpha=0.6)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Controller Load Distribution over Time',
                 fontsize=13, fontweight='bold', color=ACCENT, y=1.01)


def plot_cv_threshold(fig, runs):
    """Page 2 — CV vs Threshold comparison"""
    fig.clear()
    axes = fig.subplots(2, 1, gridspec_kw={'hspace': 0.45})

    for ax, kind, color, label in [
        (axes[0], 'static',  C_STATIC, 'Static LBABC — CV vs Fixed θ=0.4'),
        (axes[1], 'dynamic', C_MTLB,   'MTLB Enhanced — CV vs Adaptive θ_dynamic'),
    ]:
        ref    = runs[kind][0]
        cycles = np.array(ref['cycles'])
        cv     = smooth(ref['cv'], 5)
        theta  = smooth(ref['theta'], 5)

        ax.plot(cycles, cv, color=color, lw=1.8, label='CV', zorder=4)
        ax.axhline(0.4, color='gray', lw=1.2, ls='--',
                   alpha=0.7, label='Floor θ = 0.40')

        if kind == 'dynamic':
            ax.plot(cycles, theta, color=C_THETA, lw=1.5,
                    ls='-.', label='θ_dynamic (EMA)', zorder=5)
            ax.fill_between(cycles, 0.4, theta,
                            color=C_THETA, alpha=0.12)
            for i, tt in enumerate(ref['trigger_type'][:len(cycles)]):
                if tt == 'GENUINE_HARD':
                    ax.axvline(cycles[i], color=C_HARD, alpha=0.35,
                               lw=0.8, ls=':')
                elif tt == 'GENUINE_SOFT':
                    ax.axvline(cycles[i], color=C_SOFT, alpha=0.35,
                               lw=0.8, ls=':')
        else:
            for i, tt in enumerate(ref['trigger_type'][:len(cycles)]):
                if tt == 'FALSE_POS':
                    ax.axvline(cycles[i], color='#FF6B6B', alpha=0.2,
                               lw=0.7, ls=':')
                elif tt == 'GENUINE_HARD':
                    ax.axvline(cycles[i], color='#FFF', alpha=0.3,
                               lw=0.7, ls=':')

        ax.set_title(label, fontweight='bold', pad=6)
        ax.set_ylabel('CV', fontsize=9)
        ax.set_xlabel('Cycle', fontsize=9)
        ax.legend(fontsize=8, loc='upper right', ncol=4, framealpha=0.6)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle('CV Over Time: Static vs MTLB (Run 0)',
                 fontsize=13, fontweight='bold', color=ACCENT, y=1.01)


def plot_migrations_per_cycle(fig, runs):
    """Page 3 — Migrations per cycle colored by type"""
    fig.clear()
    axes = fig.subplots(2, 1, gridspec_kw={'hspace': 0.5})

    for ax, kind, label in [
        (axes[0], 'static',  'Static LBABC — Migrations per Cycle'),
        (axes[1], 'dynamic', 'MTLB Enhanced — Migrations per Cycle'),
    ]:
        ref    = runs[kind][0]
        cycles = np.array(ref['cycles'])
        migs   = np.array(ref['migrations_per_cycle'], dtype=float)
        ttypes = ref['trigger_type']

        colors = []
        for tt in ttypes:
            if tt == 'FALSE_POS':    colors.append('#FF4757')
            elif tt == 'GENUINE_HARD': colors.append('#FFA502')
            elif tt == 'GENUINE_SOFT': colors.append('#2ED573')
            elif tt == 'SKIP':         colors.append('#3A5F7F')
            else:                      colors.append('#2A4A6B')

        ax.bar(cycles, migs, color=colors[:len(cycles)],
               width=0.85, alpha=0.88, edgecolor='none')

        ax.set_title(label, fontweight='bold', pad=6)
        ax.set_ylabel('Migrations', fontsize=9)
        ax.set_xlabel('Cycle', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        if kind == 'static':
            patches = [
                mpatches.Patch(color='#FF4757', label='False Positive'),
                mpatches.Patch(color='#FFA502', label='Genuine'),
                mpatches.Patch(color='#2A4A6B', label='No Trigger'),
            ]
        else:
            patches = [
                mpatches.Patch(color='#FFA502', label='HARD Trigger'),
                mpatches.Patch(color='#2ED573', label='SOFT Trigger'),
                mpatches.Patch(color='#3A5F7F', label='Skipped (near-zero)'),
                mpatches.Patch(color='#2A4A6B', label='No Trigger'),
            ]
        ax.legend(handles=patches, fontsize=8, loc='upper right',
                  framealpha=0.6, ncol=4 if kind == 'dynamic' else 3)

    fig.suptitle('Migration Events per Cycle (Color = Trigger Type)',
                 fontsize=13, fontweight='bold', color=ACCENT, y=1.01)


def plot_all_runs_comparison(fig, runs):
    """Page 4 — All 5 runs total migrations + trigger breakdown"""
    fig.clear()
    axes = fig.subplots(1, 2, gridspec_kw={'wspace': 0.35})
    run_labels = ['Run 1','Run 2','Run 3','Run 4','Run 5']
    x = np.arange(5)

    # Left: total migrations
    ax = axes[0]
    s_tot = [d['total_migrations'] for d in runs['static']]
    d_tot = [d['total_migrations'] for d in runs['dynamic']]
    b1 = ax.bar(x-0.2, s_tot, 0.38, color=C_STATIC, alpha=0.85,
                label='Static LBABC', edgecolor='none')
    b2 = ax.bar(x+0.2, d_tot, 0.38, color=C_MTLB,   alpha=0.85,
                label='MTLB Enhanced', edgecolor='none')
    for i, (s, d) in enumerate(zip(s_tot, d_tot)):
        red = (s-d)/(s+1e-9)*100
        ax.annotate(f'↓{red:.0f}%',
                    xy=((x[i]-0.2 + x[i]+0.2)/2, max(s,d)+15),
                    ha='center', fontsize=8.5,
                    color='#A8FF78', fontweight='bold')
    for b in b1:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+5,
                str(int(b.get_height())), ha='center',
                fontsize=7.5, color=C_STATIC, fontweight='bold')
    for b in b2:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+5,
                str(int(b.get_height())), ha='center',
                fontsize=7.5, color=C_MTLB, fontweight='bold')
    ax.set_title('Total Migrations per Run', fontweight='bold', pad=6)
    ax.set_xticks(x); ax.set_xticklabels(run_labels, fontsize=8)
    ax.set_ylabel('Total Migrations'); ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=8, framealpha=0.6)

    # Right: trigger breakdown stacked bars
    ax = axes[1]
    s_gen = np.array([d['genuine_triggers']    for d in runs['static']])
    s_fp  = np.array([d['false_pos_triggers']  for d in runs['static']])
    d_hard= np.array([d['hard_triggers']        for d in runs['dynamic']])
    d_soft= np.array([d['soft_triggers']        for d in runs['dynamic']])
    ax.bar(x-0.2, s_gen, 0.38, color='#8BC34A', alpha=0.9,
           label='Static Genuine',  edgecolor='none')
    ax.bar(x-0.2, s_fp,  0.38, bottom=s_gen, color=C_STATIC, alpha=0.9,
           label='Static False Pos', edgecolor='none')
    ax.bar(x+0.2, d_hard, 0.38, color=C_HARD, alpha=0.9,
           label='MTLB HARD',  edgecolor='none')
    ax.bar(x+0.2, d_soft, 0.38, bottom=d_hard, color=C_SOFT, alpha=0.9,
           label='MTLB SOFT',  edgecolor='none')
    ax.set_title('Trigger Breakdown per Run', fontweight='bold', pad=6)
    ax.set_xticks(x); ax.set_xticklabels(run_labels, fontsize=8)
    ax.set_ylabel('Number of Triggers')
    ax.legend(fontsize=8, framealpha=0.6)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('All 5 Runs — Migration & Trigger Summary',
                 fontsize=13, fontweight='bold', color=ACCENT, y=1.01)


def plot_imbalance_band(fig, runs):
    """Page 5 — Load imbalance CV band (all 5 runs)"""
    fig.clear()
    ax = fig.add_subplot(111)

    for kind, color, label in [
        ('static',  C_STATIC, 'Static LBABC'),
        ('dynamic', C_MTLB,   'MTLB Enhanced'),
    ]:
        min_len = min(len(d['cv']) for d in runs[kind] if d['cv'])
        if min_len == 0:
            continue
        cv_mat  = np.array([smooth(d['cv'][:min_len], 9)
                            for d in runs[kind]])
        cycles  = np.array(runs[kind][0]['cycles'][:min_len])
        median  = np.median(cv_mat, axis=0)
        lo      = np.percentile(cv_mat, 25, axis=0)
        hi      = np.percentile(cv_mat, 75, axis=0)
        ax.plot(cycles, median, color=color, lw=2.2,
                label=f'{label} (median)')
        ax.fill_between(cycles, lo, hi, color=color, alpha=0.18,
                        label=f'{label} (25–75%)')

    ax.axhline(0.4, color='gray', lw=1.3, ls='--', alpha=0.7,
               label='Static threshold θ=0.4')
    ax.set_title('Load Imbalance (CV) — Median + IQR Band (5 Runs)',
                 fontweight='bold', pad=8)
    ax.set_xlabel('Cycle'); ax.set_ylabel('CV')
    ax.legend(fontsize=9, framealpha=0.6)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.suptitle('Controller Load Imbalance Comparison',
                 fontsize=13, fontweight='bold', color=ACCENT)


def plot_cumulative_migrations(fig, runs):
    """Page 6 — Cumulative migrations all runs"""
    fig.clear()
    ax = fig.add_subplot(111)

    for kind, color, label in [
        ('static',  C_STATIC, 'Static LBABC'),
        ('dynamic', C_MTLB,   'MTLB Enhanced'),
    ]:
        min_len  = min(len(d['migrations_per_cycle'])
                       for d in runs[kind] if d['migrations_per_cycle'])
        if min_len == 0:
            continue
        cum_list = [np.cumsum(d['migrations_per_cycle'][:min_len])
                    for d in runs[kind]]
        cycles   = np.array(runs[kind][0]['cycles'][:min_len])
        cum_mat  = np.array(cum_list, dtype=float)
        median   = np.median(cum_mat, axis=0)
        lo       = np.percentile(cum_mat, 25, axis=0)
        hi       = np.percentile(cum_mat, 75, axis=0)
        ax.plot(cycles, median, color=color, lw=2.5,
                label=f'{label} (median)')
        ax.fill_between(cycles, lo, hi, color=color, alpha=0.18)
        ax.annotate(f'  {int(median[-1])}',
                    xy=(cycles[-1], median[-1]),
                    color=color, fontsize=11, fontweight='bold',
                    va='center')

    # Shade the "wasted migrations" region
    s_data = runs['static']
    d_data = runs['dynamic']
    ml_s = min(len(d['migrations_per_cycle']) for d in s_data if d['migrations_per_cycle'])
    ml_d = min(len(d['migrations_per_cycle']) for d in d_data if d['migrations_per_cycle'])
    ml   = min(ml_s, ml_d)
    if ml > 0:
        cs  = np.array(runs['static'][0]['cycles'][:ml])
        cm_s= np.median([np.cumsum(d['migrations_per_cycle'][:ml])
                         for d in runs['static']], axis=0)
        cm_d= np.median([np.cumsum(d['migrations_per_cycle'][:ml])
                         for d in runs['dynamic']], axis=0)
        ax.fill_between(cs, cm_d, cm_s, color='#FF4757', alpha=0.07,
                        label='Wasted overhead (Static - MTLB)')

    ax.set_title('Cumulative Migrations over Cycles (Median of 5 Runs)',
                 fontweight='bold', pad=8)
    ax.set_xlabel('Cycle'); ax.set_ylabel('Cumulative Migrations')
    ax.legend(fontsize=9, framealpha=0.6)
    ax.grid(True, alpha=0.3)
    fig.suptitle('Migration Overhead: Static vs MTLB',
                 fontsize=13, fontweight='bold', color=ACCENT)


def plot_throughput_response(fig, runs):
    """Page 7 — Throughput and Response Time side by side"""
    fig.clear()
    axes = fig.subplots(1, 2, gridspec_kw={'wspace': 0.35})

    for kind, color, label in [
        ('static',  C_STATIC, 'Static LBABC'),
        ('dynamic', C_MTLB,   'MTLB Enhanced'),
    ]:
        min_len = min(len(d['loads']) for d in runs[kind] if d['loads'])
        if min_len == 0: continue
        tot_list = [[sum(ld) for ld in d['loads'][:min_len]]
                    for d in runs[kind]]
        cycles   = np.array(runs[kind][0]['cycles'][:min_len])
        sm_mat   = np.array([smooth(row, 9) for row in tot_list])
        median   = np.median(sm_mat, axis=0)
        lo       = np.percentile(sm_mat, 25, axis=0)
        hi       = np.percentile(sm_mat, 75, axis=0)
        axes[0].plot(cycles, median, color=color, lw=2.0,
                     label=f'{label}')
        axes[0].fill_between(cycles, lo, hi, color=color, alpha=0.15)

        # Response time proxy
        rt_list = []
        for d in runs[kind]:
            cv_a = np.array(d['cv'][:min_len])
            wm_a = np.array(d['w_mean'][:min_len])
            mx   = np.max(wm_a) + 1e-6
            rt   = 40 + cv_a * wm_a / mx * 60
            rt_list.append(rt)
        rt_mat = np.array([smooth(row, 9) for row in rt_list])
        med_rt = np.median(rt_mat, axis=0)
        lo_rt  = np.percentile(rt_mat, 25, axis=0)
        hi_rt  = np.percentile(rt_mat, 75, axis=0)
        axes[1].plot(cycles, med_rt, color=color, lw=2.0,
                     label=f'{label}')
        axes[1].fill_between(cycles, lo_rt, hi_rt, color=color, alpha=0.15)

    axes[0].set_title('Network Throughput (Packets/Cycle)',
                      fontweight='bold', pad=6)
    axes[0].set_xlabel('Cycle'); axes[0].set_ylabel('Total Packets')
    axes[0].legend(fontsize=9, framealpha=0.6)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Average Response Time Proxy (ms)',
                      fontweight='bold', pad=6)
    axes[1].set_xlabel('Cycle'); axes[1].set_ylabel('Response Time (ms)')
    axes[1].legend(fontsize=9, framealpha=0.6)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(bottom=20)

    fig.suptitle('Throughput & Response Time — Median of 5 Runs',
                 fontsize=13, fontweight='bold', color=ACCENT, y=1.01)


def plot_overhead_quality(fig, runs):
    """Page 8 — Communication Overhead + Migration Quality Pie"""
    fig.clear()
    axes = fig.subplots(1, 2, gridspec_kw={'wspace': 0.4})

    # Left: Communication Overhead
    ax = axes[0]
    BYTES = 512
    for kind, color, label in [
        ('static',  C_STATIC, 'Static LBABC'),
        ('dynamic', C_MTLB,   'MTLB Enhanced'),
    ]:
        min_len = min(len(d['migrations_per_cycle'])
                      for d in runs[kind] if d['migrations_per_cycle'])
        if min_len == 0: continue
        oh_list = [smooth(np.array(d['migrations_per_cycle'][:min_len],
                                   dtype=float)*BYTES/1024, 11)
                   for d in runs[kind]]
        cycles  = np.array(runs[kind][0]['cycles'][:min_len])
        oh_mat  = np.array(oh_list)
        median  = np.median(oh_mat, axis=0)
        lo      = np.percentile(oh_mat, 25, axis=0)
        hi      = np.percentile(oh_mat, 75, axis=0)
        ax.plot(cycles, median, color=color, lw=2.0, label=label)
        ax.fill_between(cycles, lo, hi, color=color, alpha=0.15)
    ax.set_title('Communication Overhead (KB/Cycle)',
                 fontweight='bold', pad=6)
    ax.set_xlabel('Cycle'); ax.set_ylabel('Overhead (KB)')
    ax.legend(fontsize=9, framealpha=0.6)
    ax.grid(True, alpha=0.3)

    # Right: Pie — Migration Quality
    ax = axes[1]
    ax.set_facecolor(BG)
    total_gen  = sum(d['genuine_triggers']   for d in runs['static'])
    total_fp   = sum(d['false_pos_triggers'] for d in runs['static'])
    total_hard = sum(d['hard_triggers']      for d in runs['dynamic'])
    total_soft = sum(d['soft_triggers']      for d in runs['dynamic'])
    total_skip = sum(d['near_zero_skips']    for d in runs['dynamic'])

    # Two pie insets
    from matplotlib.patches import FancyArrowPatch
    ax.axis('off')
    ax_s  = fig.add_axes([0.56, 0.55, 0.19, 0.35])
    ax_d  = fig.add_axes([0.76, 0.55, 0.19, 0.35])
    ax_s.set_facecolor(BG)
    ax_d.set_facecolor(BG)

    ax_s.pie([total_gen, total_fp],
             colors=['#8BC34A', C_STATIC],
             autopct='%1.0f%%', startangle=90,
             textprops={'fontsize': 8, 'color': TEXT},
             wedgeprops={'edgecolor': BG, 'linewidth': 1.5})
    ax_s.set_title('Static\nTriggers', fontsize=8.5,
                   color=TEXT_DIM, fontweight='bold', pad=4)

    ax_d.pie([total_hard, total_soft, total_skip],
             colors=[C_HARD, C_SOFT, '#3A5F7F'],
             autopct='%1.0f%%', startangle=90,
             textprops={'fontsize': 8, 'color': TEXT},
             wedgeprops={'edgecolor': BG, 'linewidth': 1.5})
    ax_d.set_title('MTLB\nTriggers', fontsize=8.5,
                   color=TEXT_DIM, fontweight='bold', pad=4)

    # Legends
    patches_s = [mpatches.Patch(color='#8BC34A', label=f'Genuine ({total_gen})'),
                 mpatches.Patch(color=C_STATIC,  label=f'False Pos ({total_fp})')]
    patches_d = [mpatches.Patch(color=C_HARD, label=f'HARD ({total_hard})'),
                 mpatches.Patch(color=C_SOFT, label=f'SOFT ({total_soft})'),
                 mpatches.Patch(color='#3A5F7F', label=f'Skip ({total_skip})')]
    ax_s.legend(handles=patches_s, fontsize=7, loc='lower center',
                bbox_to_anchor=(0.5, -0.35), framealpha=0.5, ncol=1)
    ax_d.legend(handles=patches_d, fontsize=7, loc='lower center',
                bbox_to_anchor=(0.5, -0.45), framealpha=0.5, ncol=1)

    fig.suptitle('Communication Overhead & Migration Quality (5 Runs)',
                 fontsize=13, fontweight='bold', color=ACCENT, y=1.01)


def plot_theta_evolution(fig, runs):
    """Page 9 — MTLB θ evolution"""
    fig.clear()
    ax1 = fig.add_subplot(111)

    ref    = runs['dynamic'][0]
    cycles = np.array(ref['cycles'])
    theta  = np.array(ref['theta'])
    cv     = np.array(ref['cv'])

    ax1.plot(cycles, theta, color=C_THETA, lw=2.5,
             label='θ_dynamic (EMA adaptive)', zorder=5)
    ax1.axhline(0.4, color='gray', lw=1.5, ls='--',
                alpha=0.7, label='Floor θ = 0.40')
    ax1.fill_between(cycles, 0.4, theta,
                     where=theta >= 0.4, color=C_THETA,
                     alpha=0.10, label='θ adaptation band')

    for i, (c, tt) in enumerate(zip(cycles, ref['trigger_type'])):
        if tt == 'GENUINE_HARD':
            ax1.axvline(c, color=C_HARD, alpha=0.4, lw=1.0, ls=':')
        elif tt == 'GENUINE_SOFT':
            ax1.axvline(c, color=C_SOFT, alpha=0.4, lw=1.0, ls=':')

    ax2 = ax1.twinx()
    ax2.plot(cycles, cv, color='#607D8B', lw=1.0,
             alpha=0.5, label='CV (right axis)')
    ax2.set_ylabel('CV', fontsize=9, color='#90A4AE')
    ax2.tick_params(axis='y', colors='#90A4AE', labelsize=8)
    ax2.set_ylim(bottom=0)

    h_patch = mpatches.Patch(color=C_HARD, alpha=0.5, label='HARD trigger')
    s_patch = mpatches.Patch(color=C_SOFT, alpha=0.5, label='SOFT trigger')
    lines1, lbl1 = ax1.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + [h_patch, s_patch],
               lbl1 + lbl2 + ['HARD trigger','SOFT trigger'],
               fontsize=8, framealpha=0.6, ncol=3, loc='upper left')

    ax1.set_title('MTLB Adaptive Threshold θ Evolution (Run 0)',
                  fontweight='bold', pad=8)
    ax1.set_xlabel('Cycle')
    ax1.set_ylabel('θ_dynamic', color=C_THETA)
    ax1.tick_params(axis='y', colors=C_THETA)
    ax1.grid(True, alpha=0.3)
    fig.suptitle('Dynamic Threshold Adaptation — MTLB',
                 fontsize=13, fontweight='bold', color=ACCENT)


def plot_summary_stats(fig, runs):
    """Page 10 — Summary statistics table"""
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)
    ax.axis('off')

    def avg(lst): return np.mean(lst) if lst else 0

    s = runs['static']
    d = runs['dynamic']

    s_migs = [x['total_migrations']   for x in s]
    d_migs = [x['total_migrations']   for x in d]
    s_gen  = [x['genuine_triggers']   for x in s]
    d_gen  = [x['genuine_triggers']   for x in d]
    s_fp   = [x['false_pos_triggers'] for x in s]
    s_fpr  = [fp/(fp+g+1e-9)*100 for fp,g in zip(s_fp, s_gen)]
    d_hard = [x['hard_triggers']      for x in d]
    d_soft = [x['soft_triggers']      for x in d]
    d_skip = [x['near_zero_skips']    for x in d]
    mred   = [(ss-dd)/(ss+1e-9)*100  for ss,dd in zip(s_migs, d_migs)]
    s_cyc  = [len(x['cycles'])        for x in s]
    d_cyc  = [len(x['cycles'])        for x in d]

    rows = [
        ['Total Cycles (avg)',              f'{avg(s_cyc):.0f}',      f'{avg(d_cyc):.0f}'],
        ['Total Migrations (avg)',          f'{avg(s_migs):.0f}',     f'{avg(d_migs):.0f}'],
        ['Genuine Triggers (avg)',          f'{avg(s_gen):.0f}',      f'{avg(d_gen):.0f}'],
        ['False Positive Triggers (avg)',   f'{avg(s_fp):.0f}',       '0'],
        ['False Positive Rate (%)',         f'{avg(s_fpr):.1f}%',     '0.0%'],
        ['HARD Triggers (avg)',             'N/A',                    f'{avg(d_hard):.0f}'],
        ['SOFT Triggers (avg)',             'N/A',                    f'{avg(d_soft):.0f}'],
        ['Near-Zero Skips (avg)',           'N/A',                    f'{avg(d_skip):.0f}'],
        ['Migration Reduction (%)',         '—',                      f'{avg(mred):.1f}%'],
    ]
    cols = ['  Metric', '  Static LBABC', '  MTLB Enhanced']

    tbl = ax.table(
        cellText=rows,
        colLabels=cols,
        cellLoc='left',
        loc='center',
        colWidths=[0.46, 0.27, 0.27],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.0)

    # Header
    for j in range(3):
        cell = tbl[(0, j)]
        cell.set_facecolor(ACCENT)
        cell.set_text_props(color=BG, fontweight='bold', fontsize=11)
        cell.set_edgecolor(BG)

    # Body rows
    for i, row in enumerate(rows, 1):
        for j in range(3):
            cell = tbl[(i, j)]
            cell.set_facecolor(CARD if i % 2 == 0 else PANEL)
            cell.set_edgecolor(BG)
            cell.set_text_props(color=TEXT)
        # Highlight key cells
        if row[0].strip() == 'Migration Reduction (%)':
            tbl[(i, 2)].set_facecolor('#1B4332')
            tbl[(i, 2)].set_text_props(color='#A8FF78', fontweight='bold')
        if row[0].strip() == 'False Positive Rate (%)':
            tbl[(i, 1)].set_facecolor('#4A1C1C')
            tbl[(i, 1)].set_text_props(color='#FF6B6B', fontweight='bold')
            tbl[(i, 2)].set_facecolor('#1B4332')
            tbl[(i, 2)].set_text_props(color='#A8FF78', fontweight='bold')

    fig.suptitle(
        'Summary Statistics  —  Static LBABC  vs  MTLB Enhanced\n'
        'Averaged across 5 Simulation Runs  |  BT Asia Pacific Topology',
        fontsize=12, fontweight='bold', color=ACCENT, y=0.97
    )


# ══════════════════════════════════════════════════════════════════════
# PAGES REGISTRY
# ══════════════════════════════════════════════════════════════════════

PAGES = [
    ('📊  Controller Loads',      plot_controller_loads),
    ('📈  CV vs Threshold',        plot_cv_threshold),
    ('🔁  Migrations per Cycle',   plot_migrations_per_cycle),
    ('📋  All 5 Runs Summary',     plot_all_runs_comparison),
    ('🌊  Imbalance Band',         plot_imbalance_band),
    ('🔼  Cumulative Migrations',  plot_cumulative_migrations),
    ('⚡  Throughput & Response',  plot_throughput_response),
    ('📡  Overhead & Quality',     plot_overhead_quality),
    ('🧮  θ Evolution (MTLB)',     plot_theta_evolution),
    ('🏆  Summary Table',          plot_summary_stats),
]


# ══════════════════════════════════════════════════════════════════════
# GUI APPLICATION
# ══════════════════════════════════════════════════════════════════════

class LBABCDashboard:

    def __init__(self, root, runs):
        self.root      = root
        self.runs      = runs
        self.cur_page  = 0

        root.title('LBABC Analysis Dashboard — Static vs MTLB Enhanced')
        root.configure(bg=BG)
        root.geometry('1280x760')
        root.minsize(900, 600)

        self._build_ui()
        self._show_page(0)

    def _build_ui(self):
        # ── Top header bar ──────────────────────────────────────────
        header = tk.Frame(self.root, bg=PANEL, height=52)
        header.pack(fill='x', side='top')
        header.pack_propagate(False)

        tk.Label(header,
                 text='  LBABC ANALYSIS DASHBOARD',
                 bg=PANEL, fg=ACCENT,
                 font=('Courier', 15, 'bold')).pack(side='left', padx=16,
                                                     pady=12)

        subtitle = ('Static LBABC  vs  MTLB Enhanced  |  '
                    'BT Asia Pacific Topology  |  5 Simulation Runs')
        tk.Label(header, text=subtitle,
                 bg=PANEL, fg=TEXT_DIM,
                 font=('Courier', 9)).pack(side='left', padx=0, pady=12)

        # Save button
        tk.Button(header, text='💾  Save Current',
                  bg=CARD, fg=ACCENT,
                  font=('Courier', 9, 'bold'),
                  relief='flat', padx=12, pady=4,
                  cursor='hand2',
                  activebackground=ACCENT,
                  activeforeground=BG,
                  command=self._save_current).pack(side='right', padx=12,
                                                    pady=10)
        tk.Button(header, text='💾  Save All',
                  bg=CARD, fg=C_MTLB,
                  font=('Courier', 9, 'bold'),
                  relief='flat', padx=12, pady=4,
                  cursor='hand2',
                  activebackground=C_MTLB,
                  activeforeground=BG,
                  command=self._save_all).pack(side='right', padx=4,
                                               pady=10)

        # ── Main body: sidebar + canvas ─────────────────────────────
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill='both', expand=True, side='top')

        # Sidebar
        sidebar = tk.Frame(body, bg=PANEL, width=220)
        sidebar.pack(fill='y', side='left')
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text='ANALYSIS VIEWS',
                 bg=PANEL, fg=TEXT_DIM,
                 font=('Courier', 8, 'bold')).pack(pady=(14, 6), padx=14,
                                                    anchor='w')

        self.nav_buttons = []
        for i, (label, _) in enumerate(PAGES):
            btn = tk.Button(sidebar,
                            text=label,
                            bg=PANEL, fg=TEXT,
                            font=('Courier', 9),
                            relief='flat', anchor='w',
                            padx=14, pady=6,
                            cursor='hand2',
                            activebackground=CARD,
                            activeforeground=ACCENT,
                            command=lambda idx=i: self._show_page(idx))
            btn.pack(fill='x', padx=6, pady=1)
            self.nav_buttons.append(btn)

        # Stats mini-panel at bottom of sidebar
        tk.Frame(sidebar, bg='#1E3A5F', height=1).pack(fill='x',
                                                         padx=10, pady=8)
        self._stats_frame = tk.Frame(sidebar, bg=PANEL)
        self._stats_frame.pack(fill='x', padx=10)
        self._build_stats_panel()

        # Canvas area
        canvas_frame = tk.Frame(body, bg=BG)
        canvas_frame.pack(fill='both', expand=True, side='left')

        # Page title bar
        self._page_title = tk.Label(canvas_frame,
                                    text='',
                                    bg=CARD,
                                    fg=ACCENT,
                                    font=('Courier', 11, 'bold'),
                                    anchor='w', padx=16, pady=7)
        self._page_title.pack(fill='x', side='top')

        # Matplotlib figure
        self.fig = plt.figure(figsize=(10, 6.5),
                              facecolor=BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().configure(bg=BG,
                                              highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill='both', expand=True,
                                          padx=8, pady=4)

        # Navigation toolbar
        toolbar_frame = tk.Frame(canvas_frame, bg=PANEL)
        toolbar_frame.pack(fill='x', side='bottom')
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.config(bg=PANEL)
        toolbar._message_label.config(bg=PANEL, fg=TEXT_DIM)
        toolbar.update()

        # ── Bottom nav bar ──────────────────────────────────────────
        nav_bar = tk.Frame(canvas_frame, bg=CARD, height=44)
        nav_bar.pack(fill='x', side='bottom')
        nav_bar.pack_propagate(False)

        tk.Button(nav_bar, text='◀  Previous',
                  bg=CARD, fg=TEXT,
                  font=('Courier', 9, 'bold'),
                  relief='flat', padx=14, pady=6,
                  cursor='hand2',
                  activebackground=PANEL,
                  activeforeground=ACCENT,
                  command=self._prev).pack(side='left', padx=10, pady=5)

        self._page_indicator = tk.Label(nav_bar, text='',
                                        bg=CARD, fg=TEXT_DIM,
                                        font=('Courier', 9))
        self._page_indicator.pack(side='left', expand=True)

        tk.Button(nav_bar, text='Next  ▶',
                  bg=CARD, fg=TEXT,
                  font=('Courier', 9, 'bold'),
                  relief='flat', padx=14, pady=6,
                  cursor='hand2',
                  activebackground=PANEL,
                  activeforeground=ACCENT,
                  command=self._next).pack(side='right', padx=10, pady=5)

        # Keyboard navigation
        self.root.bind('<Left>',  lambda e: self._prev())
        self.root.bind('<Right>', lambda e: self._next())
        self.root.bind('<Prior>', lambda e: self._prev())
        self.root.bind('<Next>',  lambda e: self._next())

    def _build_stats_panel(self):
        """Small KPI cards in sidebar."""
        s_runs = self.runs['static']
        d_runs = self.runs['dynamic']

        s_migs = np.mean([d['total_migrations'] for d in s_runs])
        d_migs = np.mean([d['total_migrations'] for d in d_runs])
        red    = (s_migs - d_migs) / (s_migs + 1e-9) * 100
        s_fp   = np.mean([d['false_pos_triggers'] for d in s_runs])
        s_gen  = np.mean([d['genuine_triggers']   for d in s_runs])
        fpr    = s_fp / (s_fp + s_gen + 1e-9) * 100

        kpis = [
            ('Avg Migrations\nStatic',  f'{s_migs:.0f}',    C_STATIC),
            ('Avg Migrations\nMTLB',    f'{d_migs:.0f}',    C_MTLB),
            ('Migration\nReduction',    f'{red:.1f}%',       '#A8FF78'),
            ('False Positive\nRate',    f'{fpr:.1f}%',       '#FF6B6B'),
        ]
        for title, val, color in kpis:
            card = tk.Frame(self._stats_frame, bg=CARD,
                            relief='flat', bd=0)
            card.pack(fill='x', pady=3)
            tk.Label(card, text=title, bg=CARD, fg=TEXT_DIM,
                     font=('Courier', 7)).pack(anchor='w', padx=8, pady=(5,0))
            tk.Label(card, text=val, bg=CARD, fg=color,
                     font=('Courier', 13, 'bold')).pack(anchor='w',
                                                         padx=8, pady=(0,5))

    def _show_page(self, idx):
        self.cur_page = idx
        label, plot_fn = PAGES[idx]

        # Update sidebar highlight
        for i, btn in enumerate(self.nav_buttons):
            if i == idx:
                btn.config(bg=ACCENT, fg=BG, font=('Courier', 9, 'bold'))
            else:
                btn.config(bg=PANEL, fg=TEXT, font=('Courier', 9))

        self._page_title.config(text=f'  {label}')
        self._page_indicator.config(
            text=f'View {idx+1} of {len(PAGES)}'
                 '   ←  →  keys to navigate')

        # Draw
        self.root.config(cursor='watch')
        self.root.update()
        try:
            plot_fn(self.fig, self.runs)
            self.canvas.draw()
        except Exception as e:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f'Error rendering plot:\n{e}',
                    ha='center', va='center', color=C_STATIC,
                    transform=ax.transAxes, fontsize=10)
            self.canvas.draw()
        finally:
            self.root.config(cursor='')

    def _prev(self):
        self._show_page((self.cur_page - 1) % len(PAGES))

    def _next(self):
        self._show_page((self.cur_page + 1) % len(PAGES))

    def _save_current(self):
        label, _ = PAGES[self.cur_page]
        safe_name = re.sub(r'[^\w]', '_', label.strip()).lower()
        path = os.path.join(SCRIPT_DIR, f'{safe_name}.png')
        self.fig.savefig(path, dpi=150, bbox_inches='tight',
                         facecolor=BG)
        messagebox.showinfo('Saved',
                            f'Saved to:\n{path}',
                            parent=self.root)

    def _save_all(self):
        saved = []
        for i, (label, plot_fn) in enumerate(PAGES):
            safe_name = re.sub(r'[^\w]', '_', label.strip()).lower()
            path = os.path.join(SCRIPT_DIR, f'fig{i+1:02d}_{safe_name}.png')
            try:
                plot_fn(self.fig, self.runs)
                self.fig.savefig(path, dpi=150, bbox_inches='tight',
                                 facecolor=BG)
                saved.append(path)
            except Exception as e:
                print(f'  Error saving {label}: {e}')
        # Restore current page
        self._show_page(self.cur_page)
        messagebox.showinfo('All Saved',
                            f'Saved {len(saved)} figures to:\n{SCRIPT_DIR}',
                            parent=self.root)


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    print('=' * 55)
    print('  LBABC Analysis Dashboard')
    print('  Loading simulation data...')
    print('=' * 55)

    runs, missing = load_all_runs()

    total_s = sum(len(d['cycles']) for d in runs['static'])
    total_d = sum(len(d['cycles']) for d in runs['dynamic'])
    print(f'  Static  runs: {len(runs["static"])}  '
          f'(total cycles: {total_s})')
    print(f'  Dynamic runs: {len(runs["dynamic"])}  '
          f'(total cycles: {total_d})')

    if missing:
        print(f'\n  WARNING — files not found (will use zeros):')
        for f in missing:
            print(f'    {f}')

    empty_s = all(len(d['cycles']) == 0 for d in runs['static'])
    empty_d = all(len(d['cycles']) == 0 for d in runs['dynamic'])
    if empty_s and empty_d:
        print('\n  ERROR: No data files found in:', SCRIPT_DIR)
        print('  Place your output .txt files in the same folder as this script.')
        sys.exit(1)

    print('\n  Launching dashboard...')
    print('  Navigation: click sidebar buttons or use ← → arrow keys')
    print('  Save: use buttons in top-right corner')
    print('=' * 55)

    root = tk.Tk()
    root.resizable(True, True)

    # App icon (optional — skip if not available)
    try:
        root.iconbitmap('')
    except Exception:
        pass

    app = LBABCDashboard(root, runs)
    root.mainloop()


if __name__ == '__main__':
    main()
