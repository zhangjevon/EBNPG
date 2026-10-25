# Added for the EBNPG manuscript.
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Generate Dolan-More performance profiles from experiment results.
Reads all_summary.csv and generates perf_profiles.pdf with 2x2 subplots.

R2 fix (2026-10-25):
  * The horizontal axis no longer collapses to [1, 1.1].  The old script
    extended tau only where some curve reached rho = 1.0; because no
    algorithm solves all 42 problems (seven algorithms fail on all runs of
    NIC1 and MPG-Implicit on six problems), the axis was capped at
    tau = 1.1 and the figure was unusable.  The axis now spans
    tau in [1, 64] with power-of-two ticks, which covers the largest
    finite performance ratio (60.8, EBNPG-N1 function evaluations) and
    follows the Dolan-More convention of power-of-two tick marks.
  * Runs recorded as 'nan' in the summary (no successful run of that
    algorithm on that problem) are treated as failures, i.e. their
    performance ratio is +infinity, so the curves level off at the
    fraction of problems solved.
  * Subplot titles no longer repeat "(42 problems)".
  * Figure size and font sizes adjusted for reproduction at
    0.94\textwidth in the manuscript (figure is downscaled by ~1.85).
  * Curves are drawn through the exact ratio values instead of a fixed
    logarithmic grid, so the step locations are exact.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
OUTPUT_DIR = SCRIPT_DIR

TAU_MAX = 64.0                 # power-of-two upper end of the tau axis
TICKS = [1, 2, 4, 8, 16, 32, 64]

# Algorithm display names and colors
# Updated for new 12-algorithm configuration: 3 MPG + 2 CNPG + 7 EBNPG
ALG_INFO = {
    'MPG-Explicit':  {'color': '#1f77b4', 'ls': '-',  'marker': 'o'},
    'MPG-Armijo':    {'color': '#ff7f0e', 'ls': '-',  'marker': 's'},
    'MPG-Implicit':  {'color': '#2ca02c', 'ls': '-',  'marker': '^'},
    'CNPG-Max':      {'color': '#d62728', 'ls': ':',  'marker': 'D'},
    'CNPG-Avg':      {'color': '#9467bd', 'ls': ':',  'marker': 'v'},
    'EBNPG-N0':      {'color': '#8c564b', 'ls': '--', 'marker': 'o'},
    'EBNPG-N1':      {'color': '#e377c2', 'ls': '--', 'marker': 's'},
    'EBNPG-N2':      {'color': '#7f7f7f', 'ls': '--', 'marker': '^'},
    'EBNPG-N3':      {'color': '#bcbd22', 'ls': '--', 'marker': 'D'},
    'EBNPG-N5':      {'color': '#17becf', 'ls': '--', 'marker': 'v'},
    'EBNPG-N6':      {'color': '#aec7e8', 'ls': '-.', 'marker': 'o'},
    'EBNPG-N7':      {'color': '#ff9896', 'ls': '-.', 'marker': 's'},
}

METRICS = [
    ('MedIter',  'Iterations'),
    ('MedNgev',  'Function evaluations'),
    ('MedNhev',  'Nonsmooth evaluations'),
    ('MedTime',  'CPU time (s)'),
]


def _to_float(v):
    """Median of the successful runs; 'nan'/empty means no successful run."""
    v = (v or '').strip()
    if v in ('', 'NaN', 'nan'):
        return float('nan')
    try:
        return float(v)
    except ValueError:
        return float('nan')


def read_summary():
    """Read all_summary.csv and return {problem: {algorithm: {metric: value}}}"""
    data = defaultdict(lambda: defaultdict(dict))
    csv_path = os.path.join(RESULTS_DIR, 'all_summary.csv')
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prob = row['Problem']
            alg = row['Algorithm']
            for m, _ in METRICS:
                data[prob][alg][m] = _to_float(row[m])
    return data


def compute_profile(data, metric):
    """Compute Dolan-More performance profile for a given metric.

    Returns (profiles, num_problems, ratios) where profiles maps each
    algorithm to (tau_array, rho_array) and ratios maps each algorithm to
    its raw ratio array (inf for failures, which include 'nan' medians).
    Only problems on which at least one algorithm has a finite median are
    used; a problem on which every algorithm fails is excluded (there is
    no such problem in the present data set).
    """
    valid_problems = sorted(
        p for p in data
        if '_partial' not in p
        and any(np.isfinite(data[p][a][metric]) for a in data[p])
    )
    num_problems = len(valid_problems)
    if num_problems == 0:
        return {}, 0, {}

    all_algs = sorted({a for p in valid_problems for a in data[p]})

    ratios = {alg: [] for alg in all_algs}
    for prob in valid_problems:
        vals = [data[prob][a][metric] for a in all_algs]
        finite = [v for v in vals if np.isfinite(v)]
        best = min(finite) if finite else float('inf')
        for a in all_algs:
            v = data[prob][a][metric]
            # failure (no successful run) -> ratio +infinity
            ratios[a].append(v / best if np.isfinite(v) and best > 0 else float('inf'))

    tau_grid = np.unique(np.concatenate([
        np.logspace(0, np.log10(TAU_MAX), 4000),
        np.array([t for a in all_algs for t in ratios[a] if np.isfinite(t)]),
        [1.0, TAU_MAX],
    ]))

    profiles = {}
    for alg in all_algs:
        r = np.sort(np.array([t for t in ratios[alg] if np.isfinite(t)]))
        rho = np.searchsorted(r, tau_grid, side='right') / num_problems
        profiles[alg] = (tau_grid, rho)
    return profiles, num_problems, ratios


def style_axis(ax, label):
    ax.set_xlabel(r'$\tau$', fontsize=16)
    ax.set_ylabel(r'$\rho_s(\tau)$', fontsize=16)
    ax.set_title(label, fontsize=17, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xlim(1, TAU_MAX)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(TICKS)
    ax.set_xticklabels([str(t) for t in TICKS])
    ax.minorticks_off()
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)


def plot_profiles(data):
    """Generate 2x2 subplot performance profiles."""
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2))
    axes = axes.flatten()

    subplot_labels = ['(a)', '(b)', '(c)', '(d)']
    handles, names = [], []

    for idx, (metric, label) in enumerate(METRICS):
        ax = axes[idx]
        profiles, n_prob, _ = compute_profile(data, metric)

        for alg_name, info in ALG_INFO.items():
            if alg_name in profiles:
                tau, rho = profiles[alg_name]
                h = ax.step(tau, rho, where='post', color=info['color'],
                            linestyle=info['ls'], linewidth=1.6, label=alg_name)
                if idx == 0:
                    handles.append(h[0])
                    names.append(alg_name)

        ax.text(0.03, 0.97, subplot_labels[idx], transform=ax.transAxes,
                fontsize=15, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='gray', alpha=0.8))
        style_axis(ax, label)

    # shared legend below the panels, so no curve is covered
    fig.legend(handles, names, loc='upper center', ncol=6, fontsize=12,
               frameon=False, handlelength=2.6, columnspacing=1.2,
               bbox_to_anchor=(0.5, 0.005))

    plt.tight_layout(rect=(0, 0.055, 1, 1))
    output_path = os.path.join(OUTPUT_DIR, 'perf_profiles.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), format='png',
                bbox_inches='tight', dpi=150)
    plt.close()
    print(f'Performance profiles saved to {output_path}')


if __name__ == '__main__':
    data = read_summary()
    print(f'Read data for {len(data)} problems')
    plot_profiles(data)
    print('Done!')
