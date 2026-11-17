# Added for the EBNPG manuscript (third revision).
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Generate Dolan-More performance profiles from experiment results.
Reads all_summary.csv and generates perf_profiles.pdf with 2x2 subplots.
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
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results')

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
    ('MedIter',  'Iterations',       'perf_iter'),
    ('MedNgev',  'Function evals',   'perf_ngev'),
    ('MedNhev',  'Nonsmooth evals',  'perf_nhev'),
    ('MedTime',  'CPU time (s)',     'perf_time'),
]

def read_summary():
    """Read all_summary.csv and return {problem: {algorithm: {metric: value}}}"""
    data = defaultdict(lambda: defaultdict(dict))
    csv_path = os.path.join(RESULTS_DIR, 'all_summary.csv')
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prob = row['Problem']
            alg = row['Algorithm']
            for _, _, _ in METRICS:
                pass
            data[prob][alg]['MedIter']  = float(row['MedIter'])  if row['MedIter']  and row['MedIter']  != 'NaN' else float('inf')
            data[prob][alg]['MedNgev']  = float(row['MedNgev'])  if row['MedNgev']  and row['MedNgev']  != 'NaN' else float('inf')
            data[prob][alg]['MedNhev']  = float(row['MedNhev'])  if row['MedNhev']  and row['MedNhev']  != 'NaN' else float('inf')
            data[prob][alg]['MedTime']  = float(row['MedTime'])  if row['MedTime']  and row['MedTime']  != 'NaN' else float('inf')
            data[prob][alg]['SuccessRate'] = float(row['SuccessRate'])
    return data

def compute_profile(data, metric):
    """Compute Dolan-More performance profile for a given metric.
    Returns {algorithm: (tau_array, rho_array)}.
    Only includes problems where at least one solver has a finite value."""
    # Filter problems: only include if at least one solver has finite value
    # Also exclude "_partial" problems (incomplete experiment data)
    valid_problems = []
    for prob in data:
        if '_partial' in prob:
            continue
        has_finite = any(data[prob][alg][metric] < float('inf') for alg in data[prob])
        if has_finite:
            valid_problems.append(prob)
    
    num_problems = len(valid_problems)
    if num_problems == 0:
        return {}
    
    # Get all algorithms
    all_algs = set()
    for prob in valid_problems:
        all_algs.update(data[prob].keys())
    all_algs = sorted(all_algs)
    
    # Compute ratios
    ratios = {alg: [] for alg in all_algs}
    for prob in valid_problems:
        # Find best (minimum) value
        best = min(data[prob][alg][metric] for alg in all_algs if alg in data[prob])
        if best == float('inf'):
            best = 1.0  # Avoid division by zero if all are inf (shouldn't happen after filtering)
        
        for alg in all_algs:
            if alg in data[prob]:
                val = data[prob][alg][metric]
                ratios[alg].append(val / best if val < float('inf') else float('inf'))
            else:
                ratios[alg].append(float('inf'))
    
    # Compute CDF for each solver
    profiles = {}
    tau_max = 500.0  # Increased to capture worst algorithm
    # Use log scale for better visualization of both good and bad algorithms
    tau_array = np.logspace(0, np.log10(tau_max), 1000)  # Log scale from 1 to tau_max
    
    for alg in all_algs:
        r = np.array(ratios[alg])
        rho = np.array([np.sum(r <= t) / num_problems for t in tau_array])
        profiles[alg] = (tau_array, rho)
    
    return profiles, num_problems

def plot_profiles(data):
    """Generate 2x2 subplot performance profiles."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']
    
    for idx, (metric, label, _) in enumerate(METRICS):
        ax = axes[idx]
        profiles, n_prob = compute_profile(data, metric)
        
        # Find the maximum tau where any algorithm reaches rho=1.0
        max_tau_reach = 1.0
        for alg_name, info in ALG_INFO.items():
            if alg_name in profiles:
                tau, rho = profiles[alg_name]
                # Find where rho first reaches 1.0
                reach_idx = np.where(rho >= 1.0)[0]
                if len(reach_idx) > 0:
                    max_tau_reach = max(max_tau_reach, tau[reach_idx[0]])
        
        # Set tau_max based on actual data, with some margin
        tau_max = min(max_tau_reach * 1.1, 100)  # Cap at 100 for readability
        
        for alg_name, info in ALG_INFO.items():
            if alg_name in profiles:
                tau, rho = profiles[alg_name]
                ax.step(tau, rho, where='post', color=info['color'],
                       linestyle=info['ls'], linewidth=1.5, label=alg_name)
        
        # Add subplot label (a)(b)(c)(d)
        ax.text(0.02, 0.98, subplot_labels[idx], transform=ax.transAxes,
                fontsize=14, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='gray', alpha=0.8))
        
        ax.set_xlabel(f'$\\tau$ (log scale)', fontsize=11)
        ax.set_ylabel(f'$\\rho_s(\\tau)$', fontsize=11)
        ax.set_title(f'{label} ({n_prob} problems)', fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xlim(1, tau_max)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        
        # Add annotation showing where worst algorithm reaches rho=1.0
        if idx == 0 and max_tau_reach > 1.0:
            ax.annotate(f'worst: $\\tau={max_tau_reach:.1f}$', 
                       xy=(max_tau_reach, 1.0), 
                       xytext=(max_tau_reach * 0.7, 0.85),
                       fontsize=9, color='darkred',
                       arrowprops=dict(arrowstyle='->', color='darkred', lw=1.2),
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                               edgecolor='darkred', alpha=0.9))
        
        if idx == 0:
            ax.legend(fontsize=7, loc='lower right', ncol=2)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'perf_profiles.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', dpi=150)
    plt.close()
    print(f'Performance profiles saved to {output_path}')
    
    # Also save individual profiles
    for metric, label, fname in METRICS:
        fig, ax = plt.subplots(figsize=(8, 6))
        profiles, n_prob = compute_profile(data, metric)
        
        # Find max tau for this metric
        max_tau_reach = 1.0
        for alg_name, info in ALG_INFO.items():
            if alg_name in profiles:
                tau, rho = profiles[alg_name]
                reach_idx = np.where(rho >= 1.0)[0]
                if len(reach_idx) > 0:
                    max_tau_reach = max(max_tau_reach, tau[reach_idx[0]])
        tau_max = min(max_tau_reach * 1.1, 100)
        
        for alg_name, info in ALG_INFO.items():
            if alg_name in profiles:
                tau, rho = profiles[alg_name]
                ax.step(tau, rho, where='post', color=info['color'],
                       linestyle=info['ls'], linewidth=1.5, label=alg_name)
        ax.set_xlabel(r'$\tau$ (log scale)', fontsize=13)
        ax.set_ylabel(r'$\rho_s(\tau)$', fontsize=13)
        ax.set_title(f'Performance profile: {label} ({n_prob} problems)', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xlim(1, tau_max)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower right', ncol=2)
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f'{fname}.pdf')
        plt.savefig(path, format='pdf', bbox_inches='tight', dpi=300)
        plt.savefig(path.replace('.pdf', '.png'), format='png', bbox_inches='tight', dpi=150)
        plt.close()
        print(f'  {fname}.pdf saved')

if __name__ == '__main__':
    data = read_summary()
    print(f'Read data for {len(data)} problems')
    plot_profiles(data)
    print('Done!')
