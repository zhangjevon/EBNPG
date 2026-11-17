# Added for the EBNPG manuscript.
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Generate Pareto front scatter plots for 2-objective problems.
Reads pareto_data.csv and generates PDF figures for selected problems.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The stored outputs live next to the code, not inside it: <root>/Code and
# <root>/Code_Data are siblings.  (An earlier version pointed at Code/results,
# a directory that does not exist, so the script could not run as shipped.)
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'Code_Data')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
OUTPUT_DIR = os.path.join(DATA_DIR, 'results')

# Algorithm colors and markers
# Updated for new 12-algorithm configuration: 3 MPG + 2 CNPG + 7 EBNPG
ALG_COLORS = {
    'MPG-Explicit':  '#1f77b4',
    'MPG-Armijo':    '#ff7f0e',
    'MPG-Implicit':  '#2ca02c',
    'CNPG-Max':      '#d62728',
    'CNPG-Avg':      '#9467bd',
    'EBNPG-N0':      '#8c564b',
    'EBNPG-N1':      '#e377c2',
    'EBNPG-N2':      '#7f7f7f',
    'EBNPG-N3':      '#bcbd22',
    'EBNPG-N5':      '#17becf',
    'EBNPG-N6':      '#aec7e8',
    'EBNPG-N7':      '#ff9896',
}

ALG_MARKERS = {
    'MPG-Explicit': 'o', 'MPG-Armijo': 's', 'MPG-Implicit': '^',
    'CNPG-Max': 'D', 'CNPG-Avg': 'v',
    'EBNPG-N0': 'o', 'EBNPG-N1': 's', 'EBNPG-N2': '^',
    'EBNPG-N3': 'D', 'EBNPG-N5': 'v', 'EBNPG-N6': 'P', 'EBNPG-N7': 'X',
}

# Select representative 2-objective problems for Pareto front visualization.
# These are exactly the six problems discussed in Subsection 6.8 of the
# manuscript: convex (BK1, DD1, DGO2), nonconvex (AP3, Hil1) and multimodal
# (Far1).  Keep this list in step with the text: the fallback below used to
# pick the first six names in alphabetical order, which is why an earlier
# version of the figure did not match the prose.
SELECTED_PROBLEMS_2x3 = ['BK1', 'DD1', 'DGO2', 'AP3', 'Hil1', 'Far1']

def read_pareto_data():
    """Read the Pareto data and return {problem: {algorithm: [(F1, F2), ...]}}

    The figure must show the *composite* objectives F_i = f_i + g_i.  Those are
    produced by export_pareto_composite.m (r3_postprocessing), which evaluates
    g_i(x) = delta*||B_i^{-T}x||_1 at the stored final points of the successful
    runs.  The older results/pareto_data.csv holds the smooth components f_i
    only and is used solely as a fallback.
    """
    data = defaultdict(lambda: defaultdict(list))
    for name, c1, c2 in ((os.path.join(DATA_DIR, 'pareto_data_composite.csv'), 'F1', 'F2'),
                         (os.path.join(RESULTS_DIR, 'pareto_data.csv'), 'f1', 'f2')):
        csv_path = name
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prob = row['Problem']
                alg = row['Algorithm']
                try:
                    f1 = float(row[c1]); f2 = float(row[c2])
                except (KeyError, ValueError):
                    continue
                if not (np.isnan(f1) or np.isnan(f2)):
                    data[prob][alg].append((f1, f2))
        return data
    print(f'Warning: no Pareto data found under {DATA_DIR}')
    return data


def plot_pareto_fronts(data, selected=None, filename='pareto_fronts.pdf'):
    """Generate scatter plots of Pareto fronts for selected problems."""
    if selected is None:
        # Use all 2-objective problems that have data
        selected = sorted(data.keys())
    
    n = len(selected)
    if n == 0:
        print('No 2-objective problems with data')
        return
    
    # Determine subplot layout
    if n <= 3:
        nrows, ncols = 1, n
    elif n <= 6:
        nrows, ncols = 2, 3
    elif n <= 9:
        nrows, ncols = 3, 3
    elif n <= 16:
        nrows, ncols = 4, 4
    elif n <= 25:
        nrows, ncols = 5, 5
    elif n <= 36:
        nrows, ncols = 6, 6
    else:
        nrows, ncols = 7, 6
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, prob in enumerate(selected):
        ax = axes[idx]
        if prob not in data:
            ax.text(0.5, 0.5, f'{prob}\n(no data)', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(prob, fontsize=12, fontweight='bold')
            continue
        
        prob_data = data[prob]
        for alg_name in ALG_COLORS:
            if alg_name in prob_data and len(prob_data[alg_name]) > 0:
                points = np.array(prob_data[alg_name])
                ax.scatter(points[:, 0], points[:, 1], c=ALG_COLORS[alg_name],
                          marker=ALG_MARKERS[alg_name], s=30, alpha=0.7,
                          edgecolors='white', linewidths=0.3, label=alg_name)
        
        ax.set_xlabel(r'$F_1$', fontsize=11)
        ax.set_ylabel(r'$F_2$', fontsize=11)
        ax.set_title(prob, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2)
        
        if idx == 0:
            ax.legend(fontsize=6, loc='best', ncol=2, framealpha=0.8)
    
    # Hide unused subplots
    for idx in range(len(selected), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', dpi=150)
    plt.close()
    print(f'Pareto fronts saved to {output_path}')

def plot_pareto_comparison(data, problem_name, filename=None):
    """Generate a detailed Pareto front for a single problem with all algorithms."""
    if problem_name not in data:
        print(f'{problem_name} not in data')
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    prob_data = data[problem_name]
    
    for alg_name in ALG_COLORS:
        if alg_name in prob_data and len(prob_data[alg_name]) > 0:
            points = np.array(prob_data[alg_name])
            ax.scatter(points[:, 0], points[:, 1], c=ALG_COLORS[alg_name],
                      marker=ALG_MARKERS[alg_name], s=50, alpha=0.7,
                      edgecolors='white', linewidths=0.3, label=alg_name)
    
    ax.set_xlabel(r'$F_1$', fontsize=13)
    ax.set_ylabel(r'$F_2$', fontsize=13)
    ax.set_title(f'Pareto front: {problem_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, loc='best', ncol=2, framealpha=0.8)
    plt.tight_layout()
    
    if filename is None:
        filename = f'pareto_{problem_name}.pdf'
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  {filename} saved')

if __name__ == '__main__':
    data = read_pareto_data()
    print(f'Read Pareto data for {len(data)} problems')
    
    if len(data) > 0:
        # Generate the 2x3 grid for the six problems named in Subsection 6.8
        available_selected = [p for p in SELECTED_PROBLEMS_2x3 if p in data]
        if len(available_selected) < len(SELECTED_PROBLEMS_2x3):
            missing = [p for p in SELECTED_PROBLEMS_2x3 if p not in data]
            print(f'WARNING: no data for {missing}; the figure will not match the text')
        plot_pareto_fronts(data, selected=available_selected[:6], filename='pareto_fronts_6.pdf')
        
        # Also generate individual plots for the same problems
        for prob in SELECTED_PROBLEMS_2x3:
            if prob in data:
                plot_pareto_comparison(data, prob)
        
        # Generate a larger grid with all 2-objective problems
        all_probs = sorted(data.keys())
        if len(all_probs) > 6:
            plot_pareto_fronts(data, selected=all_probs, filename='pareto_fronts_all.pdf')
    
    print('Done!')
