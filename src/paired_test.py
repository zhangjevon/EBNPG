# -*- coding: utf-8 -*-
# Added for the EBNPG manuscript.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Significance testing for the success-rate ranking (referee Comment 7).

IMPORTANT FACT ABOUT THE DATA GENERATION (verified with check_startpoints.m):
run_problem.m draws the starting point INSIDE the algorithm loop,

    for aidx = 1:12
        for run = 1:100
            x0_run = l + (u-l).*rand(n,1);

so each algorithm slot consumes its own block of 100 draws from the stream
seeded by rng(123).  The starting points are therefore identically distributed
across algorithms but NOT the same points, so a per-instance paired test
(McNemar) is not valid.

What IS paired is the problem: for every problem both algorithms are run on
100 starting points drawn from the same distribution.  We therefore use the
42 per-problem success rates as 42 paired observations and apply the
Wilcoxon signed-rank test.  A two-sample proportion test on the pooled
4,200 vs 4,200 runs is reported only as a secondary, anti-conservative
reference (it ignores the clustering of runs within problems).

Success flag: info_vals == 0   (1 and -2 are two failure modes).
"""
import numpy as np
import scipy.io as sio
from scipy.stats import wilcoxon, binomtest
import json

PROBS = ['AP1','AP3','AP4','BK1','DD1','DGO2','DTLZ2','FA1','Far1','FDS','Hil1',
         'IKK1','KW2','Lov4','Lov5','Lov6','LTDZ','MGH16','MGH26','MGH33','MLF2',
         'MMR1','MMR2','MMR3','MOP5','MOP6','MOP7','NIC1','NIC2','NIC3','NIM1',
         'NSP1','NSP2','PNR','QV1','SK2','SP1','Toi4','Toi9','VU2','ZDT6','ZLT1']
ALGS = ['MPG-Explicit','MPG-Armijo','MPG-Implicit','CNPG-Max','CNPG-Avg',
        'EBNPG-N0','EBNPG-N1','EBNPG-N2','EBNPG-N3','EBNPG-N5','EBNPG-N6','EBNPG-N7']


def gv(e, n):
    v = e[n]
    while v.dtype == object and v.size == 1:
        v = v.reshape(-1)[0]
    return v


# SR[a][p] = success rate in percent on problem p
SR = np.zeros((12, len(PROBS)))
OK = np.zeros((12, len(PROBS)))       # number of successful runs
for pi, p in enumerate(PROBS):
    d = sio.loadmat('results/%s_results.mat' % p)['prob_results']
    for a in range(12):
        inf = np.asarray(gv(d[a, 0], 'info_vals')).ravel().astype(float)
        ok = int((inf == 0).sum())
        OK[a, pi] = ok
        SR[a, pi] = 100.0 * ok / len(inf)

print('overall mean success rate (average over the 42 problems):')
for a in range(12):
    print('  %-14s %6.2f' % (ALGS[a], SR[a].mean()))

REF = ALGS.index('EBNPG-N2')
others = [a for a in range(12) if a != REF]

# ---------- primary: Wilcoxon signed-rank on the 42 paired problems ---------
rows = []
for a in others:
    d = SR[REF] - SR[a]
    nz = d[np.abs(d) > 1e-12]
    try:
        p = wilcoxon(d, alternative='two-sided', zero_method='wilcox').pvalue
    except ValueError:
        p = 1.0
    rows.append((ALGS[a], int((d > 1e-12).sum()), int((d < -1e-12).sum()),
                 int((np.abs(d) <= 1e-12).sum()), float(np.mean(d)), p))

# Holm over the 11 comparisons
order = sorted(range(len(rows)), key=lambda i: rows[i][5])
m = len(rows)
holm = [0.0] * len(rows)
prev = 0.0
for rank, i in enumerate(order):
    prev = max(prev, min(1.0, (m - rank) * rows[i][5]))
    holm[i] = prev

print('\n=== PRIMARY: Wilcoxon signed-rank on 42 paired problems '
      '(EBNPG-N2 minus opponent) ===')
print('%-14s %6s %6s %6s %9s %11s %11s' %
      ('opponent', 'better', 'worse', 'tied', 'mean d', 'p', 'p (Holm)'))
for i, (nm, b, w, t, md, p) in enumerate(rows):
    print('%-14s %6d %6d %6d %+9.3f %11.2e %11.2e' % (nm, b, w, t, md, p, holm[i]))

# ---------- secondary: pooled two-proportion (anti-conservative) ------------
print('\n=== SECONDARY: pooled two-proportion test on 4,200 vs 4,200 runs ===')
for i, (nm, b, w, t, md, p) in enumerate(rows):
    a = ALGS.index(nm)
    n1 = n2 = 4200
    k1, k2 = int(OK[REF].sum()), int(OK[a].sum())
    nb, nc = k1, k2
    # normal approximation with continuity correction
    pp = (k1 + k2) / (n1 + n2)
    se = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z = (k1 / n1 - k2 / n2) / se
    from scipy.stats import norm
    pv = 2 * (1 - norm.cdf(abs(z)))
    print('%-14s  %4.2f%% vs %4.2f%%   z = %6.2f   p = %.2e' %
          (nm, 100 * k1 / n1, 100 * k2 / n2, z, pv))

out = {'test': 'wilcoxon-on-42-paired-problems',
       'ref': ALGS[REF],
       'note': 'starting points are NOT shared across algorithms '
               '(regenerated per algorithm slot); pairing is at problem level',
       'vs': [{'alg': r[0], 'n2_better': r[1], 'n2_worse': r[2], 'tied': r[3],
               'mean_diff_pp': r[4], 'p': r[5], 'p_holm': holm[i]}
              for i, r in enumerate(rows)]}
json.dump(out, open('paired_test.json', 'w'), indent=1)
print('\nsaved paired_test.json')
