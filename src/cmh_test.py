# -*- coding: utf-8 -*-
# Added for the EBNPG manuscript.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stratified (Cochran--Mantel--Haenszel) test for the success-rate ranking.

Data structure: 42 strata (the problems); in each stratum a 2x2 table
(algorithm vs. success/failure) over 100 runs.  The runs of two algorithms are
independent (the starting points are regenerated per algorithm slot), but the
stratum is common to both, which is exactly the situation CMH handles.

Reference algorithm: EBNPG-N2.  Holm's adjustment over the 11 comparisons.
Success flag: info_vals == 0.
"""
import numpy as np
import scipy.io as sio
from scipy.stats import chi2, wilcoxon
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


OK = np.zeros((12, len(PROBS)))
NN = np.zeros(len(PROBS), dtype=int)
for pi, p in enumerate(PROBS):
    d = sio.loadmat('results/%s_results.mat' % p)['prob_results']
    for a in range(12):
        inf = np.asarray(gv(d[a, 0], 'info_vals')).ravel().astype(float)
        OK[a, pi] = int((inf == 0).sum())
        NN[pi] = len(inf)

REF = ALGS.index('EBNPG-N2')
others = [a for a in range(12) if a != REF]


def cmh(a):
    """Cochran--Mantel--Haenszel statistic for algorithm a versus REF."""
    num = 0.0
    den = 0.0
    for k in range(len(PROBS)):
        n1 = n2 = NN[k]
        A = OK[REF, k]                    # ref successes
        C = OK[a, k]                      # opponent successes
        m1 = A + C                        # total successes in stratum
        N = n1 + n2
        m2 = N - m1
        if m1 == 0 or m2 == 0:
            continue
        num += A - n1 * m1 / N
        den += n1 * n2 * m1 * m2 / (N * N * (N - 1))
    if den <= 0:
        return 0.0, 1.0
    stat = num * num / den
    return stat, float(chi2.sf(stat, 1))


rows = []
for a in others:
    stat, p = cmh(a)
    d = 100 * (OK[REF] - OK[a]) / NN          # per-problem success-rate diff (pp)
    _, pw = wilcoxon(d, alternative='two-sided', zero_method='wilcox')
    rows.append((ALGS[a], float(np.mean(d)), stat, p, float(pw),
                 int((d > 1e-12).sum()), int((d < -1e-12).sum()),
                 int((np.abs(d) <= 1e-12).sum())))

order = sorted(range(len(rows)), key=lambda i: rows[i][3])
m = len(rows)
holm = [0.0] * len(rows)
prev = 0.0
for rank, i in enumerate(order):
    prev = max(prev, min(1.0, (m - rank) * rows[i][3]))
    holm[i] = prev

print('=== PRIMARY: Cochran--Mantel--Haenszel (42 strata = problems) ===')
print('EBNPG-N2 versus each opponent; mean difference in percentage points')
print('%-14s %8s %8s %5s %5s %11s %11s %11s' %
      ('opponent', 'mean d', 'W-SR p', 'win', 'loss', 'CMH chi2', 'p', 'p (Holm)'))
for i, (nm, md, stat, p, pw, w, l, t) in enumerate(rows):
    print('%-14s %+8.3f %8.3f %5d %5d %11.2f %11.2e %11.2e' %
          (nm, md, pw, w, l, stat, p, holm[i]))

nsig = sum(1 for i in range(len(rows)) if holm[i] < 0.05)
print('\nsignificant after Holm (alpha = 0.05): %d of %d' % (nsig, len(rows)))

json.dump({'test': 'CMH-stratified-by-problem',
           'secondary': 'wilcoxon-on-42-problems',
           'ref': ALGS[REF],
           'vs': [{'alg': r[0], 'mean_diff_pp': r[1], 'cmh_chi2': r[2],
                   'p': r[3], 'p_holm': holm[i], 'wilcoxon_p': r[4],
                   'n2_better': r[5], 'n2_worse': r[6], 'tied': r[7]}
                  for i, r in enumerate(rows)]},
          open('cmh_test.json', 'w'), indent=1)
print('saved cmh_test.json')
