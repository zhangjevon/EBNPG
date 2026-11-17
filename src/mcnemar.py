# Added for the EBNPG manuscript (third revision).
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-
"""SUPERSEDED: instance-level McNemar analysis of the main comparison.

This script was the first attempt at a significance test for the success-rate
ranking.  It pairs the twelve algorithms run by run, i.e. it treats the 4,200
instances of a problem as 4,200 paired observations.

That pairing does not exist.  ``run_problem.m`` draws the starting point inside
the algorithm loop,

    for aidx = 1:12
        for run = 1:100
            x0_run = l + (u-l).*rand(n,1);

so each algorithm slot consumes its own block of 100 draws: the starting points
are identically distributed across algorithms but are NOT the same points (see
``check_startpoints.m``).  The manuscript states this, and the test reported
there takes the problem --- not the run --- as the paired unit
(``cmh_test.py`` for the stratified CMH test, ``paired_test.py`` for the
Wilcoxon test on the 42 per-problem success rates).

The script is retained because its output documents *why* the run-level pairing
was abandoned: the resulting p-values are anti-conservative by construction.
It enters no reported number of the manuscript.

Paths are resolved from this file, not from the working directory.
"""
import numpy as np
import scipy.io as sio
from scipy.stats import binomtest
import json, itertools, os

BASE = os.environ.get('EBPGM_DATA') or os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, 'results')

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


# flag[p][a] = boolean array over the 100 starting points
flag = {}
npts = {}
for p in PROBS:
    d = sio.loadmat(os.path.join(RES, '%s_results.mat' % p))['prob_results']
    row = []
    for a in range(12):
        inf = np.asarray(gv(d[a, 0], 'info_vals')).ravel().astype(float)
        row.append(inf == 0)
        npts[p] = len(inf)
    flag[p] = row

print('total paired instances: %d' % sum(npts.values()))

REF = ALGS.index('EBNPG-N2')


def mcnemar(a, b):
    """Return (b_only, c_only, p) for a vs b over all paired instances.

    b_only = instances where a succeeds and b fails
    c_only = instances where a fails and b succeeds
    """
    na = nc = 0
    per_problem = {}
    for p in PROBS:
        fa, fb = flag[p][a], flag[p][b]
        x = int(((fa) & (~fb)).sum())
        y = int(((~fa) & (fb)).sum())
        na += x
        nc += y
        per_problem[p] = (x, y)
    n = na + nc
    if n == 0:
        return na, nc, 1.0, per_problem
    # exact two-sided binomial test with p = 0.5
    res = binomtest(na, n, 0.5, alternative='two-sided')
    return na, nc, res.pvalue, per_problem


# ---- (1) EBNPG-N2 against each of the other eleven -------------------------
rows = []
for a in range(12):
    if a == REF:
        continue
    b_only, c_only, pv, per = mcnemar(REF, a)
    rows.append((ALGS[a], b_only, c_only, pv, per))

# Holm-Bonferroni correction over the 11 comparisons
order = sorted(range(len(rows)), key=lambda i: rows[i][3])
m = len(rows)
adj = [0.0] * len(rows)
prev = 0.0
for rank, i in enumerate(order):
    val = min(1.0, (m - rank) * rows[i][3])
    prev = max(prev, val)
    adj[i] = prev

print('\n=== McNemar: EBNPG-N2 versus each competitor (4,200 paired instances) ===')
print('%-14s %8s %8s %10s %12s' % ('competitor', 'N2 only', 'other only', 'p (raw)', 'p (Holm)'))
for i, (name, bo, co, pv, _) in enumerate(rows):
    print('%-14s %8d %8d %10.2e %12.2e' % (name, bo, co, pv, adj[i]))

# ---- (2) within the leading group ------------------------------------------
print('\n=== Pairwise inside the leading group (93.9 / 93.5 / 93.1 / 93.0 %) ===')
grp = ['EBNPG-N2', 'EBNPG-N6', 'CNPG-Max', 'EBNPG-N3']
for u, v in itertools.combinations(grp, 2):
    bo, co, pv, _ = mcnemar(ALGS.index(u), ALGS.index(v))
    print('%-10s vs %-10s  %5d / %5d   p = %.3f' % (u, v, bo, co, pv))

# ---- (3) how often is a per-problem comparison significant -----------------
print('\n=== Per-problem McNemar (EBNPG-N2 vs each, 100 pairs each) ===')
for i, (name, bo, co, pv, per) in enumerate(rows):
    sig = sum(1 for p in PROBS
              if (lambda t: binomtest(t[0], t[0] + t[1], 0.5).pvalue if t[0] + t[1] > 0 else 1.0)(per[p]) < 0.05)
    print('%-14s significant on %2d / 42 problems' % (name, sig))

out = {'provenance': {
           'script': 'mcnemar.py',
           'inputs': 'results/<problem>_results.mat (info_vals column)',
           'status': 'SUPERSEDED -- enters no reported number',
           'why': 'treats the 100 runs of a problem as paired across algorithms, but '
                  'run_problem.m redraws the starting point per algorithm slot; the '
                  'manuscript pairs at problem level instead',
           'superseded_by': ['cmh_test.py', 'paired_test.py'],
           'success_flag': 'info_vals == 0',
           'reported_in': 'none'},
       'ref': ALGS[REF],
       'vs': [{'alg': r[0], 'n2_only': r[1], 'other_only': r[2],
               'p': r[3], 'p_holm': adj[i],
               'per_problem': {k: list(v) for k, v in r[4].items()}}
              for i, r in enumerate(rows)]}
json.dump(out, open(os.path.join(BASE, 'mcnemar.json'), 'w'), indent=1)
print('\nsaved mcnemar.json')
