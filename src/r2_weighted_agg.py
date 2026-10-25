# -*- coding: utf-8 -*-
# Added for the EBNPG manuscript.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rebuild Table 9 and the Subsection 6.11 prose numbers from the RERUN of the
adaptive EBNPG-W study.

The Euclidean side is read exactly as aggregate_all.py reads it (so the
Euclidean columns of Table 9 are unchanged); the adaptive side is read from
results_w_rebuilt/, which holds the rerun produced with Code/EBNPG_W.m.

Rounding follows the manuscript conventions: MATLAB round() (half away from
zero) for class means, one decimal in Table 9.
"""
import os
import json
import numpy as np
import scipy.io as sio

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, 'results')
RESW = os.path.join(BASE, 'results_w_rebuilt')
OUT = os.path.join(BASE, 'r2_weighted.json')

probs = ['AP1','AP3','AP4','BK1','DD1','DGO2','DTLZ2','FA1','Far1','FDS','Hil1',
         'IKK1','KW2','Lov4','Lov5','Lov6','LTDZ','MGH16','MGH26','MGH33','MLF2',
         'MMR1','MMR2','MMR3','MOP5','MOP6','MOP7','NIC1','NIC2','NIC3','NIM1',
         'NSP1','NSP2','PNR','QV1','SK2','SP1','Toi4','Toi9','VU2','ZDT6','ZLT1']
algs = ['MPG-Explicit','MPG-Armijo','MPG-Implicit','CNPG-Max','CNPG-Avg',
        'EBNPG-N0','EBNPG-N1','EBNPG-N2','EBNPG-N3','EBNPG-N5','EBNPG-N6','EBNPG-N7']
VARIANTS = ['EBNPG-N0','EBNPG-N1','EBNPG-N2','EBNPG-N3','EBNPG-N5','EBNPG-N6','EBNPG-N7']
SHORTV = ['N0','N1','N2','N3','N5','N6','N7']

CLASS = {}
for p in ['AP1','AP4','BK1','DD1','DGO2','FDS','IKK1','MOP7','NIC1','NIC2','NIC3','SP1','ZLT1']:
    CLASS[p] = 'Convex'
for p in ['AP3','DTLZ2','FA1','Hil1','KW2','Lov4','Lov5','Lov6','MMR1','MMR2','MMR3',
          'MOP5','MOP6','PNR','SK2','ZDT6']:
    CLASS[p] = 'Nonconvex'
for p in ['Far1','LTDZ','MGH26','MLF2','NIM1','QV1','VU2']:
    CLASS[p] = 'Multimodal'
for p in ['MGH16','MGH33','Toi4','Toi9']:
    CLASS[p] = 'Quadratic'
for p in ['NSP1','NSP2']:
    CLASS[p] = 'Nonuniform/polynomial'
assert len(CLASS) == 42 and set(CLASS) == set(probs)
CLASSES = ['Convex','Nonconvex','Multimodal','Quadratic','Nonuniform/polynomial']


def mround(x, nd=0):
    f = 10.0 ** nd
    return float(np.sign(x) * np.floor(np.abs(x) * f + 0.5) / f)


def gv(e, n):
    v = e[n]
    while v.dtype == object and v.size == 1:
        v = v.reshape(-1)[0]
    return v


def wsr_of(path):
    """success rate of the seven adaptive variants, from a <prob>_W_results.mat"""
    d = sio.loadmat(path)['prob_results']
    a = np.full(7, np.nan)
    for r in range(7):
        e = d[r, 0]
        inf = np.asarray(gv(e, 'info_vals')).ravel().astype(float)
        a[r] = 100.0 * (inf == 0).mean()
    return a


# ---------------- Euclidean side (12 x 42) ----------------------------------
E = np.zeros((12, 42))
for pi, p in enumerate(probs):
    d = sio.loadmat(os.path.join(RES, p + '_results.mat'))['prob_results']
    for a in range(12):
        e = d[a, 0]
        inf = np.asarray(gv(e, 'info_vals')).ravel().astype(float)
        E[a, pi] = 100.0 * (inf == 0).mean()

# ---------------- adaptive side (42 x 7) ------------------------------------
W = np.full((42, 7), np.nan)
missing = []
for pi, p in enumerate(probs):
    f = os.path.join(RESW, p + '_W_results.mat')
    if not os.path.exists(f):
        missing.append(p)
        continue
    W[pi] = wsr_of(f)
out = {'missing_problems': missing}
if missing:
    print('!! missing problems (not yet rerun): %s' % ', '.join(missing))

ESR = np.array([[E[algs.index(v), pi] for v in VARIANTS] for pi in range(42)])
WSR = W

# ---------------- Table 9 ----------------------------------------------------
tab9 = []
for cl in CLASSES:
    idx = [probs.index(p) for p in probs if CLASS[p] == cl]
    e_best = np.max(ESR[idx], axis=1).mean()
    e_med = np.median(ESR[idx], axis=1).mean()
    w_best = np.max(WSR[idx], axis=1).mean()
    w_med = np.median(WSR[idx], axis=1).mean()
    tab9.append(dict(cls=cl, n=len(idx),
                     e_best=mround(e_best, 1), e_med=mround(e_med, 1),
                     w_best=mround(w_best, 1), w_med=mround(w_med, 1),
                     delta=mround(w_best - e_best, 1),
                     raw=dict(e_best=float(e_best), e_med=float(e_med),
                              w_best=float(w_best), w_med=float(w_med),
                              delta=float(w_best - e_best))))
out['tab9'] = tab9

# ---------------- per-problem best / median ---------------------------------
EBEST = ESR.max(axis=1); WBEST = WSR.max(axis=1)
EMED = np.median(ESR, axis=1); WMED = np.median(WSR, axis=1)
dd = WBEST - EBEST
improve = sorted([(probs[i], float(EBEST[i]), float(WBEST[i]))
                  for i in range(42) if dd[i] > 1e-9], key=lambda t: -(t[2] - t[1]))
drop = sorted([(probs[i], float(EBEST[i]), float(WBEST[i]))
               for i in range(42) if dd[i] < -1e-9], key=lambda t: (t[2] - t[1]))
out['improve'] = dict(n=len(improve), top=improve[:8], all=improve)
out['drop'] = dict(n=len(drop), worst=drop[:6], all=drop,
                   lo=float(dd[dd < -1e-9].min()) if len(drop) else 0.0,
                   hi=float(dd[dd < -1e-9].max()) if len(drop) else 0.0)
out['n_total_improve'] = int((dd > 1e-9).sum())

# NIC1 detail
ni = probs.index('NIC1')
out['nic1'] = dict(e=[float(x) for x in ESR[ni]], w=[float(x) for x in WSR[ni]],
                   variants=SHORTV)

# per-variant overall mean SR, Euclidean vs adaptive
out['var_mean'] = {SHORTV[k]: (float(ESR[:, k].mean()), float(np.nanmean(WSR[:, k])))
                   for k in range(7)}
out['var_delta'] = {SHORTV[k]: float(np.nanmean(WSR[:, k]) - ESR[:, k].mean())
                    for k in range(7)}

# per-problem named entries used in the prose
named = {}
for p in ['MGH33','MGH26','PNR','AP3','Far1','LTDZ','Lov6','ZLT1','NIC1','DTLZ2']:
    i = probs.index(p)
    named[p] = dict(e_best=float(EBEST[i]), w_best=float(WBEST[i]),
                    e_med=float(EMED[i]), w_med=float(WMED[i]))
out['named'] = named

# convex delta (used by the "+6.2pp" sentence)
ci = [probs.index(p) for p in probs if CLASS[p] == 'Convex']
out['convex_delta'] = float(np.max(WSR[ci], axis=1).mean() - np.max(ESR[ci], axis=1).mean())
out['convex_delta_r'] = mround(out['convex_delta'], 1)

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=1)

# ---------------- printable report -------------------------------------------
print('=== Table 9 (per-class average success rate, %) ===')
print('%-22s %3s %7s %7s %7s %7s %7s' % ('class', 'n', 'E best', 'E med', 'W best', 'W med', 'd'))
for r in tab9:
    print('%-22s %3d %7.1f %7.1f %7.1f %7.1f %+7.1f' %
          (r['cls'], r['n'], r['e_best'], r['e_med'], r['w_best'], r['w_med'], r['delta']))
print()
print('NIC1 per variant  E: %s' % ['%g' % x for x in out['nic1']['e']])
print('NIC1 per variant  W: %s' % ['%g' % x for x in out['nic1']['w']])
print('improved problems : %d   dropped: %d' % (out['n_total_improve'], out['drop']['n']))
print('  top improvements: %s' % ['%s %g->%g' % t for t in out['improve']['top'][:6]])
print('  worst drops     : %s' % ['%s %g->%g' % t for t in out['drop']['worst'][:4]])
print('  drop range      : %.1f .. %.1f pp' % (out['drop']['lo'], out['drop']['hi']))
print('per-variant delta : %s' % ['%s %+.1f' % (k, v) for k, v in out['var_delta'].items()])
print('convex Delta best : %+.1f pp' % out['convex_delta'])
print()
print('wrote %s' % OUT)
