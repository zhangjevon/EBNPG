# -*- coding: utf-8 -*-
# Added for the EBNPG manuscript.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Aggregate the fixed non-Euclidean kernel experiment (referee Comment 2).

K0 = Euclidean kernel (omega = 1/2||x||^2), K1 = fixed diagonal kernel
(omega = 1/2 x'diag(w)x, w frozen before the run).  Same 100 starting points
for both kernels on every problem, so the comparison is paired run by run
(McNemar is valid here, unlike in the main experiment).
"""
import numpy as np
import scipy.io as sio
import os, json
from scipy.stats import binomtest, wilcoxon

PROBS = ['AP1','AP3','AP4','BK1','DD1','DGO2','DTLZ2','FA1','Far1','FDS',
         'Hil1','IKK1','KW2','Lov4','Lov5','Lov6','LTDZ','MGH16','MGH26',
         'MGH33','MLF2','MMR1','MMR2','MMR3','MOP5','MOP6','MOP7','NIC1',
         'NIC2','NIC3','NIM1','NSP1','NSP2','PNR','QV1','SK2','SP1','Toi4',
         'Toi9','VU2','ZDT6','ZLT1']
D = 'ne_full'


def gv(e, n):
    v = e[n]
    while v.dtype == object and v.size == 1:
        v = v.reshape(-1)[0]
    return v


rows = []
for p in PROBS:
    f = os.path.join(D, '%s_N2.mat' % p)
    if not os.path.exists(f):
        continue
    m = sio.loadmat(f)['R']
    k0 = gv(m, 'K0')[0, 0]
    k1 = gv(m, 'K1')[0, 0]
    i0 = np.asarray(gv(k0, 'info')).ravel()
    i1 = np.asarray(gv(k1, 'info')).ravel()
    it0 = np.asarray(gv(k0, 'iter')).ravel()
    it1 = np.asarray(gv(k1, 'iter')).ravel()
    w = np.asarray(gv(m, 'w')).ravel()
    s0, s1 = (i0 == 0), (i1 == 0)
    b = int((s1 & ~s0).sum())      # K1 solves, K0 fails
    c = int((s0 & ~s1).sum())      # K0 solves, K1 fails
    pv = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    rows.append(dict(p=p, sr0=100 * s0.mean(), sr1=100 * s1.mean(),
                     mi0=np.median(it0[s0]) if s0.any() else np.nan,
                     mi1=np.median(it1[s1]) if s1.any() else np.nan,
                     b=b, c=c, pval=pv, wratio=float(w.max() / w.min())))

n = len(rows)
sr0 = np.array([r['sr0'] for r in rows])
sr1 = np.array([r['sr1'] for r in rows])
d = sr1 - sr0
print('problems: %d' % n)
print('mean success rate: K0 = %.2f   K1 = %.2f   difference = %+.2f pp'
      % (sr0.mean(), sr1.mean(), d.mean()))
print('K1 better on %d, worse on %d, tied on %d'
      % ((d > 1e-9).sum(), (d < -1e-9).sum(), (np.abs(d) <= 1e-9).sum()))

B = sum(r['b'] for r in rows)
C = sum(r['c'] for r in rows)
print('pooled McNemar over %d runs: K1-only=%d  K0-only=%d  p=%.3f'
      % (100 * n, B, C, binomtest(B, B + C, 0.5).pvalue))
print('Wilcoxon on the %d paired per-problem differences: p=%.3f'
      % (n, wilcoxon(d, zero_method='wilcox').pvalue))

mi0 = np.array([r['mi0'] for r in rows], float)
mi1 = np.array([r['mi1'] for r in rows], float)
ok = np.isfinite(mi0) & np.isfinite(mi1)
print('median iterations over problems: K0 = %.1f   K1 = %.1f  '
      '(K1 fewer on %d/%d)' % (np.median(mi0[ok]), np.median(mi1[ok]),
                               (mi1[ok] < mi0[ok]).sum(), ok.sum()))

# ---- split by the anisotropy of the kernel -------------------------------
wr = np.array([r['wratio'] for r in rows])
for thr, lab in [(1.5, 'near-isotropic kernels (max/min w < 1.5)'),
                 (2.0, 'mildly anisotropic (max/min w >= 1.5)'),
                 (3.0, 'clearly anisotropic (max/min w >= 3)')]:
    if thr == 1.5:
        sel = wr < 1.5
    elif thr == 2.0:
        sel = (wr >= 1.5) & (wr < 3.0)
    else:
        sel = wr >= 3.0
    if sel.sum() == 0:
        continue
    print('\n%-42s  n=%2d  K0=%6.2f  K1=%6.2f  diff=%+.2f pp'
          % (lab, sel.sum(), sr0[sel].mean(), sr1[sel].mean(), d[sel].mean()))
    bb = sum(rows[i]['b'] for i in range(n) if sel[i])
    cc = sum(rows[i]['c'] for i in range(n) if sel[i])
    if bb + cc:
        print('   McNemar: K1-only=%d K0-only=%d p=%.3g'
              % (bb, cc, binomtest(bb, bb + cc, 0.5).pvalue))

print('\nper-problem, sorted by K1-K0:')
print('%-6s %6s %6s %9s %9s %8s %9s' %
      ('prob', 'K0', 'K1', 'medIt K0', 'medIt K1', 'w ratio', 'McNemar p'))
for r in sorted(rows, key=lambda r: -(r['sr1'] - r['sr0'])):
    print('%-6s %6.1f %6.1f %9.1f %9.1f %8.2f %9.3f' %
          (r['p'], r['sr0'], r['sr1'], r['mi0'], r['mi1'], r['wratio'], r['pval']))

json.dump(rows, open('kernel_study.json', 'w'), indent=1)
print('\nsaved kernel_study.json')
