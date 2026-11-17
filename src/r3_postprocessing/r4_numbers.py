# Added for the EBNPG manuscript (third revision).
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-
"""Every number the R3 report and the response letter need, from one pass
over the certified stationarity table (Code_Data/crit_out_r4).

Rule used throughout (stated in the manuscript):
  a point satisfies a common criterion  <=>  its value is certified and <= tol.
A value is certified when the solver's optimum and the model evaluated at the
returned point agree to max(1e-10, 1e-12 |value|) and the point is in the box.
Uncertified / unsolved points are counted as failures; the denominator is
always the 100 runs of the (algorithm, problem) cell.
"""
import os
import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'Code_Data')
probs = ['AP1','AP3','AP4','BK1','DD1','DGO2','DTLZ2','FA1','Far1','FDS','Hil1',
         'IKK1','KW2','Lov4','Lov5','Lov6','LTDZ','MGH16','MGH26','MGH33','MLF2',
         'MMR1','MMR2','MMR3','MOP5','MOP6','MOP7','NIC1','NIC2','NIC3','NIM1',
         'NSP1','NSP2','PNR','QV1','SK2','SP1','Toi4','Toi9','VU2','ZDT6','ZLT1']
algs = ['MPG-Explicit','MPG-Armijo','MPG-Implicit','CNPG-Max','CNPG-Avg',
        'EBNPG-N0','EBNPG-N1','EBNPG-N2','EBNPG-N3','EBNPG-N5','EBNPG-N6','EBNPG-N7']

th = np.full((42, 12, 100), np.nan); dn = np.full((42, 12, 100), np.nan)
ce = np.zeros((42, 12, 100));        info = np.full((42, 12, 100), np.nan)
for i, p in enumerate(probs):
    o = sio.loadmat(os.path.join(DATA, 'crit_out_r4', p + '_crit.mat'),
                    squeeze_me=True, struct_as_record=False)['out']
    th[i] = np.atleast_2d(o.crit); dn[i] = np.atleast_2d(o.dnorm)
    ce[i] = np.atleast_2d(o.cert).astype(float)
    info[i] = np.atleast_2d(o.info)

OK = ce > 0                       # certified value
F = np.isfinite(th) & OK
FD = np.isfinite(dn) & OK
succ = info == 0

print('=== coverage')
print('points                       : %d' % th.size)
print('certified (-Theta and ||d||)  : %d / %d   (%.2f%%)' % (F.sum(), th.size, 100*F.sum()/th.size))
print('uncertified or unsolved      : %d' % (th.size - F.sum()))
print('  of which value <= 1e-4 (crit): %d' % ((np.isfinite(th) & (th <= 1e-4) & ~OK).sum()))
print('  of which value <= 1e-4 (disp): %d' % ((np.isfinite(dn) & (dn <= 1e-4) & ~OK).sum()))

def cols(mask_th, mask_dn):
    rt = np.array([[100.0*mask_th[i, a].sum()/100 for a in range(12)] for i in range(42)])
    rd = np.array([[100.0*mask_dn[i, a].sum()/100 for a in range(12)] for i in range(42)])
    return rt, rd, rt.mean(0), (rt >= 90).sum(0)/42*100, rd.mean(0), (rd >= 90).sum(0)/42*100

rt, rd, AT, RT, AD, RD = cols(F & (th <= 1e-4), FD & (dn <= 1e-4))
print()
print('=== Table 4 (certified rule)')
print('%-14s %8s %8s %8s %8s | %s' % ('Algorithm', 'avg-Th', 'rob-Th', 'avg-dn', 'rob-dn', 'one-pass rounded'))
for a in range(12):
    print('%-14s %8.4f %8.2f %8.4f %8.2f | %5.1f %5.1f %5.1f %5.1f' %
          (algs[a], AT[a], RT[a], AD[a], RD[a], AT[a], RT[a], AD[a], RD[a]))

print()
print('=== sensitivity: admitting the uncertified values that would pass')
At2, Rt2, Ad2, Rd2 = cols(F & (th <= 1e-4) | (np.isfinite(th) & (th <= 1e-4)), FD & (dn <= 1e-4))[2:]
print('max change in avg-Th : %.4f pp' % np.abs(At2-AT).max())
print('max change in avg-dn : %.4f pp' % np.abs(Ad2-AD).max())
print('max change in rob-Th : %.2f pp' % np.abs(Rt2-RT).max())
print('max change in rob-dn : %.2f pp' % np.abs(Rd2-RD).max())

print()
print('=== tolerance sensitivity (certified values only)')
for tol in (1e-4, 1e-6):
    _, _, A, R, _, _ = cols(F & (th <= tol), FD & (dn <= tol))
    print('  tol=%.0e  avg-Th: %s' % (tol, ' '.join('%.2f' % v for v in A)))
_, _, A6, R6, _, _ = cols(F & (th <= 1e-6), FD & (dn <= 1e-6))
print('  drop 1e-4 -> 1e-6 (pp): %s' % ' '.join('%.2f' % v for v in (AT - A6)))

print()
print('=== flag vs common criteria')
nsucc = int(succ.sum())
print('runs with info == 0                      : %d' % nsucc)
print('  of which -Theta <= 1e-4                : %d' % (succ & F & (th <= 1e-4)).sum())
print('  of which -Theta >  1e-4                : %d' % (succ & F & (th > 1e-4)).sum())
print('  per algorithm (-Theta > 1e-4):')
for a in range(12):
    ns = int((succ[:, a]).sum())
    nbad = int((succ[:, a] & F[:, a] & (th[:, a] > 1e-4)).sum())
    v = th[:, a][succ[:, a] & F[:, a] & (th[:, a] > 1e-4)]
    p99 = np.percentile(v, 99) if v.size else float('nan')
    print('    %-14s %4d / %4d  (%.2f%%)  p99=%.2e' % (algs[a], nbad, ns, 100.0*nbad/max(ns,1), p99))
print('  flagged failed but -Theta <= 1e-4      : %d' % ((~succ) & F & (th <= 1e-4)).sum())

print()
print('=== the two criteria are not interchangeable')
c_pass = F & (th <= 1e-4); d_pass = FD & (dn <= 1e-4)
print('  crit pass & disp pass : %d' % (c_pass & d_pass).sum())
print('  crit pass & disp fail : %d' % (c_pass & ~d_pass).sum())
print('  crit fail & disp pass : %d' % (~c_pass & d_pass).sum())
for a in range(12):
    n = int((~c_pass[:, a] & d_pass[:, a]).sum())
    if n:
        pp = [probs[i] for i in range(42) if (~c_pass[i, a] & d_pass[i, a]).any()]
        print('    %-14s %3d  on %s' % (algs[a], n, ', '.join(pp)))
bad = F & (dn**2 > 2*1e-4) & (th <= 1e-4)
print('  inequality ||d||^2 <= 2(-Theta) contradicted: %d' % bad.sum())
