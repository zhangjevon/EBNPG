# -*- coding: utf-8 -*-
"""LEGACY diagnostic: reconstruct superseded readings at returned directions.

These readings do not certify the minimizer of the common subproblem and enter
no final table.  Output is deliberately named legacy_adjudicated_returned.csv;
this historical helper cannot overwrite the final adjudicated.csv.  The final
pipeline is certify_adjudicated.py plus replay_adjudication.py.

Usage: python make_adjudicated.py [crit_out_r4]
"""
import csv, os, sys
import numpy as np
import scipy.io as sio

probs = ['AP1','AP3','AP4','BK1','DD1','DGO2','DTLZ2','FA1','Far1','FDS','Hil1',
         'IKK1','KW2','Lov4','Lov5','Lov6','LTDZ','MGH16','MGH26','MGH33','MLF2',
         'MMR1','MMR2','MMR3','MOP5','MOP6','MOP7','NIC1','NIC2','NIC3','NIM1',
         'NSP1','NSP2','PNR','QV1','SK2','SP1','Toi4','Toi9','VU2','ZDT6','ZLT1']
algs = ['MPG-Explicit','MPG-Armijo','MPG-Implicit','CNPG-Max','CNPG-Avg',
        'EBNPG-N0','EBNPG-N1','EBNPG-N2','EBNPG-N3','EBNPG-N5','EBNPG-N6','EBNPG-N7']
TOL = 1e-4

HERE = os.path.dirname(os.path.abspath(__file__))


def find_crit():
    d = HERE
    for _ in range(4):
        cand = os.path.join(d, 'Code_Data', 'crit_out_r4')
        if os.path.isdir(cand):
            return cand
        d = os.path.dirname(d)
    return os.path.join(HERE, '..', 'Code_Data', 'crit_out_r4')


CRIT = sys.argv[1] if len(sys.argv) > 1 else find_crit()


def gv(e, n):
    v = e[n]
    while v.dtype == object and v.size == 1:
        v = v.reshape(-1)[0]
    return v


# theta_def, written by recompute_common_stationarity_r4.m, one row per point
# that was re-solved in that pass.
td = {}
with open(os.path.join(CRIT, 'retry_log.csv')) as f:
    for r in csv.DictReader(f):
        td[(r['prob'], r['alg'], int(r['run']))] = float(r['theta_def'])

out = os.path.join(CRIT, 'legacy_adjudicated_returned.csv')
nrow = npass_th = npass_dn = 0
with open(out, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['prob', 'alg', 'run', 'theta_solver', 'theta_model', 'dnorm',
                'pass_theta', 'pass_dnorm'])
    for p in probs:
        o = sio.loadmat(os.path.join(CRIT, p + '_crit.mat'))['out']
        cert = np.asarray(gv(o, 'cert'), dtype=float)
        dn = np.asarray(gv(o, 'dnorm'), dtype=float)
        crit = np.asarray(gv(o, 'crit'), dtype=float)
        for a in range(len(algs)):
            for k in np.where(cert[a] == 0)[0]:
                key = (p, algs[a], k + 1)
                if key not in td or not np.isfinite(td[key]):
                    continue                      # one of the six unsolved points
                th_model = td[key]
                pt = int(-th_model <= TOL)
                pd = int(dn[a, k] <= TOL)
                nrow += 1
                npass_th += pt
                npass_dn += pd
                w.writerow([p, algs[a], k + 1, '%.12g' % (-crit[a, k]),
                            '%.12g' % th_model, '%.12g' % dn[a, k], pt, pd])

print('written %s' % out)
print('points with a value but no certificate : %d' % nrow)
print('  satisfying -Theta_1 <= %.0e           : %d' % (TOL, npass_th))
print('  satisfying ||d_1||  <= %.0e           : %d' % (TOL, npass_dn))
print('  counted as failures under both        : %d' % (nrow - max(npass_th, npass_dn)))
