# -*- coding: utf-8 -*-
# Added for the EBNPG manuscript.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regenerate all_summary.csv from the (updated) results/*.mat files.

Mirrors export_all_results.m: one row per (problem, algorithm) with the
per-problem median statistics used by generate_perf_profiles.py.
"""
import numpy as np, scipy.io as sio, os, csv

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
probs = ['AP1','AP3','AP4','BK1','DD1','DGO2','DTLZ2','FA1','Far1','FDS','Hil1',
         'IKK1','KW2','Lov4','Lov5','Lov6','LTDZ','MGH16','MGH26','MGH33','MLF2',
         'MMR1','MMR2','MMR3','MOP5','MOP6','MOP7','NIC1','NIC2','NIC3','NIM1',
         'NSP1','NSP2','PNR','QV1','SK2','SP1','Toi4','Toi9','VU2','ZDT6','ZLT1']

def gv(e, n):
    v = e[n]
    while v.dtype == object and v.size == 1:
        v = v.reshape(-1)[0]
    return v

rows = []
for p in probs:
    pr = sio.loadmat(os.path.join(BASE, p + '_results.mat'))['prob_results']
    for a in range(pr.shape[0]):
        r = pr[a, 0]
        def val(name):
            v = np.asarray(gv(r, name)).ravel()
            return v[0] if v.size else float('nan')
        rows.append(dict(Problem=p,
                         Algorithm=str(val('algorithm')),
                         n=int(val('n')), m=int(val('m')),
                         NumRuns=int(val('num_runs')),
                         SuccessRate=float(val('success_rate')),
                         MedIter=float(val('med_iter')),
                         MedNgev=float(val('med_ngev')),
                         MedNhev=float(val('med_nhev')),
                         MedTime=float(val('med_time'))))

with open(os.path.join(BASE, 'all_summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['Problem','Algorithm','n','m','NumRuns',
                                      'SuccessRate','MedIter','MedNgev','MedNhev','MedTime'])
    w.writeheader()
    for r in rows:
        w.writerow(r)
print('wrote all_summary.csv with %d rows' % len(rows))
for r in rows:
    if r['Problem'] == 'MGH33':
        print('  %-13s SR=%5.1f medIt=%s medTime=%s' % (r['Algorithm'], r['SuccessRate'],
              ('%.1f' % r['MedIter']) if r['MedIter'] == r['MedIter'] else 'NaN',
              ('%.2f' % r['MedTime']) if r['MedTime'] == r['MedTime'] else 'NaN'))
