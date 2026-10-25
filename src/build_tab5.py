# Added for the EBNPG manuscript.
# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np, scipy.io as sio, os, json

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, 'results')
QP2 = os.path.join(BASE, 'qp_out2')

probs = ['AP1','AP3','AP4','BK1','DD1','DGO2','DTLZ2','FA1','Far1','FDS','Hil1',
         'IKK1','KW2','Lov4','Lov5','Lov6','LTDZ','MGH16','MGH26','MGH33','MLF2',
         'MMR1','MMR2','MMR3','MOP5','MOP6','MOP7','NIC1','NIC2','NIC3','NIM1',
         'NSP1','NSP2','PNR','QV1','SK2','SP1','Toi4','Toi9','VU2','ZDT6','ZLT1']
algs = ['MPG-Explicit','MPG-Armijo','MPG-Implicit','CNPG-Max','CNPG-Avg',
        'EBNPG-N0','EBNPG-N1','EBNPG-N2','EBNPG-N3','EBNPG-N5','EBNPG-N6','EBNPG-N7']
plus_one = {'MPG-Explicit': 1, 'MPG-Armijo': 1}


def gv(e, n):
    v = e[n]
    while v.dtype == object and v.size == 1:
        v = v.reshape(-1)[0]
    return v


# ---- all-zero-success sets -------------------------------------------------
zero = {}
for ai, a in enumerate(algs):
    s = []
    for p in probs:
        pr = sio.loadmat(os.path.join(RES, p + '_results.mat'))['prob_results']
        inf = np.asarray(gv(pr[ai, 0], 'info_vals')).ravel().astype(float)
        if (inf == 0).sum() == 0:
            s.append(p)
    zero[a] = s
common = [p for p in probs if all(p not in zero[a] for a in algs)]
print('common subset size = %d' % len(common))
print('excluded:', [p for p in probs if p not in common])

# ---- MPG-Implicit measured QP counts --------------------------------------
imp_qp, imp_it = {}, {}
for p in probs:
    f = os.path.join(QP2, p + '_qp.mat')
    if not os.path.exists(f):
        continue
    o = sio.loadmat(f)['out']
    q = np.asarray(gv(o, 'qp_count')).ravel().astype(float)
    inf = np.asarray(gv(o, 'info')).ravel().astype(float)
    it = np.asarray(gv(o, 'iter')).ravel().astype(float)
    ok = inf == 0
    if ok.sum():
        imp_qp[p] = float(np.median(q[ok]))
        imp_it[p] = float(np.median(it[ok]))
print('MPG-Implicit measured on %d problems' % len(imp_qp))

# ---- per-problem medians on the common subset ------------------------------
rows = []
for ai, a in enumerate(algs):
    pit, pqp = [], []
    for p in common:
        pr = sio.loadmat(os.path.join(RES, p + '_results.mat'))['prob_results']
        e = pr[ai, 0]
        it = np.asarray(gv(e, 'iter_vals')).ravel().astype(float)
        inf = np.asarray(gv(e, 'info_vals')).ravel().astype(float)
        ok = inf == 0
        mit = float(np.median(it[ok]))
        if a == 'MPG-Implicit':
            mqp = imp_qp[p]
        else:
            mqp = mit + plus_one.get(a, 0)
        pit.append(mit)
        pqp.append(mqp)
    rows.append((a, np.median(pqp), np.median(pit), np.median(pqp) / np.median(pit)))

print('\n%-14s %10s %10s %8s' % ('Algorithm', 'med #QP', 'med iter', 'ratio'))
for a, q, t, r in rows:
    print('%-14s %10.1f %10.1f %8.2f' % (a, q, t, r))

# per-problem ratio for MPG-Implicit
r = np.array([imp_qp[p] / max(imp_it[p], 1) for p in imp_qp])
print('\nMPG-Implicit per-problem ratio: min=%.2f  q1=%.2f  median=%.2f  q3=%.2f  max=%.2f'
      % (r.min(), np.percentile(r, 25), np.median(r), np.percentile(r, 75), r.max()))
worst = sorted(imp_qp, key=lambda p: -imp_qp[p] / max(imp_it[p], 1))[:5]
print('largest ratios:', [(p, round(imp_qp[p] / max(imp_it[p], 1), 2)) for p in worst])

json.dump({'rows': [(a, float(q), float(t), float(rr)) for a, q, t, rr in rows],
           'common': common,
           'imp_ratio': {p: imp_qp[p] / max(imp_it[p], 1) for p in imp_qp}},
          open(os.path.join(BASE, 'tab5_data.json'), 'w'), indent=1)
