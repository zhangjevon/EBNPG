# Added for the EBNPG manuscript (third revision).
# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np, scipy.io as sio, os, json

_BASE = os.environ.get('EBPGM_DATA') or os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_BASE, 'results')
probs = ['AP1','AP3','AP4','BK1','DD1','DGO2','DTLZ2','FA1','Far1','FDS','Hil1',
         'IKK1','KW2','Lov4','Lov5','Lov6','LTDZ','MGH16','MGH26','MGH33','MLF2',
         'MMR1','MMR2','MMR3','MOP5','MOP6','MOP7','NIC1','NIC2','NIC3','NIM1',
         'NSP1','NSP2','PNR','QV1','SK2','SP1','Toi4','Toi9','VU2','ZDT6','ZLT1']

algs = None
# data[p][a] = dict(ngev, nhev, iter, time, info) arrays length 100
data = {}
for p in probs:
    m = sio.loadmat(os.path.join(RES, p + '_results.mat'))
    pr = m['prob_results']
    lst = []
    def g(e, name):
        v = e[name]
        while v.dtype == object:
            v = v[0, 0] if v.shape == (1, 1) else v[0]
        return v

    for a in range(pr.shape[0]):
        e = pr[a, 0]
        d = dict(
            alg=str(g(e, 'algorithm')[0]) if g(e, 'algorithm').dtype.kind in 'US' else str(g(e, 'algorithm')),
            ngev=np.asarray(g(e, 'ngev_vals')).ravel().astype(float),
            nhev=np.asarray(g(e, 'nhev_vals')).ravel().astype(float),
            it=np.asarray(g(e, 'iter_vals')).ravel().astype(float),
            tm=np.asarray(g(e, 'time_vals')).ravel().astype(float),
            info=np.asarray(g(e, 'info_vals')).ravel().astype(float),
        )
        lst.append(d)
    if algs is None:
        algs = [d['alg'] for d in lst]
    data[p] = lst

print('algorithms:', algs)
print('n problems:', len(data))

ratios = [1, 2, 5, 10, 20, 50]
rows = []
for rho in ratios:
    # median weighted cost per (problem, algorithm) over successful runs
    C = np.full((len(probs), len(algs)), np.nan)
    for pi, p in enumerate(probs):
        for ai in range(len(algs)):
            d = data[p][ai]
            ok = (d['info'] == 0) & np.isfinite(d['ngev']) & np.isfinite(d['nhev'])
            if ok.sum() == 0:
                continue
            c = d['ngev'][ok] + rho * d['nhev'][ok]
            C[pi, ai] = np.median(c)
    # wins: lowest median cost on the problem (ties -> all counted)
    wins = np.zeros(len(algs))
    rel = np.full((len(probs), len(algs)), np.nan)
    for pi in range(len(probs)):
        row = C[pi]
        if np.all(np.isnan(row)):
            continue
        mn = np.nanmin(row)
        for ai in range(len(algs)):
            if np.isnan(row[ai]):
                continue
            rel[pi, ai] = row[ai] / mn
            if row[ai] <= mn * (1 + 1e-12):
                wins[ai] += 1
    medrel = np.nanmedian(rel, axis=0)
    rows.append((rho, wins, medrel))
    print('\n--- w_h/w_g = %d ---' % rho)
    order = np.argsort(-wins)
    for ai in order:
        print('  %-13s wins=%2d/42   median ratio to best = %.2f' % (algs[ai], wins[ai], medrel[ai]))

# also store raw for later
out = {'provenance': {
           'script': 'oracle_cost.py',
           'inputs': 'results/<problem>_results.mat (ngev_vals, nhev_vals, iter_vals, '
                     'time_vals, info_vals)',
           'status': 'EARLIER VERSION of the oracle-cost computation; superseded by '
                     'r2_oracle_cost.py, which reads the same columns and is the one '
                     'that produces Table 8',
           'superseded_by': 'r2_oracle_cost.py',
           'reported_in': 'none (kept for reference)'},
       'algs': algs, 'probs': probs}
for rho, wins, medrel in rows:
    out['rho%d' % rho] = {'wins': wins.tolist(), 'medrel': medrel.tolist()}
json.dump(out, open(os.path.join(_BASE,'oracle_cost.json'), 'w'), indent=1)
print('\nsaved oracle_cost.json')
