# Added for the EBNPG manuscript (third revision).
# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np, scipy.io as sio, os, json

_BASE = os.environ.get('EBPGM_DATA') or os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_BASE, 'results')
probs = ['AP1','AP3','AP4','BK1','DD1','DGO2','DTLZ2','FA1','Far1','FDS','Hil1',
         'IKK1','KW2','Lov4','Lov5','Lov6','LTDZ','MGH16','MGH26','MGH33','MLF2',
         'MMR1','MMR2','MMR3','MOP5','MOP6','MOP7','NIC1','NIC2','NIC3','NIM1',
         'NSP1','NSP2','PNR','QV1','SK2','SP1','Toi4','Toi9','VU2','ZDT6','ZLT1']

def g(e, name):
    v = e[name]
    while v.dtype == object:
        v = v[0, 0] if v.shape == (1, 1) else v[0]
    return np.asarray(v)

algs = None
D = {}
for p in probs:
    m = sio.loadmat(os.path.join(RES, p + '_results.mat'))
    pr = m['prob_results']
    lst = []
    def gv(e, name):
        v = e[name]
        while v.dtype == object:
            v = v[0, 0] if v.shape == (1, 1) else v[0]
        return v
    for a in range(pr.shape[0]):
        e = pr[a, 0]
        nm = str(gv(e, 'algorithm'))
        lst.append(dict(alg=nm,
                        ngev=g(e, 'ngev_vals').ravel().astype(float),
                        nhev=g(e, 'nhev_vals').ravel().astype(float),
                        it=g(e, 'iter_vals').ravel().astype(float),
                        tm=g(e, 'time_vals').ravel().astype(float),
                        info=g(e, 'info_vals').ravel().astype(float)))
    if algs is None:
        algs = [d['alg'] for d in lst]
    D[p] = lst

# ---------- per-iteration oracle intensities ----------
print('=== mean evaluations per iteration (successful runs, pooled over 42 problems) ===')
print('%-13s %8s %8s %10s' % ('algorithm', 'n_g/it', 'n_h/it', 'n_h/(m*it)'))
perit_h = {}
for ai, a in enumerate(algs):
    num_g = num_h = num_it = 0.0
    for p in probs:
        d = D[p][ai]
        ok = (d['info'] == 0) & (d['it'] > 0)
        if ok.sum() == 0: continue
        num_g += d['ngev'][ok].sum(); num_h += d['nhev'][ok].sum(); num_it += d['it'][ok].sum()
    print('%-13s %8.2f %8.2f' % (a, num_g/num_it, num_h/num_it))
    perit_h[a] = num_h/num_it

# ---------- break-even diagnostic: MPG trials per iteration ----------
print('\n=== trials per iteration implied for MPG (n_h/(m*iter)) per problem ===')
mpg_trials = {}
for p in probs:
    m = sio.loadmat(os.path.join(RES, p + '_results.mat'))
    pr = m['prob_results']
    e = pr[0, 0]  # MPG-Explicit
    def gv(ee, name):
        v = ee[name]
        while v.dtype == object:
            v = v[0, 0] if v.shape == (1, 1) else v[0]
        return v
    mm = int(np.asarray(gv(e, 'm')).ravel()[0])
    it = g(e, 'iter_vals').ravel().astype(float)
    nh = g(e, 'nhev_vals').ravel().astype(float)
    inf = g(e, 'info_vals').ravel().astype(float)
    ok = (inf == 0) & (it > 0)
    t = nh[ok].sum() / (mm * it[ok].sum()) if ok.sum() else np.nan
    mpg_trials[p] = t
vals = np.array([mpg_trials[p] for p in probs])
print('mean %.2f  median %.2f  min %.2f  max %.2f' % (np.nanmean(vals), np.nanmedian(vals), np.nanmin(vals), np.nanmax(vals)))
above = [p for p in probs if mpg_trials[p] > 2]
print('problems with MPG trials/iter > 2 (EBNPG then uses fewer nonsmooth evals): %d -> %s'
      % (len(above), ', '.join(above)))

# ---------- weighted oracle cost ----------
ratios = [1, 5, 20]
print('\n=== weighted oracle cost: wins out of 42 (ties counted) ===')
print('%-13s %8s %8s %8s %10s' % ('algorithm', 'rho=1', 'rho=5', 'rho=20', 'medrel20'))
tab = {}
for rho in ratios:
    C = np.full((len(probs), len(algs)), np.nan)
    for pi, p in enumerate(probs):
        for ai in range(len(algs)):
            d = D[p][ai]
            ok = (d['info'] == 0) & np.isfinite(d['ngev']) & np.isfinite(d['nhev'])
            if ok.sum() == 0: continue
            C[pi, ai] = np.median(d['ngev'][ok] + rho * d['nhev'][ok])
    wins = np.zeros(len(algs)); rel = np.full((len(probs), len(algs)), np.nan)
    for pi in range(len(probs)):
        row = C[pi]
        if np.all(np.isnan(row)): continue
        mn = np.nanmin(row)
        for ai in range(len(algs)):
            if np.isnan(row[ai]): continue
            rel[pi, ai] = row[ai]/mn
            if row[ai] <= mn*(1+1e-12): wins[ai] += 1
    tab[rho] = (wins, np.nanmedian(rel, axis=0))
for ai, a in enumerate(algs):
    print('%-13s %8d %8d %8d %10.2f' % (a, tab[1][0][ai], tab[5][0][ai], tab[20][0][ai], tab[20][1][ai]))

json.dump({'provenance': {
               'script': 'oracle_cost2.py',
               'inputs': 'results/<problem>_results.mat (ngev_vals, nhev_vals, iter_vals)',
               'status': 'EARLIER VERSION of the oracle-cost computation; superseded by '
                         'r2_oracle_cost.py, which is the one that produces Table 8',
               'superseded_by': 'r2_oracle_cost.py',
               'reported_in': 'none (kept for reference)'},
           'algs': algs, 'per_iter_h': perit_h,
           'mpg_trials': mpg_trials,
           'wins': {str(r): tab[r][0].tolist() for r in ratios},
           'medrel': {str(r): tab[r][1].tolist() for r in ratios}},
          open(os.path.join(_BASE,'oracle_cost2.json'), 'w'), indent=1)
print('\nsaved')
