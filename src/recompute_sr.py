# Added for the EBNPG manuscript.
# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np, scipy.io as sio, os, json, collections

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'crit_out')
probs = ['AP1','AP3','AP4','BK1','DD1','DGO2','DTLZ2','FA1','Far1','FDS','Hil1',
         'IKK1','KW2','Lov4','Lov5','Lov6','LTDZ','MGH16','MGH26','MGH33','MLF2',
         'MMR1','MMR2','MMR3','MOP5','MOP6','MOP7','NIC1','NIC2','NIC3','NIM1',
         'NSP1','NSP2','PNR','QV1','SK2','SP1','Toi4','Toi9','VU2','ZDT6','ZLT1']

algs = None
CR = {}   # CR[p][ai] = (crit, dnorm, info)
for p in probs:
    f = os.path.join(OUT, p + '_crit.mat')
    o = sio.loadmat(f)['out']
    def gv(e, n):
        # unwrap only while the container is a scalar object cell
        v = e[n]
        while v.dtype == object and v.size == 1:
            v = v.reshape(-1)[0]
        return v
    raw_names = np.asarray(gv(o, 'algorithm')).ravel()
    names = [str(np.asarray(x).ravel()[0]) for x in raw_names]
    if algs is None:
        algs = names
    CR[p] = (np.asarray(gv(o, 'crit')), np.asarray(gv(o, 'dnorm')),
             np.asarray(gv(o, 'info')))

nA = len(algs)
print('algs:', algs)

# distribution of -Theta for originally-failed vs originally-successful runs
succ_vals, fail_vals = [], []
for p in probs:
    c, d, info = CR[p]
    for ai in range(nA):
        ok = info[ai] == 0
        v = c[ai][ok]; v = v[np.isfinite(v)]
        succ_vals.extend(v.tolist())
        v2 = c[ai][~ok]; v2 = v2[np.isfinite(v2)]
        fail_vals.extend(v2.tolist())
succ_vals = np.array(succ_vals); fail_vals = np.array(fail_vals)
print('\n-Theta distribution')
print('  originally successful: n=%d  median=%.2e  q90=%.2e  q99=%.2e  max=%.2e'
      % (len(succ_vals), np.median(succ_vals), np.quantile(succ_vals,.90),
         np.quantile(succ_vals,.99), succ_vals.max()))
print('  originally failed    : n=%d  min=%.2e  q01=%.2e  q10=%.2e  median=%.2e'
      % (len(fail_vals), fail_vals.min(), np.quantile(fail_vals,.01),
         np.quantile(fail_vals,.10), np.median(fail_vals)))
for tol in [1e-4, 1e-6, 1e-8]:
    fr = (fail_vals <= tol).mean()
    sr = (succ_vals <= tol).mean()
    print('  tol=%.0e : frac of originally-failed below tol = %.4f ; of successful = %.4f' % (tol, fr, sr))

def sr_table(tol, mode):
    """mode: 'A' measure only, 'B' flag AND measure"""
    per_prob = np.full((len(probs), nA), np.nan)
    for pi, p in enumerate(probs):
        c, d, info = CR[p]
        for ai in range(nA):
            ct = c[ai]
            m = np.isfinite(ct) & (ct <= tol)
            if mode == 'B':
                m = m & (info[ai] == 0)
            per_prob[pi, ai] = 100.0 * m.sum() / ct.size
    return per_prob

print('\n=== average success rate over the 42 problems ===')
res = {}
for name, (tol, mode) in {'orig(flag)': (None, None),
                          'A:-Theta<=1e-4': (1e-4, 'A'),
                          'B:flag&-Theta<=1e-4': (1e-4, 'B'),
                          'A:-Theta<=1e-6': (1e-6, 'A')}.items():
    if tol is None:
        per_prob = np.full((len(probs), nA), np.nan)
        for pi, p in enumerate(probs):
            c, d, info = CR[p]
            for ai in range(nA):
                per_prob[pi, ai] = 100.0 * (info[ai] == 0).sum() / info[ai].size
    else:
        per_prob = sr_table(tol, mode)
    avg = np.nanmean(per_prob, axis=0)
    rob = (per_prob >= 90).sum(axis=0) / len(probs) * 100.0
    res[name] = (avg, rob, per_prob)
    print('\n-- %s --' % name)
    order = np.argsort(-avg)
    for k in order:
        print('  %-13s avg=%5.1f%%  robustness(>=90%% on)=%4.1f%%' % (algs[k], avg[k], rob[k]))

# per-problem original vs common-criterion for the record
ppA = res['A:-Theta<=1e-4'][2]
ppO = res['orig(flag)'][2]
print('\nmean absolute change (common - original) per algorithm:')
for ai in range(nA):
    print('  %-13s %+.2f pp' % (algs[ai], np.nanmean(ppA[:, ai] - ppO[:, ai])))

json.dump({'algs': algs,
           'avg': {k: res[k][0].tolist() for k in res},
           'rob': {k: res[k][1].tolist() for k in res}},
          open(os.path.join(BASE, 'common_sr.json'), 'w'), indent=1)
print('\nsaved common_sr.json')
