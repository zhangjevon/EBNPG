# Added for the EBNPG manuscript (third revision).
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-
"""Success rates under the original flag and under the common criteria.

Produces ``common_sr.json``: for the twelve algorithms, the average success rate
over the 42 problems and the robustness (fraction of problems on which the
success rate is at least 90%), computed

  * from the flag the algorithm itself returned (``info_vals == 0``), and
  * from the two common criteria evaluated at every returned point with
    beta = 1 and the Euclidean kernel:  -Theta_1(x) <= 1e-4  and
    ||d_1(x)|| <= 1e-4, each with and without the additional requirement that
    the algorithm declared the run successful.

Input.  ``crit_out_r4/<prob>_crit.mat`` (the final pass) and, when present,
``crit_out_r4/adjudicated.csv``.  The values the pass could not certify are
masked exactly as in ``aggregate_all.py`` and ``r4_table4.py``: a point whose
``cert`` flag is zero is treated as having no value, so it cannot count as a
success -- except at the points that ``adjudicated.csv`` decides, where the
criterion is settled on the value the model attains at the returned point (the
value the solver returns there is positive by about 1e-9, which no value of
Theta_1 can be) and on the displacement of that point.  Set ``EBPGM_CRIT`` to
read a different table (for instance ``crit_out``, the first pass, where every
stored value is certified by construction and the 607 re-solved points are
absent).

The numbers of this file are the ones printed in Table 4 of the manuscript.
They coincide with ``r4_table4.py``; the two scripts differ only in layout.
Earlier versions of this package shipped a ``common_sr.json`` produced from
``crit_out`` without the certification mask, without the verdict-level
adjudication and without the displacement criterion; that file no longer
reproduces and has been replaced.
"""
import numpy as np, scipy.io as sio, os, json

_BASE = os.environ.get('EBPGM_DATA') or os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('EBPGM_CRIT') or os.path.join(_BASE, 'crit_out_r4')
probs = ['AP1','AP3','AP4','BK1','DD1','DGO2','DTLZ2','FA1','Far1','FDS','Hil1',
         'IKK1','KW2','Lov4','Lov5','Lov6','LTDZ','MGH16','MGH26','MGH33','MLF2',
         'MMR1','MMR2','MMR3','MOP5','MOP6','MOP7','NIC1','NIC2','NIC3','NIM1',
         'NSP1','NSP2','PNR','QV1','SK2','SP1','Toi4','Toi9','VU2','ZDT6','ZLT1']

algs = None
CR = {}   # CR[p][ai] = (crit, dnorm, info)   crit = -Theta_1, NaN if not certified
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
    c = np.asarray(gv(o, 'crit'), dtype=float)
    d = np.asarray(gv(o, 'dnorm'), dtype=float)
    info = np.asarray(gv(o, 'info'), dtype=float)
    try:                                   # present only in the R4 pass
        ok = np.asarray(gv(o, 'cert'), dtype=float) > 0
        c = np.where(ok, c, np.nan)
        d = np.where(ok, d, np.nan)
    except (KeyError, ValueError):
        pass
    CR[p] = (c, d, info)

nA = len(algs)
print('table  :', OUT)
print('algs   :', algs)

# ---- verdict-level adjudication of the uncertified points ------------------
# Same rule as r4_table4.py: at the points whose value the pass could not
# certify, the criterion is a statement about the point, so it is decided on
# the value the model attains at the returned point (adjudicated.csv, produced
# by make_adjudicated.py from retry_log.csv) and on the displacement there.
# A point marked 1 is counted as a pass; everything else, including the six
# points at which the subproblem was not solved at all, as a failure.
ADJ = {}
adj_csv = os.path.join(OUT, 'adjudicated.csv')
n_adj_th = n_adj_dn = 0
if os.path.exists(adj_csv):
    import csv as _csv
    ip = {n: i for i, n in enumerate(probs)}
    ia = {n: i for i, n in enumerate(algs)}
    for p in probs:
        ADJ[p] = (np.zeros((nA, 100), dtype=bool), np.zeros((nA, 100), dtype=bool))
    for r in _csv.DictReader(open(adj_csv)):
        pi, ai, ri = ip[r['prob']], ia[r['alg']], int(r['run']) - 1
        if np.isfinite(CR[r['prob']][0][ai, ri]):      # already certified: ignore
            continue
        ADJ[r['prob']][0][ai, ri] = r['pass_theta'] == '1'
        ADJ[r['prob']][1][ai, ri] = r['pass_dnorm'] == '1'
    n_adj_th = int(sum(v[0].sum() for v in ADJ.values()))
    n_adj_dn = int(sum(v[1].sum() for v in ADJ.values()))
    print('adjudicated : %d points decided from adjudicated.csv '
          '(Theta %d, disp %d)' % (n_adj_th + n_adj_dn, n_adj_th, n_adj_dn))
else:
    print('adjudicated : adjudicated.csv not found, uncertified points count as failures')

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
      % (len(succ_vals), np.median(succ_vals), np.quantile(succ_vals, .90),
         np.quantile(succ_vals, .99), succ_vals.max()))
print('  originally failed    : n=%d  min=%.2e  q01=%.2e  q10=%.2e  median=%.2e'
      % (len(fail_vals), fail_vals.min(), np.quantile(fail_vals, .01),
         np.quantile(fail_vals, .10), np.median(fail_vals)))


def sr_table(field, tol, mode):
    """mode 'A': the common criterion alone; 'B': flag and the criterion."""
    per_prob = np.full((len(probs), nA), np.nan)
    for pi, p in enumerate(probs):
        c, d, info = CR[p]
        v = c if field == 'crit' else d
        adj = ADJ[p][0 if field == 'crit' else 1] if p in ADJ else None
        for ai in range(nA):
            m = np.isfinite(v[ai]) & (v[ai] <= tol)
            if adj is not None:
                m = m | adj[ai]                      # decided point by point
            if mode == 'B':
                m = m & (info[ai] == 0)
            per_prob[pi, ai] = 100.0 * m.sum() / v[ai].size
    return per_prob


def flag_table():
    per_prob = np.full((len(probs), nA), np.nan)
    for pi, p in enumerate(probs):
        _, _, info = CR[p]
        for ai in range(nA):
            per_prob[pi, ai] = 100.0 * (info[ai] == 0).sum() / info[ai].size
    return per_prob


CASES = {'orig(flag)': None,
         'A:-Theta<=1e-4': ('crit', 1e-4, 'A'),
         'B:flag&-Theta<=1e-4': ('crit', 1e-4, 'B'),
         'A:-Theta<=1e-6': ('crit', 1e-6, 'A'),
         'A:||d1||<=1e-4': ('dnorm', 1e-4, 'A'),
         'B:flag&||d1||<=1e-4': ('dnorm', 1e-4, 'B')}

print('\n=== average success rate over the 42 problems ===')
res = {}
for name, spec in CASES.items():
    per_prob = flag_table() if spec is None else sr_table(*spec)
    avg = np.nanmean(per_prob, axis=0)
    rob = (per_prob >= 90).sum(axis=0) / len(probs) * 100.0
    res[name] = (avg, rob, per_prob)
    print('\n-- %s --' % name)
    for k in np.argsort(-avg):
        print('  %-13s avg=%5.1f%%  robustness(>=90%% on)=%4.1f%%' % (algs[k], avg[k], rob[k]))

ppA = res['A:-Theta<=1e-4'][2]
ppO = res['orig(flag)'][2]
print('\nmean absolute change (common - original) per algorithm:')
for ai in range(nA):
    print('  %-13s %+.2f pp' % (algs[ai], np.nanmean(ppA[:, ai] - ppO[:, ai])))

json.dump({'provenance': {
               'script': 'recompute_sr.py',
               'inputs': [os.path.basename(OUT.rstrip('/\\')) + '/<problem>_crit.mat',
                          os.path.basename(OUT.rstrip('/\\')) + '/adjudicated.csv'],
               'mask': 'a point with cert == 0 has no value and cannot count as a '
                       'success, except at the points adjudicated in adjudicated.csv',
               'adjudication': {
                   'rule': 'the criterion is decided on the value the model attains at '
                           'the returned point and on the displacement there',
                   'passes_criticality': n_adj_th,
                   'passes_displacement': n_adj_dn,
                   'not_passes_and_unsolved': 'counted as failures'},
               'criteria': 'beta = 1, Euclidean kernel; -Theta_1 <= 1e-4 and ||d_1|| <= 1e-4',
               'coincides_with': 'r4_table4.py (same rule, different layout)',
               'reported_in': 'Table 4 of the manuscript (R3)'},
           'algs': algs,
           'avg': {k: res[k][0].tolist() for k in res},
           'rob': {k: res[k][1].tolist() for k in res}},
          open(os.path.join(_BASE, 'common_sr.json'), 'w'), indent=1)
print('\nsaved common_sr.json')
