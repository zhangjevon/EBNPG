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

Input.  ``crit_out_r4/<prob>_crit.mat``, ``adjudicated.csv``, and its bound
``adjudication_witnesses.json``.  Shared ``adjudication.py`` requires one unique
adjudication for each of the 134 uncertified points.  It recomputes each verdict
from the interval at the requested tolerance; an undecided interval or missing
evidence stops the script.  Algorithm success flags do not affect criterion A.
``EBPGM_DATA`` selects the data root and ``EBPGM_CRIT`` selects an absolute
certificate directory or a directory relative to that data root.

The numbers of this file are the ones printed in Table 4 of the manuscript.
They coincide with ``r4_table4.py``; the two scripts differ only in layout.
Earlier versions of this package shipped a ``common_sr.json`` produced from
``crit_out`` without the certification mask, without the verdict-level
adjudication and without the displacement criterion; that file no longer
reproduces and has been replaced.
"""
import numpy as np, os, json
from adjudication import CommonCriteria

_BASE = os.environ.get('EBPGM_DATA') or os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_BASE, os.environ.get('EBPGM_CRIT', 'crit_out_r4'))
probs = ['AP1','AP3','AP4','BK1','DD1','DGO2','DTLZ2','FA1','Far1','FDS','Hil1',
         'IKK1','KW2','Lov4','Lov5','Lov6','LTDZ','MGH16','MGH26','MGH33','MLF2',
         'MMR1','MMR2','MMR3','MOP5','MOP6','MOP7','NIC1','NIC2','NIC3','NIM1',
         'NSP1','NSP2','PNR','QV1','SK2','SP1','Toi4','Toi9','VU2','ZDT6','ZLT1']

algs = ['MPG-Explicit','MPG-Armijo','MPG-Implicit','CNPG-Max','CNPG-Avg',
        'EBNPG-N0','EBNPG-N1','EBNPG-N2','EBNPG-N3','EBNPG-N5','EBNPG-N6','EBNPG-N7']
COMMON = CommonCriteria(OUT, probs, algs)
CR = {p: (COMMON.criticality[pi], COMMON.displacement[pi], COMMON.info[pi])
      for pi, p in enumerate(probs)}
nA = len(algs)
print('table  :', OUT)
print('algs   :', algs)
DIAGNOSTICS = COMMON.diagnostics()
n_adj_th = DIAGNOSTICS['passes_criticality']
n_adj_dn = DIAGNOSTICS['passes_displacement']
print('adjudicated : %d unique points (Theta %d pass, disp %d pass, undecided 0)'
      % (DIAGNOSTICS['adjudicated_points'], n_adj_th, n_adj_dn))

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
print('  retained scalar evaluations only; adjudicated intervals excluded from quantiles')
print('  originally successful: n=%d  median=%.2e  q90=%.2e  q99=%.2e  max=%.2e'
      % (len(succ_vals), np.median(succ_vals), np.quantile(succ_vals, .90),
         np.quantile(succ_vals, .99), succ_vals.max()))
print('  originally failed    : n=%d  min=%.2e  q01=%.2e  q10=%.2e  median=%.2e'
      % (len(fail_vals), fail_vals.min(), np.quantile(fail_vals, .01),
         np.quantile(fail_vals, .10), np.median(fail_vals)))


def sr_table(field, tol, mode):
    """mode 'A': the common criterion alone; 'B': flag and the criterion."""
    return COMMON.rates(field, tol, mode)


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
                          os.path.basename(OUT.rstrip('/\\')) + '/adjudicated.csv',
                          os.path.basename(OUT.rstrip('/\\')) + '/adjudication_witnesses.json'],
               'mask': 'certified values plus interval decisions for every uncertified '
                       'point; missing or undecided evidence raises an error',
               'adjudication': DIAGNOSTICS,
               'distribution_coverage': {
                   'summary_basis': 'retained scalar evaluations only; no interval point estimates',
                   'originally_successful_runs': int((COMMON.info == 0).sum()),
                   'successful_retained_scalar_values': int(len(succ_vals)),
                   'successful_adjudicated_intervals': int(((COMMON.info == 0) & ~COMMON.certified).sum()),
                   'originally_failed_runs': int((COMMON.info != 0).sum()),
                   'failed_retained_scalar_values': int(len(fail_vals)),
                   'failed_adjudicated_intervals': int(((COMMON.info != 0) & ~COMMON.certified).sum()),
               },
               'criteria': 'beta = 1, Euclidean kernel; -Theta_1 <= 1e-4 and ||d_1|| <= 1e-4',
               'coincides_with': 'r4_table4.py (same rule, different layout)',
               'reported_in': 'Table 4 of the manuscript (R3)'},
           'algs': algs,
           'avg': {k: res[k][0].tolist() for k in res},
           'rob': {k: res[k][1].tolist() for k in res}},
          open(os.path.join(_BASE, 'common_sr.json'), 'w'), indent=1)
print('\nsaved common_sr.json')
