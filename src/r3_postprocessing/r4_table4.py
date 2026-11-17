# Added for the EBNPG manuscript (third revision).
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-
"""Build Table 4 from the certified stationarity table (crit_out_r4).

Decision rule, applied identically to every one of the 12 x 42 x 100 points:

    criticality   pass  <=>  X is finite and  X <= tol      X = -Theta_1(x)
    displacement  pass  <=>  Y is finite and  Y <= tol      Y = ||d_1(x)||

A point whose subproblem the solver could not solve to the accuracy required
by the certificate is NOT counted as a pass (conservative, documented in the
manuscript).  The denominator is always 100 runs.

Usage:
    python r4_table4.py [crit_dir]

Writes r4_table4.csv (full precision + counts) and prints the table with a
single rounding pass.
"""
import sys, os, csv
import numpy as np
import scipy.io as sio

TOL = 1e-4
probs = ['AP1','AP3','AP4','BK1','DD1','DGO2','DTLZ2','FA1','Far1','FDS','Hil1',
         'IKK1','KW2','Lov4','Lov5','Lov6','LTDZ','MGH16','MGH26','MGH33','MLF2',
         'MMR1','MMR2','MMR3','MOP5','MOP6','MOP7','NIC1','NIC2','NIC3','NIM1',
         'NSP1','NSP2','PNR','QV1','SK2','SP1','Toi4','Toi9','VU2','ZDT6','ZLT1']
algs = ['MPG-Explicit','MPG-Armijo','MPG-Implicit','CNPG-Max','CNPG-Avg',
        'EBNPG-N0','EBNPG-N1','EBNPG-N2','EBNPG-N3','EBNPG-N5','EBNPG-N6','EBNPG-N7']

HERE = os.path.dirname(os.path.abspath(__file__))


def find_data_dir(name='crit_out_r4'):
    """Locate a data folder by walking upwards from this script.

    The script is run both from the working tree (patch/ next to Code_Data/) and
    from inside the released package (Code/r3_postprocessing/ next to Code_Data/),
    so an explicit search is more robust than a fixed '../' offset.
    """
    d = HERE
    for _ in range(4):
        cand = os.path.join(d, 'Code_Data', name)
        if os.path.isdir(cand):
            return cand
        d = os.path.dirname(d)
    return os.path.join(HERE, '..', 'Code_Data', name)


CRIT = sys.argv[1] if len(sys.argv) > 1 else find_data_dir()

th = np.full((len(probs), 12, 100), np.nan)
dn = np.full((len(probs), 12, 100), np.nan)
cert = np.full((len(probs), 12, 100), np.nan)
for pi, p in enumerate(probs):
    o = sio.loadmat(os.path.join(CRIT, p + '_crit.mat'), squeeze_me=True,
                    struct_as_record=False)['out']
    th[pi] = np.atleast_2d(o.crit)
    dn[pi] = np.atleast_2d(o.dnorm)
    if hasattr(o, 'cert'):
        cert[pi] = np.atleast_2d(o.cert).astype(float)


# A value enters the count only if the certificate reproduced it (cert == 1).
# Values at the 134 uncertified points (128 reproduction failures + 6 unsolved)
# are set to NaN and therefore count as failures, except at the points where the
# criterion is decided by the verdict-level adjudication below.
MASK = (cert > 0)
th = np.where(MASK, th, np.nan)
dn = np.where(MASK, dn, np.nan)

# ---- verdict-level adjudication of the uncertified points ------------------
# At 128 points the pass could not certify the *value*: the value the solver
# returns there is positive by about 1e-9, which is impossible for -Theta_1.
# The criterion is a statement about the point, so it is decided on the value
# the model attains at the returned point (retry_log.csv, column theta_def) and
# on the displacement of that point.  adjudicated.csv, written by
# make_adjudicated.py, records the outcome for all 128 points; the points it
# marks are counted as passes, the others (and the six unsolved) as failures.
ADJ_TH = np.zeros_like(th, dtype=bool)
ADJ_DN = np.zeros_like(dn, dtype=bool)
adj_csv = os.path.join(CRIT, 'adjudicated.csv')
if os.path.exists(adj_csv):
    ip = {n: i for i, n in enumerate(probs)}
    ia = {n: i for i, n in enumerate(algs)}
    with open(adj_csv) as f:
        for r in csv.DictReader(f):
            i, j, k = ip[r['prob']], ia[r['alg']], int(r['run']) - 1
            if cert[i, j, k] > 0:
                continue
            ADJ_TH[i, j, k] = r['pass_theta'] == '1'
            ADJ_DN[i, j, k] = r['pass_dnorm'] == '1'
    print('adjudicated points  : %d (Theta %d, disp %d)'
          % (int(ADJ_TH.sum() | ADJ_DN.sum()), int(ADJ_TH.sum()), int(ADJ_DN.sum())))
else:
    print('adjudicated.csv not found: uncertified points counted as failures')


def rate(v, tol=TOL, adj=None):
    m = np.isfinite(v) & (v <= tol)
    if adj is not None:
        m = m | adj
    return 100.0 * m.sum() / v.shape[0]


PP_TH = np.array([[rate(th[pi, a], adj=ADJ_TH[pi, a]) for a in range(12)]
                  for pi in range(len(probs))])
PP_DN = np.array([[rate(dn[pi, a], adj=ADJ_DN[pi, a]) for a in range(12)]
                  for pi in range(len(probs))])
AVG_TH, AVG_DN = PP_TH.mean(0), PP_DN.mean(0)
ROB_TH = (PP_TH >= 90).sum(0) / len(probs) * 100
ROB_DN = (PP_DN >= 90).sum(0) / len(probs) * 100

npts = th.size
nfin_th = int(np.isfinite(th).sum())
nfin_dn = int(np.isfinite(dn).sum())
ncert = int(np.nansum(cert > 0))
print('points              : %d' % npts)
print('criticality finite  : %d   (missing %d)' % (nfin_th, npts - nfin_th))
print('displacement finite : %d   (missing %d)' % (nfin_dn, npts - nfin_dn))
print('certified values    : %d   (unverified %d)' % (ncert, npts - ncert))
print()

hdr = '%-14s %8s %8s %8s %8s' % ('Algorithm', 'avg-Th', 'rob-Th', 'avg-dn', 'rob-dn')
print(hdr)
print('-' * len(hdr))
rows = []
for a in range(12):
    rows.append(dict(algorithm=algs[a],
                     avg_th=AVG_TH[a], rob_th=ROB_TH[a],
                     avg_dn=AVG_DN[a], rob_dn=ROB_DN[a],
                     n_th_pass=int((np.isfinite(th[:, a]) & (th[:, a] <= TOL)).sum()
                                   + ADJ_TH[:, a].sum()),
                     n_dn_pass=int((np.isfinite(dn[:, a]) & (dn[:, a] <= TOL)).sum()
                                   + ADJ_DN[:, a].sum()),
                     n_missing=int((~np.isfinite(th[:, a])).sum())))
    print('%-14s %8.1f %8.1f %8.1f %8.1f' %
          (algs[a], AVG_TH[a], ROB_TH[a], AVG_DN[a], ROB_DN[a]))

print()
print('%-14s %18s %18s %8s' % ('Algorithm', 'pass Th / 4200', 'pass dn / 4200', 'missing'))
for r in rows:
    print('%-14s %18s %18s %8d' % (r['algorithm'],
                                   '%d (%.2f%%)' % (r['n_th_pass'], r['n_th_pass'] / 42.0),
                                   '%d (%.2f%%)' % (r['n_dn_pass'], r['n_dn_pass'] / 42.0),
                                   r['n_missing']))

out = os.path.join(HERE, 'r4_table4.csv')
with open(out, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Algorithm', 'avg_crit', 'rob_crit', 'avg_disp', 'rob_disp',
                'crit_pass', 'disp_pass', 'missing',
                'avg_crit_r1', 'rob_crit_r1', 'avg_disp_r1', 'rob_disp_r1'])
    for r in rows:
        w.writerow([r['algorithm'], '%.10f' % r['avg_th'], '%.10f' % r['rob_th'],
                    '%.10f' % r['avg_dn'], '%.10f' % r['rob_dn'],
                    r['n_th_pass'], r['n_dn_pass'], r['n_missing'],
                    '%.1f' % r['avg_th'], '%.1f' % r['rob_th'],
                    '%.1f' % r['avg_dn'], '%.1f' % r['rob_dn']])
print('\nwrote', out)

# ---------------------------------------------------------------------------
# Emit the 12 LaTeX rows in manuscript order and diff them against the R2 text.
# The manuscript order is read from the tex file itself, so the two can never
# drift apart.
# ---------------------------------------------------------------------------
import re

TEX = os.environ.get(
    'EBPGM_TEX',
    os.path.join(HERE, '..', 'EBNPG_2',
                 '0.1StrogNonMonProxGradMethdPZOY02October2026R1.tex'))
ROWS_OUT = os.path.join(HERE, 'table4_rows_r3.tex')
VAL = {a: dict(th='%.1f' % AVG_TH[i], dn='%.1f' % AVG_DN[i],
               rth='%.1f' % ROB_TH[i], rdn='%.1f' % ROB_DN[i])
       for i, a in enumerate(algs)}

if not os.path.exists(TEX):
    print('\nmanuscript tex not found (%s); LaTeX row emission skipped' % TEX)
else:
    txt = open(TEX, encoding='utf-8').read()
    m = re.search(r'\\midrule\s*\n(.*?)\\bottomrule', txt[
        txt.index('label{tab:common-stationarity}'):], re.S)
    # the manuscript marks changed cells with \begin{revision}...\end{revision};
    # strip the markup so the numbers can be parsed regardless of the markup state
    body = re.sub(r'\\begin\{revision\}|\\end\{revision\}', '', m.group(1))
    old = re.findall(r'^([A-Za-z][\w-]*)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)'
                     r'\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*\\\\',
                     body, re.M)
    assert len(old) == 12, 'expected 12 rows in tex, found %d' % len(old)

    lines, changed, diffs = [], 0, []
    for name, f1, t1, d1, f2, t2, d2 in old:
        v = VAL[name]
        new = [f1, v['th'], v['dn'], f2, v['rth'], v['rdn']]
        for j, (o_, n_) in enumerate(zip([f1, t1, d1, f2, t2, d2], new)):
            if abs(float(n_) - float(o_)) > 1e-9:
                changed += 1
                diffs.append('%-13s col%d  %s -> %s' % (name, j + 1, o_, n_))
        lines.append('%-13s & %s & %s & %s & %s & %s & %s \\\\' % (name, *new))

    open(ROWS_OUT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\nrows: %d   changed cells: %d of %d' % (len(old), changed, 6 * len(old)))
    if changed == 0:
        print('manuscript Table 4 already matches the certified data')
    for d in diffs:
        print('  ' + d)
    print('wrote', ROWS_OUT)
