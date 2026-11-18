# -*- coding: utf-8 -*-
"""Build Table 4 from the certified stationarity table (crit_out_r4).

Decision rule, applied identically to every one of the 12 x 42 x 100 points:

    criticality   pass  <=>  X is finite and  X <= tol      X = -Theta_1(x)
    displacement  pass  <=>  Y is finite and  Y <= tol      Y = ||d_1(x)||

The 134 points without a certified scalar value are decided from complete
primal-dual intervals through shared ``Code_Data/adjudication.py``.  A missing
file, incomplete coverage, or undecided interval raises an error.  The
denominator is always 100 returned points, irrespective of algorithm flags.
The CSV ``missing`` column counts uncertified scalar values, all of which have
adjudicated outcomes; it does not count undecided outcomes.

Usage:
    python r4_table4.py [crit_dir]

Writes r4_table4.csv (full precision + counts) and prints the table with a
single rounding pass.
"""
import sys, os, csv
import numpy as np

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


if len(sys.argv) > 1:
    CRIT = os.path.abspath(sys.argv[1])
elif os.environ.get('EBPGM_CRIT'):
    CRIT = os.path.join(os.environ.get('EBPGM_DATA', os.path.dirname(find_data_dir())),
                        os.environ['EBPGM_CRIT'])
else:
    CRIT = find_data_dir()
sys.path.insert(0, os.path.dirname(os.path.abspath(CRIT)))
from adjudication import CommonCriteria

COMMON = CommonCriteria(CRIT, probs, algs)
th, dn = COMMON.criticality, COMMON.displacement
MASK = COMMON.certified
PASS_TH, PASS_DN = COMMON.pass_mask('crit', TOL), COMMON.pass_mask('dnorm', TOL)
DIAGNOSTICS = COMMON.diagnostics()
print('adjudicated points  : %d unique (Theta %d pass, disp %d pass, undecided 0)'
      % (DIAGNOSTICS['adjudicated_points'], DIAGNOSTICS['passes_criticality'],
         DIAGNOSTICS['passes_displacement']))
PP_TH, PP_DN = COMMON.rates('crit', TOL), COMMON.rates('dnorm', TOL)
AVG_TH, AVG_DN = PP_TH.mean(0), PP_DN.mean(0)
ROB_TH = (PP_TH >= 90).sum(0) / len(probs) * 100
ROB_DN = (PP_DN >= 90).sum(0) / len(probs) * 100

npts = th.size
nfin_th = int(np.isfinite(th).sum())
nfin_dn = int(np.isfinite(dn).sum())
ncert = int(MASK.sum())
print('points              : %d' % npts)
print('criticality finite  : %d   (missing %d)' % (nfin_th, npts - nfin_th))
print('displacement finite : %d   (missing %d)' % (nfin_dn, npts - nfin_dn))
print('certified values    : %d   (adjudicated without scalar value %d)'
      % (ncert, npts - ncert))
print('undecided outcomes  : 0')
print()

hdr = '%-14s %8s %8s %8s %8s' % ('Algorithm', 'avg-Th', 'rob-Th', 'avg-dn', 'rob-dn')
print(hdr)
print('-' * len(hdr))
rows = []
for a in range(12):
    rows.append(dict(algorithm=algs[a],
                     avg_th=AVG_TH[a], rob_th=ROB_TH[a],
                     avg_dn=AVG_DN[a], rob_dn=ROB_DN[a],
                     n_th_pass=int(PASS_TH[:, a].sum()),
                     n_dn_pass=int(PASS_DN[:, a].sum()),
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
