# Added for the EBNPG manuscript (third revision).
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-
"""Cell-by-cell check of Table 4 (tab:common-stationarity) against common_sr.json.

The table has 12 rows x 6 numeric columns = 72 cells:
    avg   SR   : flag, -Theta_1, ||d_1||
    robustness : flag, -Theta_1, ||d_1||

The LaTeX is parsed rather than retyped, so a stale manuscript cell cannot be
excused by a stale constant in this script.  \\begin{revision} wrappers and
thousands separators are stripped.  Exit code 1 on any mismatch.

The data package has no LaTeX source, so inside it the script reports that and
exits successfully, as verify_manuscript.py does.
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def find_up(*names):
    """The nearest folder above this script that contains all the given names.

    The script sits in ``patch/`` in the working tree and in
    ``Code/r3_postprocessing/`` in the released package, so a fixed number of
    ``..`` steps cannot be right in both.
    """
    d = HERE
    while True:
        if all(os.path.exists(os.path.join(d, n)) for n in names):
            return d
        up = os.path.dirname(d)
        if up == d:
            return None
        d = up


ROOT = find_up('EBNPG_2', 'Code_Data') or find_up('Code_Data') or HERE
TEX = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, 'EBNPG_2', '0.1StrogNonMonProxGradMethdPZOY02October2026R1.tex')
SRJ = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    ROOT, 'Code_Data', 'common_sr.json')

if not os.path.exists(TEX):
    print('manuscript source not present at %s' % TEX)
    print('this data package does not include the LaTeX sources; nothing to verify.')
    sys.exit(0)

txt = open(TEX, encoding='utf-8').read()
i0 = txt.find(r'\label{tab:common-stationarity}')
i1 = txt.find(r'\end{tabular*}', i0)
body = txt[i0:i1]

ORDER = ['EBNPG-N2', 'CNPG-Avg', 'EBNPG-N3', 'EBNPG-N1', 'EBNPG-N6', 'EBNPG-N5',
         'CNPG-Max', 'MPG-Armijo', 'EBNPG-N0', 'EBNPG-N7', 'MPG-Explicit',
         'MPG-Implicit']

COLS = ['orig(flag)', 'A:-Theta<=1e-4', 'A:||d1||<=1e-4']
LAB = {'orig(flag)': 'flag', 'A:-Theta<=1e-4': r'-\Theta_1', 'A:||d1||<=1e-4': r'\|d_1\|'}


def clean(s):
    s = re.sub(r'\\begin\{revision\}', '', s)
    s = re.sub(r'\\end\{revision\}', '', s)
    s = s.replace('\\\\', ' ')                     # row terminator
    s = s.replace('\\%', '').replace('{', '').replace('}', '')
    s = s.replace('$', '').replace('\\ ', '')
    return s.strip()


rows = {}
for line in body.splitlines():
    if '&' not in line or '\\\\' not in line:
        continue
    parts = [clean(p) for p in line.split('&')]
    if len(parts) != 7 or parts[0] not in ORDER:
        continue
    rows[parts[0]] = parts[1:]

js = json.load(open(SRJ, encoding='utf-8'))
avg = js['avg']
rob = js['rob']
algs = js['algs']

bad = 0
print('%-13s %-9s %8s %8s %s' % ('algorithm', 'column', 'tex', 'json', ''))
print('-' * 46)
for a in ORDER:
    if a not in rows:
        print('%-13s  MISSING ROW IN TEX' % a)
        bad += 1
        continue
    k = algs.index(a)
    cells = rows[a]
    for j, key in enumerate(COLS):
        for off, tab, is_avg in ((0, avg, True), (3, rob, False)):
            texv = cells[j + off]
            jsv = tab[key][k]
            ok = abs(float(texv) - jsv) < 0.05 + 1e-9   # both printed to 0.1
            if not ok:
                bad += 1
            print('%-13s %-9s %8s %8.1f %s'
                  % (a if j == 0 and off == 0 else '',
                     ('avg ' if is_avg else 'rob ') + LAB[key],
                     texv, jsv, '' if ok else '<-- MISMATCH'))

print()
print('rows compared : %d of %d' % (len(rows), len(ORDER)))
print('cells compared: %d' % (len(ORDER) * 6))
print('mismatches    : %d' % bad)
sys.exit(1 if bad else 0)
