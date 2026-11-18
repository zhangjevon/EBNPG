# -*- coding: utf-8 -*-
"""Data-driven verification of the manuscript tables.

Parses every numerical table of the tex file, strips the blue revision markup and
compares each cell with the value recomputed by aggregate_all.py (r2_tables.json).
Then asserts a list of prose claims and performs structural checks.

Usage: python verify_manuscript.py --tex /path/to/manuscript.tex --data /path/to/Code_Data
Missing source or result files are errors. All existing numerical-table assertions
are retained. Prose checks combine stable wording assertions with parsed numeric
claims checked against the final summaries and per-point certificate record.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys
from decimal import Decimal, ROUND_CEILING

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tex', type=Path,
                    default=HERE.parent / 'EBNPG_2' /
                    '0.1StrogNonMonProxGradMethdPZOY02October2026R1.tex',
                    help='explicit manuscript LaTeX source')
parser.add_argument('--data', type=Path, default=HERE,
                    help='Code_Data folder holding r2_tables.json and common_sr.json')
args = parser.parse_args()
TEX = args.tex.expanduser().resolve()
DATA = args.data.expanduser().resolve()
for required in [TEX, DATA / 'r2_tables.json', DATA / 'common_sr.json']:
    if not required.is_file():
        parser.error('required input does not exist: %s' % required)
with (DATA / 'r2_tables.json').open(encoding='utf-8') as stream:
    D = json.load(stream)
with (DATA / 'common_sr.json').open(encoding='utf-8') as stream:
    COMMON_SUMMARY = json.load(stream)
ALGS = D['algs']
IX = {a: i for i, a in enumerate(ALGS)}
txt = TEX.read_text(encoding='utf-8-sig')


def strip_rev(s):
    """Drop the blue revision markup and collapse whitespace.

    The prose checks below must be about the content of the manuscript, not
    about where a \begin{revision} happened to be inserted, nor about where a
    line was broken.  Both the text and every claim are normalised this way.
    """
    s = re.sub(r'\\begin\{revision\}|\\end\{revision\}', '', s)
    return re.sub(r'\s+', ' ', s)


TXT_N = strip_rev(txt)

FAIL = []
OK = 0


def check(cond, msg):
    global OK
    if cond:
        OK += 1
    else:
        FAIL.append(msg)


# ----------------------------------------------------------------- helpers
def mround(x, nd=0):
    """MATLAB round: halves away from zero."""
    f = 10.0 ** nd
    import math
    return math.copysign(math.floor(abs(x) * f + 0.5), x) / f


def he(x, nd):
    """C printf rounding: halves to even."""
    return float('%%.%df' % nd % x)


def table_rows(caption_label):
    r"""Return the rows of the tabular* belonging to \label{caption_label}."""
    i = txt.index('\\label{%s}' % caption_label)
    j = txt.index('\\begin{tabular*}', i)
    k = txt.index('\\bottomrule', j)
    body = txt[j:k]
    rows = []
    for line in body.split('\n'):
        s = line.strip()
        if not s.endswith('\\\\'):
            continue
        if '\\midrule' in s or '\\toprule' in s or '\\cmidrule' in s:
            continue
        if s.startswith('&') or s.startswith('Class') or s.startswith('Algorithm') \
           or s.startswith('Rank') or s.startswith('Opponent') or s.startswith('Problem'):
            continue
        s = s[:-2].rstrip()                      # drop \\
        cells = [c.strip() for c in s.split('&')]
        cells = [re.sub(r'\\begin\{revision\}(.*?)\\end\{revision\}', r'\1', c).strip()
                 for c in cells]
        rows.append(cells)
    return rows


def num(c):
    """First number in a cell; skips the argument of an enclosing \\multicolumn."""
    m = re.match(r'\\multicolumn\{\d+\}\{[^}]*\}\{(.*)\}$', c.strip())
    if m:
        c = m.group(1)
    m = re.search(r'-?\d+(?:\.\d+)?', c)
    return float(m.group()) if m else None


# ------------------------------------------------- Table 2 (problem class)
exp = {r[0]: r for r in D['tab2'] if r[0] != 'Overall'}
rows = table_rows('tab:problem-class')
check(len(rows) == 5, 'Table 2 has %d rows (expected 5)' % len(rows))
for r in rows:
    cls = r[0].split('(')[0].strip()
    e = exp[cls]
    w = []
    for v in [e[1], e[2], e[3]]:
        p = v.split('--')
        w += [num(p[0]), num(p[0]) if len(p) == 1 else num(p[1])]
    got = []
    for c in r[1:4]:
        p = c.split('--')
        got += [num(p[0]), num(p[0]) if len(p) == 1 else num(p[1])]
    for g, x in zip(got, w):
        check(g == x, 'Table 2 %s: %s vs %s' % (cls, got, w))

# ---------------------------------------- Table 3 (efficiency/robustness)
rows = table_rows('tab:efficiency-robustness')
check(len(rows) == 12, 'Table 3 has %d rows (expected 12)' % len(rows))
for r in rows:
    a = r[1].strip()
    i = IX[a]
    # Avg SR (1 dp), Robustness (1 dp), Best count (0 dp), MedIter (1 dp), MedTime (2 dp)
    for c, v, nd in zip(r[2:], [D['avg'][i], D['rob'][i], D['best'][i],
                                D['medit'][i], D['medtime'][i]], [1, 1, 0, 1, 2]):
        g = num(c)
        w = he(v, nd)
        check(abs(g - w) < 1e-9,
              'Table 3 %s: %s expected %s (raw %s)' % (a, g, w, v))

# --------------------------------------- Table 4 (common stationarity)
rows = table_rows('tab:common-stationarity')
check(len(rows) == 12, 'Table 4 has %d rows (expected 12)' % len(rows))
for r in rows:
    a = r[0].strip(); i = IX[a]
    for c, v in zip(r[1:], [D['avg'][i], D['avg_th'][i], D['avg_dn'][i],
                            D['rob'][i], D['rob_th'][i], D['rob_dn'][i]]):
        g = num(c); w = he(v, 1)
        check(abs(g - w) < 1e-9, 'Table 4 %s: %s expected %s (raw %s)' % (a, g, w, v))

# ------------------------------- Table 5 (EBNPG control configurations)
rows = table_rows('tab:nonmonotone-comparison')
check(len(rows) == 5, 'Table 5 has %d rows (expected 5)' % len(rows))
for r in rows:
    cls = r[0].split('(')[0].strip()
    for a, c in zip(['EBNPG-N0', 'EBNPG-N1', 'EBNPG-N2', 'EBNPG-N3',
                     'EBNPG-N5', 'EBNPG-N6', 'EBNPG-N7'], r[1:]):
        w = mround(D['extra']['clsmean'][cls][a])
        check(num(c) == w, 'Table 5 %s/%s: %s expected %s' % (cls, a, num(c), w))

# --------------------------------------------- Table 6 (paired tests)
rows = table_rows('tab:paired-tests')
check(len(rows) == 11, 'Table 6 has %d rows (expected 11)' % len(rows))
for r in rows:
    a = r[0].strip()
    d = [x for x in D['cmh'] if x['alg'] == a][0]
    g = num(r[1]); w = he(d['mean_d'], 2)
    check(abs(g - w) < 1e-9, 'Table 6 %s meanD: %s expected %s' % (a, g, w))
    bwt = '%d/%d/%d' % (d['b'], d['w'], d['t'])
    check(re.sub(r'\s+', '', r[2]) == bwt,
          'Table 6 %s b/w/t: %s expected %s' % (a, r[2], bwt))
    if abs(d['chi2']) < 1e100:
        g = num(r[3]); w = he(d['chi2'], 2)
        check(abs(g - w) < 1e-9, 'Table 6 %s chi2: %s expected %s' % (a, g, w))
    if '<' in r[5]:                       # reported as $<10^{-3}$
        check(d['p_w'] < 1e-3, 'Table 6 %s p_W printed as <1e-3 but is %s' % (a, d['p_w']))
    else:
        g = num(r[5]); w = he(d['p_w'], 2)
        check(abs(g - w) < 1e-9, 'Table 6 %s p_W: %s expected %s' % (a, g, w))

# ------------------------------------------- Table 8 (weighted oracle cost)
rows = table_rows('tab:oracle-cost')
check(len(rows) == 12, 'Table 8 has %d rows (expected 12)' % len(rows))
for r in rows:
    a = r[0].strip()
    ng, nh = D['perit'][a]
    vals = [he(ng, 2), he(nh, 2),
            D['oracle']['1']['wins'][IX[a]], D['oracle']['5']['wins'][IX[a]],
            D['oracle']['20']['wins'][IX[a]], he(D['oracle']['20']['medrel'][IX[a]], 2)]
    for c, w in zip(r[1:], vals):
        g = num(c)
        check(abs(g - w) < 1e-9, 'Table 8 %s: %s expected %s' % (a, g, w))

# ---------------------------------- Table 7 (QP subproblem solves)
rows = table_rows('tab:qpsolve')
check(len(rows) == 12, 'Table 7 has %d rows (expected 12)' % len(rows))
for row, d in zip(rows, D['qp']):
    a = row[0].strip()
    check(a == d['alg'], 'Table 7 algorithm order: %s vs %s' % (a, d['alg']))
    # medians are printed with 1 dp (halves to even), the ratio with 2 dp
    for c, v, nd in zip(row[1:], [d['qp'], d['it'], d['ratio']], [1, 1, 2]):
        g = num(c); w = he(v, nd)
        check(abs(g - w) < 1e-9, 'Table 7 %s: %s expected %s' % (a, g, w))

# ---------------------------------- Table 9 (Euclidean vs adaptive)
rows = table_rows('tab:weighted-summary')
check(len(rows) == 6, 'Table 9 has %d rows (expected 6)' % len(rows))
for row, e in zip(rows, D['tab9']):
    cls = row[0].split('(')[0].strip()
    check(e['cls'] in cls, 'Table 9 class: %s vs %s' % (cls, e['cls']))
    check(('(%d)' % e['n']) in row[0], 'Table 9 %s count: %s' % (cls, row[0]))
    for c, v in zip(row[1:], [e['e_best'], e['e_med'], e['w_best'], e['w_med'], e['d']]):
        g = num(c); w = he(v, 1)
        check(abs(g - w) < 1e-9, 'Table 9 %s: %s expected %s' % (cls, g, w))

# ----------------------------------------- Table 10 (fixed-kernel study)
rows = table_rows('tab:fixed-kernel')
nrow = len(rows)
check(nrow == 22, 'Table 10 has %d rows (expected 21 + summary)' % nrow)
by_p = {x['p']: x for x in D['krows']}
for r in rows[:-1]:
    p = r[0].strip()
    d = by_p[p]
    for c, w in zip(r[1:4], [d['sr0'], d['sr1'], d['sr1'] - d['sr0']]):
        g = num(c)
        check(abs(g - he(w, 1)) < 1e-9,
              'Table 10 %s SR: %s expected %s' % (p, g, he(w, 1)))
    for c, w in zip(r[4:6], [d['mi0'], d['mi1']]):
        if c.strip() == '---':
            continue
        g = num(c)
        check(abs(g - he(w, 1)) < 1e-9, 'Table 10 %s medIt: %s expected %s' % (p, g, he(w, 1)))
ks = D['kstat']
sr = rows[-1]
check('42' in sr[0], 'Table 10 summary row does not say 42 problems')
for c, w in zip([sr[1], sr[2], sr[3]], [ks['diff'], ks['med0'], ks['med1']]):
    g = num(c)
    check(abs(g - he(w, 2)) < 1e-9, 'Table 10 summary: %s expected %s' % (g, he(w, 2)))
check(abs(num(sr[4]) - he(ks['pool_p'], 2)) < 1e-9,
      'Table 10 summary p: %s expected %s' % (num(sr[4]), he(ks['pool_p'], 2)))

# ---------------------------------------------------- prose assertions
# The release identifiers are detected first, so that the prose claims that quote
# them do not hard-code a tag: a hard-coded tag turns the checker into a failure
# the moment the package is re-released, which is exactly what happened when the
# data package moved from v3.0 to v3.1.
_tags = sorted(set(re.findall(r'release\s+\\texttt\{([^}]*)\}', txt)))
TAG = _tags[0] if len(_tags) == 1 else '<AMBIGUOUS-TAG:%s>' % _tags
_shas = sorted(set(re.findall(r'\\texttt\{([0-9a-f]{7,40})\}', txt)))
SHA = _shas[0] if len(_shas) == 1 else '<AMBIGUOUS-SHA:%s>' % _shas
_dois = sorted(set(re.findall(r'10\.5281/zenodo\.\d+', txt)))
DOI = _dois[0] if len(_dois) == 1 else '<AMBIGUOUS-DOI:%s>' % _dois

CLAIMS = [
    # 6.3
    'On convex problems, \\begin{revision}EBNPG-N6 leads at 91\\%, with CNPG-Avg, EBNPG-N1 and EBNPG-N2 at 90\\%\\end{revision}',
    'On nonconvex problems, \\begin{revision}EBNPG-N2 leads at 95\\% and N3 at 94\\%\\end{revision}',
    '(MPG-Armijo: \\begin{revision}90\\end{revision}\\%)',
    'with the CNPG variants at \\begin{revision}91--97\\end{revision}\\% and the EBNPG variants at \\begin{revision}86--96\\end{revision}\\%',
    # 6.5
    'EBNPG-N2 achieves the highest average success rate at \\begin{revision}93.6\\end{revision}\\%, followed by EBNPG-N6 (\\begin{revision}93.4\\end{revision}\\%), EBNPG-N3 (\\begin{revision}93.2\\end{revision}\\%), EBNPG-N1 (\\begin{revision}93.1\\end{revision}\\%) and CNPG-Avg (\\begin{revision}92.8\\end{revision}\\%)',
    '\\begin{revision}EBNPG-N6 alone achieves 85.7\\%, meaning that it achieves a success rate of at least \\(90\\%\\) on 36 out of 42 problems, followed by EBNPG-N2, EBNPG-N1, CNPG-Avg, and EBNPG-N5 at 83.3\\%\\end{revision}',
    'and \\begin{revision}7.1\\end{revision} percentage points between the weakest (\\begin{revision}71.4\\% for EBNPG-N7 and EBNPG-N0\\end{revision}',
    'EBNPG-N2 attains the highest average success rate under both common criteria',
    'For MPG-Explicit, that joint average is $85.6\\%$, compared with its flag-based $89.1\\%$',
    # 6.6
    'N2 achieves the highest overall average (\\begin{revision}93.6\\end{revision}\\%), followed by N6 (\\begin{revision}93.4\\end{revision}\\%) and N3 (\\begin{revision}93.2\\end{revision}\\%), with N1 at \\begin{revision}93.1\\end{revision}\\%',
    'On quadratic problems, CNPG-Max and MPG-Implicit achieve 97\\%, with \\begin{revision}EBNPG-N3 and N5 (both 96\\%)\\end{revision} close behind',
    'its lower robustness (\\begin{revision}71.4\\end{revision}\\%)',
    # 6.7
    '\\begin{revision}Third, the robustness of the EBNPG/CNPG family (71--86\\%) lies above that of the MPG baselines (64--76\\%).\\end{revision}',
    'whose adjusted $p$-values lie between $\\begin{revision}0.22\\end{revision}$ and $\\begin{revision}0.67\\end{revision}$',
    'the $p$-value for EBNPG-N7 being now $0.09$ (it was $0.03$ before the MGH33 rerun)',
    # 6.8
    'N0 ($\\begin{revision}24.2\\end{revision}$, $\\begin{revision}0.54\\end{revision}$ s) is better than N5 ($\\begin{revision}28.2\\end{revision}$',
    # 6.9--6.10
    'The EBNPG variants show relatively consistent iteration counts',
    'the clearest reduction in dispersion is relative to MPG-Implicit',
    'For MPG-Explicit this statistic ranges from $1.01$ to $1.51$ over the $42$ problems',
    'Table~\\ref{tab:oracle-cost} reports the resulting weighted-cost comparisons at $w_h/w_g=1,5,20$',
    'For general-purpose use, EBNPG-N2 is a reasonable default',
    'The nonmonotone variants improve average success and robustness over N0 overall',
    'N0 remains faster than N5 and N7 and is competitive or better on several problem classes',
    'N2 offers a useful overall balance, N6 attains the highest robustness, and CNPG-Avg remains competitive',
    'Overall robustness ranges from $71.4\\%$ to $85.7\\%$ for EBNPG, from $81.0\\%$ to $83.3\\%$ for CNPG, and from $64.3\\%$ to $76.2\\%$ for MPG',
    # 6.2 (P1/P2 disclosure)
    'Interrupted sessions shifted this assignment on four problems',
    'on AP4, BK1, and PNR, algorithm 2 reuses block 1 and each later algorithm moves one block earlier; on DGO2, algorithms 4--12 use blocks 1--9',
    'two algorithms share their starting points on each of AP4, BK1, and PNR, and three pairs share theirs on DGO2',
    'the released code reproduces the complete assignment exactly',
    'reconstructed by rerunning the released code, which regenerates the starting-point assignment of every problem from the stated stream together with the block map',
    # 6.10 (P5)
    'CNPG-Avg is a competitive and simply implemented alternative, but it is not a lower-cost alternative to EBNPG-N2 on this test set',
    # 6.11 (P6/P9)
    'with degradations confined to a minority of the problems ($6$ of $42$) and to at most $19$ percentage points (DD1, on which the Euclidean EBNPG-N2 already attains $100\\%$)',
    'generated with the same rule as in the principal experiments (seed $123$, one block of $100$ draws per variant from the same stream)',
    # 6.11
    'with the best-variant SR increasing by \\begin{revision}\\(5.1\\) percentage points\\end{revision}',
    '(\\begin{revision}N0 by \\(4.0\\) percentage points, N2 by \\(3.0\\) percentage points, N3 by \\(2.3\\) percentage points\\end{revision})',
    'the max-type variant N5 changes by \\begin{revision}\\(-0.7\\) percentage points\\end{revision}',
    '\\begin{revision}Beyond NIC1, the adaptive kernel raises the best-variant success rate on eight further problems---MMR3 (92\\% to 100\\%), MGH26 (95\\% to 99\\%), PNR (97\\% to 100\\%), AP3 (65\\% to 67\\%), MGH33 (best variant, $91\\%$ to $93\\%$), MMR1 (96\\% to 98\\%), Far1 (99\\% to 100\\%) and LTDZ (99\\% to 100\\%)---so that 9 of the 42 problems improve in total.',
    'On 26 of the 42 problems, both kernels achieve 100\\% best-variant success rate, and on ZLT1 both reach 95\\%.',
    'On the remaining 6 problems, the adaptive kernel lowers the best-variant success rate by 1--19 percentage points, the largest drop occurring on DD1 (from 100\\% to 81\\%), followed by Lov6 (from 96\\% to 89\\%).\\end{revision}',
    'the median ratio between the largest and the smallest weight is $1.0$ on BK1 and $1.1$ on DD1',
    'the same median is $83$ and $45$.',
    'The best-variant degradations on the affected problems are nevertheless confined to at most 19 percentage points',
    'the ill-conditioned problem NIC1, followed by MMR3 and the least-squares problems MGH26 and MGH33,',
    'With the adaptive BB-weighted kernel, five of the seven variants \\begin{revision}achieve a success rate of \\(100\\%\\)\\end{revision} (N0--N3, N6), N7 reaches 98\\%, and the remaining variant (N5) improves from 1\\% to 33\\%.',
    # 6.12
    'the average success rate is \\begin{revision}$93.64\\%$\\end{revision} under the Euclidean kernel and \\begin{revision}$94.02\\%$\\end{revision} under the fixed diagonal kernel, a difference of \\begin{revision}$+0.38$\\end{revision} percentage points',
    'the pooled McNemar test gives \\begin{revision}$p=0.27$\\end{revision} (\\begin{revision}$100$\\end{revision} pairs solved only by the diagonal kernel against \\begin{revision}$84$\\end{revision}',
    'gives \\begin{revision}$p=0.57$\\end{revision}',
    'The fixed kernel is better on \\begin{revision}$12$\\end{revision} problems, worse on $9$ and ties on $21$; the median iteration count over the \\begin{revision}$41$\\end{revision} problems on which both kernels record at least one successful run is $24.5$ for the Euclidean kernel and \\begin{revision}$25.0$\\end{revision} for the diagonal one',
    '\\begin{revision}MGH33 from $81\\%$ to $91\\%,$ \\end{revision}IKK1 from $94\\%$ to $100\\%$ and MMR3 from $87\\%$ to $93\\%$; these \\begin{revision}five\\end{revision} are the problems',
    'and \\begin{revision}$+0.72$ on the $18$\\end{revision} problems with ratio above $3$; none of these three subgroups is significant on its own ($p=0.22$, $0.17$ and \\begin{revision}$0.23$\\end{revision} respectively)',
    'Table~\\ref{tab:fixed-kernel} reports \\begin{revision}$21$\\end{revision} tests side by side',
    # conclusion (shortened in the final pass)
    'with EBNPG-N2 highest at $93.6\\%$; its advantage over the three MPG baselines is significant',
    'The same families remain in the leading positions under the common criticality and displacement criteria',
    'so the case for the method rests on reliability rather than on oracle cost',
    'gives no statistically significant average gain ($+0.38$ percentage points over the $42$ problems, $p=0.27$)',
    'substantially improves some ill-conditioned instances (NIC1: $11\\%$ to $100\\%$)',
    # data availability: public code regenerates everything, nothing withheld
    'every generated quantity (the reference points $x^{\\rm ref}$, the matrices $B_i$, the thresholds $\\delta$, the starting points and the fixed-kernel weights) is fully regenerated by the publicly available code',
    'No problem, run or data set is excluded from the experiments reported in this paper',
    'The data package deposited with release \\texttt{%s} contains the per-problem and per-run results' % TAG,
    # code availability: public repository cited in Code availability and the reply letter
    'is publicly available at \\url{https://github.com/zhangjevon/EBNPG}, release \\texttt{%s}' % TAG,
]
for c in CLAIMS:
    n = TXT_N.count(strip_rev(c))
    check(n == 1, 'claim occurs %d times: %s' % (n, c[:70]))

# ---------------------------------------- data-driven prose and coverage checks
# Keep wording assertions above for unchanged passages. For the rewritten
# passages, parse the relevant fields inside the named section instead of
# asserting an entire paragraph with a historical sentence structure.
def section(label, end_marker=None):
    marker = r'\label{' + label + '}'
    start = txt.index(marker) + len(marker)
    if end_marker is None:
        following = re.search(r'\\(?:subsection|section)\b', txt[start:])
        end = start + following.start() if following else len(txt)
    else:
        end = txt.index(end_marker, start)
    return strip_rev(txt[start:end])


def readable(source):
    source = strip_rev(source)
    source = re.sub(r'\\begin\{table\}.*?\\end\{table\}', ' ', source, flags=re.S)
    source = source.replace(r'\%', '%').replace('$', '').replace('{,}', ',')
    source = source.replace(r'\(', '').replace(r'\)', '')
    return re.sub(r'\s+', ' ', source)


def parse_numbers(source, expression, expected, description, decimals=None):
    found = re.search(expression, source)
    check(found is not None, '%s: numeric claim is missing' % description)
    if found is None:
        return
    values = [float(value.replace(',', '')) for value in found.groups()]
    check(len(values) == len(expected), '%s: wrong numeric field count' % description)
    for value, target in zip(values, expected):
        target = he(target, decimals) if decimals is not None else target
        check(math.isclose(value, target, rel_tol=0.0, abs_tol=1e-9),
              '%s: read %s, expected %s' % (description, value, target))


def ranks(values):
    return [1 + sum(other < value for other in values) +
            (sum(other == value for other in values) - 1) / 2
            for value in values]


classes = D['extra']['clscount']
problem_rows = {}
for line in txt.splitlines():
    clean = strip_rev(line).strip()
    parts = [part.strip() for part in clean.split('&')]
    if len(parts) == 7 and parts[0] in D['CLASS']:
        problem_rows[parts[0]] = parts[1]
check(set(problem_rows) == set(D['CLASS']), 'Table 1 problem coverage differs from CLASS')
for problem, expected_class in D['CLASS'].items():
    check(problem_rows.get(problem) == expected_class,
          'Table 1 %s classification: %s vs %s' %
          (problem, problem_rows.get(problem), expected_class))
check(D['CLASS'].get('DTLZ2') == 'Nonconvex', 'DTLZ2 must belong to the nonconvex group')
for label in ['tab:problem-class', 'tab:nonmonotone-comparison']:
    for row in table_rows(label):
        match = re.fullmatch(r'(.+?)\s*\((\d+)\)', row[0])
        check(match is not None, '%s: class count missing in %s' % (label, row[0]))
        if match:
            check(int(match.group(2)) == classes.get(match.group(1).strip()),
                  '%s: wrong count for %s' % (label, row[0]))

common_text = section('subsec:efficiency')
common_text = common_text[common_text.index(r'\paragraph*{Sensitivity'):]
common_plain = readable(common_text)
parse_numbers(common_plain,
              r'first pass left ([\d,]+) points.*?produced values at ([\d,]+) of them, including '
              r'([\d,]+) that satisfy the model-value consistency test',
              [607, 601, 473], 'Section 6.5 initial recovery and consistency coverage')
check('with retained scalar common-subproblem evaluations' in common_plain,
      'scalar quantile scope must be explicit')
check('supporting-vector identity establishes the minimiser' in common_plain,
      'the analytical origin certificate must be described')
check(r'objective $i$ attains the maximum in $\Phi_x(-x)$' in common_text,
      'the origin certificate must require objective activity at d=-x')
check(r'$\|z_i\|_\infty\le1$' in common_text,
      'the origin certificate must require supporting-vector feasibility')

counts_match = re.search(r'Of them, ([\d,]+) satisfy both.*?remaining ([\d,]+) fail both',
                         common_plain)
check(counts_match is not None, 'Section 6.5 must report the final pass/fail partition')
if counts_match:
    check(tuple(map(lambda value: int(value.replace(',', '')), counts_match.groups())) == (31, 103),
          'Section 6.5: final additional-check partition must be 31/103')

n2 = IX['EBNPG-N2']
parse_numbers(common_plain,
              r'EBNPG-N2 attains the highest average success rate under both common criteria, '
              r'([+-]?\d+(?:\.\d+)?)% for criticality and ([+-]?\d+(?:\.\d+)?)% for displacement',
              [D['avg_th'][n2], D['avg_dn'][n2]], 'Section 6.5 N2 averages', 1)
check(D['avg_th'][n2] == max(D['avg_th']), 'N2 is not the highest common-criticality average')
check(D['avg_dn'][n2] == max(D['avg_dn']), 'N2 is not the highest common-displacement average')
parse_numbers(common_plain,
              r'For MPG-Explicit, that joint average is ([+-]?\d+(?:\.\d+)?)%, compared with its flag-based ([+-]?\d+(?:\.\d+)?)%',
              [D['avg_th_b'][IX['MPG-Explicit']], D['avg'][IX['MPG-Explicit']]],
              'Section 6.5 joint flag/criticality comparison', 1)

corr_flag_dn = statistics.correlation(ranks(D['avg']), ranks(D['avg_dn']))
corr_th_dn = statistics.correlation(ranks(D['avg_th']), ranks(D['avg_dn']))
parse_numbers(common_plain, r'their Spearman rank correlation is ([+-]?\d+(?:\.\d+)?)',
              [corr_flag_dn], 'Section 6.5 flag/displacement correlation', 3)
parse_numbers(common_plain, r'rank correlation between the common columns is ([+-]?\d+(?:\.\d+)?)',
              [corr_th_dn], 'Section 6.5 correlation of common criteria', 2)
parse_numbers(common_plain,
              r'CNPG-Max and EBNPG-N5.*?displacement averages are ([+-]?\d+(?:\.\d+)?)% and ([+-]?\d+(?:\.\d+)?)%',
              [D['avg_dn'][IX['CNPG-Max']], D['avg_dn'][IX['EBNPG-N5']]],
              'Section 6.5 displayed unresolved mean tie', 3)
check(D['avg'][IX['CNPG-Max']] == D['avg'][IX['EBNPG-N5']],
      'CNPG-Max and N5 must have equal flag-based averages for the tie claim')
parse_numbers(common_plain,
              r'In the displacement robustness column, CNPG-Avg reaches ([+-]?\d+(?:\.\d+)?)% and joins EBNPG-N6',
              [D['rob_dn'][IX['CNPG-Avg']]], 'Section 6.5 CNPG-Avg robustness', 1)
check(D['rob_dn'][IX['CNPG-Avg']] == D['rob_dn'][IX['EBNPG-N6']] == max(D['rob_dn']),
      'CNPG-Avg and N6 must share the highest common-displacement robustness')
max_change = max(abs(a - b) for a, b in zip(D['avg'], D['avg_dn']))
limit_match = re.search(r'differ from the flag-based averages by at most ([+-]?\d+(?:\.\d+)?) percentage point',
                        common_plain)
check(limit_match is not None, 'Section 6.5 flag/displacement difference bound is missing')
if limit_match:
    check(max_change <= float(limit_match.group(1)) + 1e-12,
          'Section 6.5 flag/displacement difference exceeds printed bound')

# Compare the tolerance sensitivity with the numerical summary rather than
# retaining a hard-coded claim about a former revision's threshold behavior.
summary_algs = COMMON_SUMMARY.get('algs', [])
check(summary_algs == ALGS, 'common_sr.json algorithm order differs from r2_tables.json')
try:
    avg_4 = COMMON_SUMMARY['avg']['A:-Theta<=1e-4']
    avg_6 = COMMON_SUMMARY['avg']['A:-Theta<=1e-6']
    drops = [a - b for a, b in zip(avg_4, avg_6)]
    other = [drops[i] for i, alg in enumerate(ALGS) if alg != 'MPG-Explicit']
    parse_numbers(common_plain,
                  r'lowers the average success rate by ([+-]?\d+(?:\.\d+)?)--([+-]?\d+(?:\.\d+)?) percentage points for eleven algorithms',
                  [min(other), max(other)], 'Section 6.5 criticality sensitivity range', 1)
    parse_numbers(common_plain, r'and by ([+-]?\d+(?:\.\d+)?) points for MPG-Explicit',
                  [drops[IX['MPG-Explicit']]], 'Section 6.5 MPG-Explicit sensitivity', 1)
except KeyError as exc:
    check(False, 'common_sr.json does not provide required tolerance case %s' % exc)

coverage = COMMON_SUMMARY.get('provenance', {}).get('distribution_coverage', {})
check(coverage.get('summary_basis') == 'retained scalar evaluations only; no interval point estimates',
      'common_sr.json must identify the quantile basis')
check('originally successful runs with retained scalar common-subproblem evaluations' in common_plain,
      'Section 6.5 must limit scalar quantiles to retained evaluations')
if coverage:
    parse_numbers(common_plain,
                  r'These quantiles use ([\d,]+) of the ([\d,]+) originally successful runs',
                  [coverage['successful_retained_scalar_values'], coverage['originally_successful_runs']],
                  'Section 6.5 quantile coverage')

# Additional-check records are checked for complete keys, interval verdicts,
# and the manuscript's retained upper bounds. Feasibility is validated by the
# shared adjudication loader in the aggregation workflow.
certificate_file = DATA / 'crit_out_r4' / 'adjudicated.csv'
check(certificate_file.is_file(), 'missing final per-point adjudication file')
if certificate_file.is_file():
    with certificate_file.open(encoding='utf-8-sig', newline='') as stream:
        certificates = list(csv.DictReader(stream))
    keys = [(row['prob'], row['alg'], int(row['run'])) for row in certificates]
    check(len(keys) == len(set(keys)) == 134, 'adjudication must contain 134 unique point keys')
    for row in certificates:
        key = (row['prob'], row['alg'], row['run'])
        for field, lo_name, hi_name in [('theta', 'neg_theta_lo', 'neg_theta_hi'),
                                         ('dnorm', 'dnorm_lo', 'dnorm_hi')]:
            lo = Decimal(row[lo_name]); hi = Decimal(row[hi_name])
            check(lo <= hi, 'adjudication %s %s: reversed interval' % (key, field))
            expected = '1' if hi <= Decimal('1e-4') else '0' if lo > Decimal('1e-4') else '-1'
            check(row['pass_' + field] == expected and expected != '-1',
                  'adjudication %s %s: undecided or inconsistent verdict' % (key, field))
    passed = [row for row in certificates if row['pass_theta'] == row['pass_dnorm'] == '1']
    failed = [row for row in certificates if row['pass_theta'] == row['pass_dnorm'] == '0']
    check(len(passed) == 31 and len(failed) == 103, 'final additional-check partition differs from 31/103')
    if passed:
        check(max(Decimal(row['neg_theta_hi']) for row in passed) <= Decimal('3.6e-9'),
              'passing criticality intervals exceed the stated upper bound')
        check(max(Decimal(row['dnorm_hi']) for row in passed) <= Decimal('9.9e-5'),
              'passing displacement intervals exceed the stated upper bound')
    if failed:
        check(min(Decimal(row['neg_theta_lo']) for row in failed) >= Decimal('1.0198e-4'),
              'failed criticality intervals contradict the stated margin')
        check(sum(Decimal(row['dnorm_lo']) >= Decimal('4.84') for row in failed) == 102,
              'failed displacement count above 4.84 differs from 102')
        check(min(Decimal(row['dnorm_lo']) for row in failed) >= Decimal('1.4e-2'),
              'failed displacement interval is below the stated smallest margin')

    # Parse the printed safety bounds and compare them with the per-point
    # intervals, so editing a bound in prose cannot bypass the data checks.
    bounds = re.search(
        r'with the bounds .*?\\le([\d.]+)\\times10\^\{(-?\d+)\}'
        r' and .*?\\le([\d.]+)\\times10\^\{(-?\d+)\}', common_plain)
    check(bounds is not None, 'Section 6.5 passing-interval bounds are missing')
    if bounds and passed:
        printed = [Decimal(bounds.group(1)) * Decimal(10) ** int(bounds.group(2)),
                   Decimal(bounds.group(3)) * Decimal(10) ** int(bounds.group(4))]
        for actual, bound, name in [
                (max(Decimal(row['neg_theta_hi']) for row in passed), printed[0], 'criticality'),
                (max(Decimal(row['dnorm_hi']) for row in passed), printed[1], 'displacement')]:
            quantum = Decimal(1).scaleb(actual.adjusted() - 1)
            rounded_out = actual.quantize(quantum, rounding=ROUND_CEILING)
            check(bound == rounded_out,
                  'Section 6.5 %s bound %s must round outward from %s to %s' %
                  (name, bound, actual, rounded_out))

# Use the shared interval reader to bind cached table values to raw evaluations
# and the complete certificate/witness set. This also tests the tolerance cases
# and the original-success exception claim from the current input files.
try:
    from adjudication import CommonCriteria
    import numpy as np
    common = CommonCriteria(DATA / 'crit_out_r4', D['probs'], ALGS)
    diag = common.diagnostics()
    check(diag['adjudicated_points'] == 134 and
          diag['passes_criticality'] == diag['passes_displacement'] == 31 and
          diag['undecided_points'] == 0, 'shared reader does not confirm 134 / 31 / 103')
    for field, mode, avg_name, rob_name in [
            ('crit', 'A', 'avg_th', 'rob_th'),
            ('dnorm', 'A', 'avg_dn', 'rob_dn'),
            ('crit', 'B', 'avg_th_b', None),
            ('dnorm', 'B', 'avg_dn_b', None)]:
        rates = common.rates(field, '1e-4', mode)
        for i, alg in enumerate(ALGS):
            check(math.isclose(float(rates[:, i].mean()), D[avg_name][i], abs_tol=1e-9),
                  '%s/%s average for %s disagrees with stored evaluations' % (field, mode, alg))
            if rob_name is not None:
                robustness = float((rates[:, i] >= 90).mean() * 100)
                check(math.isclose(robustness, D[rob_name][i], abs_tol=1e-9),
                      '%s robustness for %s disagrees with stored evaluations' % (field, alg))
    theta_pass = common.pass_mask('crit', '1e-4')
    own_success = common.info == 0
    for i, alg in enumerate(ALGS):
        fails = int((own_success[:, i] & ~theta_pass[:, i]).sum())
        check(fails == (148 if alg == 'MPG-Explicit' else 0),
              'originally-successful common-criticality exceptions changed for %s: %d' % (alg, fails))
    values = common.criticality[own_success & np.isfinite(common.criticality)]
    check(values.size == coverage.get('successful_retained_scalar_values'),
          'quantile retained-scalar coverage disagrees with raw evaluations')
    check(int(own_success.sum()) == coverage.get('originally_successful_runs'),
          'original success coverage disagrees with raw evaluations')
    check(math.isclose(float(np.median(values)), D['theta_med'], rel_tol=1e-12),
          'median in r2_tables.json differs from retained scalar values')
    check(math.isclose(float(np.quantile(values, .99)), D['theta_q99'], rel_tol=1e-12),
          'q99 in r2_tables.json differs from retained scalar values')
except (ImportError, OSError, ValueError, KeyError) as exc:
    check(False, 'current raw evaluation/certificate audit failed: %s' % exc)

# The trial-count sentence describes an implied statistic for MPG-Explicit,
# pooled over successful runs of each problem. Recompute that exact statistic
# from the per-run export, not from a broad claim about individual iterations.
raw_export = DATA / 'results' / 'all_raw.csv'
check(raw_export.is_file(), 'per-run export missing for the trial-count check')
if raw_export.is_file():
    totals = {}
    with raw_export.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['Algorithm'] != 'MPG-Explicit' or float(row['Info']) != 0 or float(row['Iter']) <= 0:
                continue
            values = totals.setdefault(row['Problem'], [0.0, 0.0])
            values[0] += float(row['Nhev'])
            values[1] += float(row['m']) * float(row['Iter'])
    check(set(totals) == set(D['probs']), 'MPG-Explicit trial statistic must cover all 42 problems')
    trials = [numerator / denominator for numerator, denominator in totals.values()]
    parse_numbers(readable(section('subsec:discussion')),
                  r'For MPG-Explicit this statistic ranges from ([\d.]+) to ([\d.]+) over the 42 problems',
                  [min(trials), max(trials)], 'Section 6.10 implied mean trials', 2)

class_text = readable(section('subsec:problem-class'))
parse_numbers(class_text,
              r'five classes contain ([\d,]+), ([\d,]+), ([\d,]+), ([\d,]+) and ([\d,]+) problems',
              [classes[name] for name in ['Convex', 'Nonconvex', 'Multimodal', 'Quadratic',
                                         'Nonuniform/polynomial']], 'Section 6.4 class counts')
check('most pronounced on multimodal' not in class_text,
      'unsupported largest multimodal gap claim remains')
parse_numbers(class_text,
              r'On multimodal problems, EBNPG-N0, N1, N2, N3 and N6 each attain a class-average '
              r'success rate of ([+-]?\d+(?:\.\d+)?)% after rounding, compared with ([+-]?\d+(?:\.\d+)?)% for MPG-Implicit',
              [mround(D['extra']['clsmean']['Multimodal']['EBNPG-N2']),
               mround(D['extra']['clsmean']['Multimodal']['MPG-Implicit'])],
              'Section 6.4 multimodal average comparison')
for alg in ['EBNPG-N0', 'EBNPG-N1', 'EBNPG-N2', 'EBNPG-N3', 'EBNPG-N6']:
    check(mround(D['extra']['clsmean']['Multimodal'][alg]) == 98,
          '%s no longer rounds to 98 in the multimodal comparison' % alg)

discussion = readable(section('subsec:discussion'))
parse_numbers(discussion,
              r'On quadratic problems, CNPG-Max and MPG-Implicit tie at ([+-]?\d+(?:\.\d+)?)% '
              r'\(([+-]?\d+(?:\.\d+)?)% after rounding\)',
              [D['extra']['clsmean']['Quadratic']['CNPG-Max'],
               mround(D['extra']['clsmean']['Quadratic']['CNPG-Max'])],
              'Section 6.10 quadratic tie')
check(D['extra']['clsmean']['Quadratic']['CNPG-Max'] ==
      D['extra']['clsmean']['Quadratic']['MPG-Implicit'], 'data do not support the quadratic tie')
family_ranges = []
for prefix in ['EBNPG', 'CNPG', 'MPG']:
    vals = [D['rob'][i] for i, alg in enumerate(ALGS) if alg.startswith(prefix)]
    family_ranges.extend([min(vals), max(vals)])
parse_numbers(discussion,
              r'Overall robustness ranges from ([+-]?\d+(?:\.\d+)?)% to ([+-]?\d+(?:\.\d+)?)% for EBNPG, '
              r'from ([+-]?\d+(?:\.\d+)?)% to ([+-]?\d+(?:\.\d+)?)% for CNPG, and from ([+-]?\d+(?:\.\d+)?)% to ([+-]?\d+(?:\.\d+)?)% for MPG',
              family_ranges, 'Section 6.10 overall robustness ranges', 1)
check('Table~\\ref{tab:efficiency-robustness}' in section('subsec:discussion'),
      'overall robustness must cite its numerical table')
parse_numbers(discussion,
              r'class-average success rates reported.*?are ([+-]?\d+(?:\.\d+)?)--([+-]?\d+(?:\.\d+)?)% for EBNPG, '
              r'([+-]?\d+(?:\.\d+)?)% for CNPG, and ([+-]?\d+(?:\.\d+)?)--([+-]?\d+(?:\.\d+)?)% for MPG after rounding',
              [95, 98, 97, 70, 97], 'Section 6.10 multimodal class means')
check('especially on multimodal problems' not in discussion,
      'class-average success rate is still described as class robustness')

# These unchanged recommendations are now checked against the weighted-cost
# data as well as being asserted in the prose above.
mpg_explicit = IX['MPG-Explicit']
for ratio in ['1', '5', '20']:
    costs = D['oracle'][ratio]['wins']
    check(costs[mpg_explicit] == max(costs),
          'MPG-Explicit is no longer the method with most lowest-cost problems at ratio %s' % ratio)
check(D['oracle']['20']['medrel'][mpg_explicit] <
      D['oracle']['20']['medrel'][IX['EBNPG-N2']] <
      D['oracle']['20']['medrel'][IX['CNPG-Avg']],
      'weighted-cost ordering does not support the recommendation')


# ------------------------------------------------- structural checks
nb = txt.count('\\begin{revision}')
ne = txt.count('\\end{revision}')
check(nb == ne, 'revision environments unbalanced: %d begin / %d end' % (nb, ne))
# no nested revision (stack based)
depth = 0
nested = 0
for m in re.finditer(r'\\(begin|end)\{revision\}', txt):
    depth += 1 if m.group(1) == 'begin' else -1
    if depth > 1:
        nested += 1
check(nested == 0, '%d nested revision environments' % nested)
check(depth == 0, 'revision environments do not balance (final depth %d)' % depth)
# every \ref has a \label
labels = set(re.findall(r'\\label\{([^}]*)\}', txt))
refs = set(re.findall(r'\\ref\{([^}]*)\}', txt))
missing = sorted(refs - labels)
check(not missing, 'dangling \\ref: %s' % missing)
# no leftover placeholders
for pat in ['<<', 'TODO', 'FIXME', 'XXX']:
    check(pat not in txt, 'leftover placeholder %r in the manuscript' % pat)
# the release identifiers are checked for internal consistency rather than against
# hardcoded constants: a constant goes stale the moment the package is re-released
# (v3.0 -> v3.1), and a checker that fails for that reason is worse than none.
for pat in ['<SHA>', '<ID>', 'zenodo.<ID>']:
    check(pat not in txt, 'unfilled placeholder %r in the manuscript' % pat)

tags = set(_tags)
check(len(tags) == 1,
      'the manuscript cites %d distinct release tags: %s' % (len(tags), sorted(tags)))
tags_url = set(re.findall(r'releases/tag/([^}]+)\}', txt))
check(tags_url == tags,
      'the release tag in the URL %s does not match the tag in the text %s'
      % (sorted(tags_url), sorted(tags)))

shas = set(_shas)
check(len(shas) == 1,
      'the manuscript cites %d distinct commit ids: %s' % (len(shas), sorted(shas)))

dois = set(_dois)
check(len(dois) == 1,
      'the manuscript cites %d distinct Zenodo DOIs: %s' % (len(dois), sorted(dois)))
check(all(('\\url{https://doi.org/%s}' % d) in txt for d in dois),
      'the Zenodo DOI is not written as a doi.org URL: %s' % sorted(dois))

print('release tag : %s' % (sorted(tags)[0] if len(tags) == 1 else sorted(tags)))
print('commit      : %s' % (sorted(shas)[0] if len(shas) == 1 else sorted(shas)))
print('Zenodo DOI  : %s' % (sorted(dois)[0] if len(dois) == 1 else sorted(dois)))
print('   -- these must be the identifiers actually published; the checker only')
print('      verifies that the manuscript is consistent with itself.')
# tables/figures numbered consistently
n_tab = txt.count('\\begin{table}') + txt.count('\\begin{longtable}')
check(n_tab == 10, 'expected 10 tables (9 table + 1 longtable), found %d' % n_tab)
check(txt.count('\\begin{figure}') == 2, 'expected 2 figures')

# ------------------------------------------------- report
print('passed checks : %d' % OK)
print('failed checks : %d' % len(FAIL))
for f in FAIL:
    print('   FAIL', f)
sys.exit(1 if FAIL else 0)
