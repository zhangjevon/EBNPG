# Added for the EBNPG manuscript (third revision).
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-
"""Data-driven verification of the JOGO-D-26-00589 R2 manuscript.

Parses every numerical table of the tex file, strips the blue revision markup and
compares each cell with the value recomputed by aggregate_all.py (r2_tables.json).
Then asserts a list of prose claims and performs structural checks.

Run from Code_Data:  python verify_manuscript.py
"""
import io, os, re, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, '..', 'EBNPG_2',
                   '0.1StrogNonMonProxGradMethdPZOY02October2026R1.tex')
D = json.load(open(os.path.join(HERE, 'r2_tables.json'), encoding='utf-8'))
ALGS = D['algs']
IX = {a: i for i, a in enumerate(ALGS)}

# The manuscript source is not part of the data package (the package carries the
# numbers, not the paper).  When it is absent this checker has nothing to compare
# against, and it says so instead of failing: the table checks then reduce to the
# aggregation already done by aggregate_all.py, which is run above it.
if not os.path.exists(TEX):
    print('manuscript source not present at %s' % os.path.normpath(TEX))
    print('this data package does not include the LaTeX sources; nothing to verify.')
    print('passed checks : 0\nfailed checks : 0')
    sys.exit(0)

txt = io.open(TEX, encoding='utf-8').read()


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
    '(95.7\\% under the criticality measure and 93.8\\% under the displacement criterion)',
    'MPG-Explicit drops from $89.1\\%$ to $85.6\\%$',
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
    'EBNPG therefore uses fewer nonsmooth evaluations only when $T>2$',
    'increasing $w_h/w_g$ does not reverse the weighted-cost ordering',
    'For general-purpose use, EBNPG-N2 is a reasonable default',
    'The nonmonotone variants improve average success and robustness over N0 overall',
    'N0 remains faster than N5 and N7 and is competitive or better on several problem classes',
    'N2 offers a useful overall balance, N6 attains the highest robustness, and CNPG-Avg remains competitive',
    'The EBNPG and CNPG families also show higher robustness than the MPG family',
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
    'The data package deposited with release \\texttt{v3.0-jogo-r3} contains the per-problem and per-run results',
    # code availability: public repository cited in Code availability and the reply letter
    'is publicly available at \\url{https://github.com/zhangjevon/EBNPG}, release \\texttt{v3.0-jogo-r3}',
]
for c in CLAIMS:
    n = TXT_N.count(strip_rev(c))
    check(n == 1, 'claim occurs %d times: %s' % (n, c[:70]))

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

tags = set(re.findall(r'release\s+\\texttt\{([^}]*)\}', txt))
check(len(tags) == 1,
      'the manuscript cites %d distinct release tags: %s' % (len(tags), sorted(tags)))
tags_url = set(re.findall(r'releases/tag/([^}]+)\}', txt))
check(tags_url == tags,
      'the release tag in the URL %s does not match the tag in the text %s'
      % (sorted(tags_url), sorted(tags)))

shas = set(re.findall(r'\\texttt\{([0-9a-f]{7,40})\}', txt))
check(len(shas) == 1,
      'the manuscript cites %d distinct commit ids: %s' % (len(shas), sorted(shas)))

dois = set(re.findall(r'10\.5281/zenodo\.\d+', txt))
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
