# -*- coding: utf-8 -*-
# Added for the EBNPG manuscript.
# SPDX-License-Identifier: GPL-3.0-or-later
"""R2 master aggregation for the EBNPG manuscript.

Recomputes every numerical table of the manuscript from the stored solver
outputs, after (i) the MGH33 block has been replaced by a fully documented
rerun and (ii) DTLZ2 has been reclassified from Convex to Nonconvex.

Outputs:
  r2_tables.json              all numbers needed for the tex tables
"""
import numpy as np, scipy.io as sio, os, json
from scipy.stats import chi2, wilcoxon, binomtest

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, 'results')
RESW = os.path.join(BASE, 'results_w_rebuilt')   # R2 rerun of the adaptive study
CRIT = os.path.join(BASE, 'crit_out')
QP2 = os.path.join(BASE, 'qp_out2')
NEF = os.path.join(BASE, 'ne_full')

probs = ['AP1','AP3','AP4','BK1','DD1','DGO2','DTLZ2','FA1','Far1','FDS','Hil1',
         'IKK1','KW2','Lov4','Lov5','Lov6','LTDZ','MGH16','MGH26','MGH33','MLF2',
         'MMR1','MMR2','MMR3','MOP5','MOP6','MOP7','NIC1','NIC2','NIC3','NIM1',
         'NSP1','NSP2','PNR','QV1','SK2','SP1','Toi4','Toi9','VU2','ZDT6','ZLT1']
algs = ['MPG-Explicit','MPG-Armijo','MPG-Implicit','CNPG-Max','CNPG-Avg',
        'EBNPG-N0','EBNPG-N1','EBNPG-N2','EBNPG-N3','EBNPG-N5','EBNPG-N6','EBNPG-N7']
MPG = algs[:3]; CNPG = algs[3:5]; EBNPG = algs[5:]
VARIANTS = ['EBNPG-N0','EBNPG-N1','EBNPG-N2','EBNPG-N3','EBNPG-N5','EBNPG-N6','EBNPG-N7']

# ---- problem classes (DTLZ2 reclassified: Convex -> Nonconvex) -------------
CLASS = {}
for p in ['AP1','AP4','BK1','DD1','DGO2','FDS','IKK1','MOP7','NIC1','NIC2','NIC3','SP1','ZLT1']:
    CLASS[p] = 'Convex'
for p in ['AP3','DTLZ2','FA1','Hil1','KW2','Lov4','Lov5','Lov6','MMR1','MMR2','MMR3',
          'MOP5','MOP6','PNR','SK2','ZDT6']:
    CLASS[p] = 'Nonconvex'
for p in ['Far1','LTDZ','MGH26','MLF2','NIM1','QV1','VU2']:
    CLASS[p] = 'Multimodal'
for p in ['MGH16','MGH33','Toi4','Toi9']:
    CLASS[p] = 'Quadratic'
for p in ['NSP1','NSP2']:
    CLASS[p] = 'Nonuniform/polynomial'
assert len(CLASS) == 42 and set(CLASS) == set(probs)
CLASSES = ['Convex','Nonconvex','Multimodal','Quadratic','Nonuniform/polynomial']
SHORT = {'CNPG-Avg':'Avg','CNPG-Max':'Max'}


def mround(x, nd=0):
    """MATLAB round: half away from zero (the R1 tables were produced by MATLAB)."""
    f = 10.0 ** nd
    return float(np.sign(x) * np.floor(np.abs(x) * f + 0.5) / f)


def gv(e, n):
    v = e[n]
    while v.dtype == object and v.size == 1:
        v = v.reshape(-1)[0]
    return v


def short_name(a):
    return SHORT.get(a, a.replace('EBNPG-', ''))


# ================= load per-run data =======================================
SR = np.zeros((12, 42))          # success rate in percent
OK = np.zeros((12, 42))          # number of successful runs
MEDIT = np.full((12, 42), np.nan)
MEDTIME = np.full((12, 42), np.nan)
NGEV = [None] * 42; NHEV = [None] * 42; IT = [None] * 42; INF = [None] * 42
for pi, p in enumerate(probs):
    d = sio.loadmat(os.path.join(RES, p + '_results.mat'))['prob_results']
    for a in range(12):
        e = d[a, 0]
        inf = np.asarray(gv(e, 'info_vals')).ravel().astype(float)
        it = np.asarray(gv(e, 'iter_vals')).ravel().astype(float)
        ngev = np.asarray(gv(e, 'ngev_vals')).ravel().astype(float)
        nhev = np.asarray(gv(e, 'nhev_vals')).ravel().astype(float)
        tm = np.asarray(gv(e, 'time_vals')).ravel().astype(float)
        NGEV[pi] = ngev if a == 0 else NGEV[pi]
        if a == 0:
            NGEV[pi], NHEV[pi], IT[pi], INF[pi] = [], [], [], []
        NGEV[pi].append(ngev); NHEV[pi].append(nhev); IT[pi].append(it); INF[pi].append(inf)
        ok = inf == 0
        OK[a, pi] = ok.sum()
        SR[a, pi] = 100.0 * ok.sum() / len(inf)
        if ok.sum():
            MEDIT[a, pi] = np.median(it[ok])
            MEDTIME[a, pi] = np.median(tm[ok])

AVG = SR.mean(axis=1)                       # average success rate over 42 problems
ROB = (SR >= 90).sum(axis=1) / 42 * 100     # robustness: per algorithm, over problems
BEST = np.zeros(12)                          # best count (ties counted)
for pi in range(42):
    mx = SR[:, pi].max()
    for a in range(12):
        if SR[a, pi] >= mx - 1e-9:
            BEST[a] += 1
MEDIT_MED = np.nanmedian(MEDIT, axis=1)
MEDTIME_MED = np.nanmedian(MEDTIME, axis=1)

# ================= common stationarity (crit_out) ==========================
CR = {}; DN = {}; INFO = {}
for p in probs:
    o = sio.loadmat(os.path.join(CRIT, p + '_crit.mat'))['out']
    CR[p] = np.asarray(gv(o, 'crit'))
    DN[p] = np.asarray(gv(o, 'dnorm'))
    INFO[p] = np.asarray(gv(o, 'info'))

def crit_rates(field, tol, mode='A'):
    pp = np.full((42, 12), np.nan)
    for pi, p in enumerate(probs):
        v = CR[p] if field == 'crit' else DN[p]
        for a in range(12):
            m = np.isfinite(v[a]) & (v[a] <= tol)
            if mode == 'B':
                m = m & (INFO[p][a] == 0)
            pp[pi, a] = 100.0 * m.sum() / v[a].size
    return pp

PP_TH = crit_rates('crit', 1e-4, 'A')          # -Theta_1 <= 1e-4
PP_DN = crit_rates('dnorm', 1e-4, 'A')         # ||d_1|| <= 1e-4
PP_TH_B = crit_rates('crit', 1e-4, 'B')
PP_DN_B = crit_rates('dnorm', 1e-4, 'B')
AVG_TH = PP_TH.mean(axis=0); ROB_TH = (PP_TH >= 90).sum(axis=0) / 42 * 100
AVG_DN = PP_DN.mean(axis=0); ROB_DN = (PP_DN >= 90).sum(axis=0) / 42 * 100
AVG_TH_B = PP_TH_B.mean(axis=0); AVG_DN_B = PP_DN_B.mean(axis=0)

succ_vals = []
for p in probs:
    for a in range(12):
        v = CR[p][a][INFO[p][a] == 0]; v = v[np.isfinite(v)]
        succ_vals.extend(v.tolist())
succ_vals = np.array(succ_vals)

# ================= QP counts (Table 7) ======================================
plus_one = {'MPG-Explicit': 1, 'MPG-Armijo': 1}
imp_qp, imp_it = {}, {}
for p in probs:
    f = os.path.join(QP2, p + '_qp.mat')
    if not os.path.exists(f):
        continue
    o = sio.loadmat(f)['out']
    q = np.asarray(gv(o, 'qp_count')).ravel().astype(float)
    inf = np.asarray(gv(o, 'info')).ravel().astype(float)
    it = np.asarray(gv(o, 'iter')).ravel().astype(float)
    ok = inf == 0
    if ok.sum():
        imp_qp[p] = float(np.median(q[ok])); imp_it[p] = float(np.median(it[ok]))

zero = {a: [p for p in probs if (SR[algs.index(a), probs.index(p)] == 0)] for a in algs}
common = [p for p in probs if all(SR[algs.index(a), probs.index(p)] > 0 for a in algs)]
QP_ROWS = []
for a in algs:
    ai = algs.index(a)
    pit, pqp = [], []
    for p in common:
        mit = MEDIT[ai, probs.index(p)]
        mqp = imp_qp[p] if a == 'MPG-Implicit' else mit + plus_one.get(a, 0)
        pit.append(mit); pqp.append(mqp)
    mqp_med = float(np.median(pqp)); mit_med = float(np.median(pit))
    QP_ROWS.append((a, mqp_med, mit_med, mqp_med / mit_med))

# ================= oracle cost (Table 8) ====================================
perit = {}
for ai, a in enumerate(algs):
    ng = nh = nit = 0.0
    for pi in range(42):
        inf = INF[pi][ai]; it = IT[pi][ai]
        ok = (inf == 0) & (it > 0)
        if ok.sum() == 0:
            continue
        ng += NGEV[pi][ai][ok].sum(); nh += NHEV[pi][ai][ok].sum(); nit += it[ok].sum()
    perit[a] = (ng / nit, nh / nit)

ORACLE = {}
for rho in [1, 5, 20]:
    C = np.full((42, 12), np.nan)
    for pi in range(42):
        for ai in range(12):
            inf = INF[pi][ai]; ngev = NGEV[pi][ai]; nhev = NHEV[pi][ai]
            ok = (inf == 0) & np.isfinite(ngev) & np.isfinite(nhev)
            if ok.sum() == 0:
                continue
            C[pi, ai] = np.median(ngev[ok] + rho * nhev[ok])
    wins = np.zeros(12); rel = np.full((42, 12), np.nan)
    for pi in range(42):
        row = C[pi]
        if np.all(np.isnan(row)):
            continue
        mn = np.nanmin(row)
        for ai in range(12):
            if np.isnan(row[ai]):
                continue
            rel[pi, ai] = row[ai] / mn
            if row[ai] <= mn * (1 + 1e-12):
                wins[ai] += 1
    ORACLE[rho] = (wins, np.nanmedian(rel, axis=0))

# ================= CMH + Wilcoxon (Table 6) =================================
REF = algs.index('EBNPG-N2')
def cmh(a):
    num = den = 0.0
    for k in range(42):
        n1 = n2 = 100
        A = OK[REF, k]; C = OK[a, k]
        m1 = A + C; N = 200; m2 = N - m1
        if m1 == 0 or m2 == 0:
            continue
        num += A - n1 * m1 / N
        den += n1 * n2 * m1 * m2 / (N * N * (N - 1))
    if den <= 0:
        return 0.0, 1.0
    stat = num * num / den
    return stat, float(chi2.sf(stat, 1))

CMH = []
for a in range(12):
    if a == REF:
        continue
    stat, p = cmh(a)
    d = 100 * (OK[REF] - OK[a]) / 100.0
    pw = float(wilcoxon(d, alternative='two-sided', zero_method='wilcox').pvalue)
    CMH.append((algs[a], float(np.mean(d)), stat, p, pw,
                int((d > 1e-12).sum()), int((d < -1e-12).sum()), int((np.abs(d) <= 1e-12).sum())))
order = sorted(range(len(CMH)), key=lambda i: CMH[i][3])
holm = [0.0] * len(CMH); prev = 0.0
for rank, i in enumerate(order):
    prev = max(prev, min(1.0, (len(CMH) - rank) * CMH[i][3]))
    holm[i] = prev

# ================= fixed kernel study (Table 10) ============================
KROWS = []
for p in probs:
    f = os.path.join(NEF, p + '_N2.mat')
    if not os.path.exists(f):
        continue
    m = sio.loadmat(f)['R']
    k0 = gv(m, 'K0')[0, 0]; k1 = gv(m, 'K1')[0, 0]
    i0 = np.asarray(gv(k0, 'info')).ravel(); i1 = np.asarray(gv(k1, 'info')).ravel()
    it0 = np.asarray(gv(k0, 'iter')).ravel(); it1 = np.asarray(gv(k1, 'iter')).ravel()
    w = np.asarray(gv(m, 'w')).ravel()
    s0, s1 = i0 == 0, i1 == 0
    b = int((s1 & ~s0).sum()); c = int((s0 & ~s1).sum())
    pv = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    KROWS.append(dict(p=p, sr0=100 * s0.mean(), sr1=100 * s1.mean(),
                      mi0=float(np.median(it0[s0])) if s0.any() else np.nan,
                      mi1=float(np.median(it1[s1])) if s1.any() else np.nan,
                      b=b, c=c, pval=float(pv), wratio=float(w.max() / w.min())))
nk = len(KROWS)
sr0 = np.array([r['sr0'] for r in KROWS]); sr1 = np.array([r['sr1'] for r in KROWS])
kd = sr1 - sr0
B = sum(r['b'] for r in KROWS); C = sum(r['c'] for r in KROWS)
POOL_P = binomtest(B, B + C, 0.5).pvalue
WIL_P = float(wilcoxon(kd, zero_method='wilcox').pvalue)
mi0 = np.array([r['mi0'] for r in KROWS]); mi1 = np.array([r['mi1'] for r in KROWS])
okm = np.isfinite(mi0) & np.isfinite(mi1)
KSTAT = dict(n=nk, mean0=float(sr0.mean()), mean1=float(sr1.mean()),
             diff=float(kd.mean()), better=int((kd > 1e-9).sum()),
             worse=int((kd < -1e-9).sum()), tied=int((np.abs(kd) <= 1e-9).sum()),
             pool_B=B, pool_C=C, pool_p=float(POOL_P), wil_p=WIL_P,
             med0=float(np.median(mi0[okm])), med1=float(np.median(mi1[okm])),
             ndiff=int((np.abs(kd) > 1e-9).sum()))
wr = np.array([r['wratio'] for r in KROWS])
KSPLIT = []
for sel, lab in [(wr < 1.5, 'iso'), ((wr >= 1.5) & (wr < 3.0), 'mid'), (wr >= 3.0, 'aniso')]:
    nsel = int(sel.sum())
    if nsel == 0:
        KSPLIT.append((lab, 0, None, None, None, None, None)); continue
    bb = sum(KROWS[i]['b'] for i in range(nk) if sel[i])
    cc = sum(KROWS[i]['c'] for i in range(nk) if sel[i])
    pv = binomtest(bb, bb + cc, 0.5).pvalue if bb + cc else 1.0
    KSPLIT.append((lab, nsel, float(sr0[sel].mean()), float(sr1[sel].mean()),
                   float(kd[sel].mean()), bb, cc, float(pv)))

# ================= class tables (Tables 2, 5, 9) ============================
def class_rows():
    """Table 2: per-class family ranges + best.

    R1 determined the ranges and the winners on the class means rounded to the
    nearest integer (MATLAB round, half away from zero); we reproduce that
    convention exactly (it is what turns the Quadratic 96.5 means into 97 and
    lists five multimodal winners at 98).
    """
    out = []
    for cl in CLASSES:
        idx = [probs.index(p) for p in probs if CLASS[p] == cl]
        mpg = [SR[algs.index(a), idx].mean() for a in MPG]
        cn = [SR[algs.index(a), idx].mean() for a in CNPG]
        eb = [SR[algs.index(a), idx].mean() for a in EBNPG]
        vals = {a: mround(SR[algs.index(a), idx].mean()) for a in algs}
        mx = max(vals.values())
        winners = [short_name(a) for a in algs if vals[a] == mx]
        def rng(v):
            lo, hi = mround(min(v)), mround(max(v))
            return '%d' % lo if lo == hi else '%d--%d' % (lo, hi)
        out.append((cl, rng(mpg), rng(cn), rng(eb), '%d (%s)' % (mx, ','.join(winners))))
    return out

TAB2 = class_rows()

def tab5_rows():
    """Table 5: per-class average SR of the seven EBNPG variants."""
    out = []
    for cl in CLASSES:
        idx = [probs.index(p) for p in probs if CLASS[p] == cl]
        out.append((cl, [mround(SR[algs.index(v), idx].mean()) for v in VARIANTS]))
    return out

TAB5 = tab5_rows()

def weighted_summary():
    """Table 9: per-class Best/Median of the seven variants, Euclidean vs adaptive."""
    WSR = np.full((42, 7), np.nan)
    for pi, p in enumerate(probs):
        m = sio.loadmat(os.path.join(RESW, p + '_W_results.mat'))['prob_results']
        for a in range(7):
            e = m[a, 0]
            inf = np.asarray(gv(e, 'info_vals')).ravel().astype(float)
            WSR[pi, a] = 100.0 * (inf == 0).mean() / 1.0
    ESR = np.zeros((42, 7))
    for pi, p in enumerate(probs):
        for a, v in enumerate(VARIANTS):
            ESR[pi, a] = SR[algs.index(v), pi]
    out = []
    for cl in CLASSES + ['Overall']:
        if cl == 'Overall':
            idx = list(range(42))
        else:
            idx = [probs.index(p) for p in probs if CLASS[p] == cl]
        e_best = np.max(ESR[idx], axis=1).mean(); e_med = np.median(ESR[idx], axis=1).mean()
        w_best = np.max(WSR[idx], axis=1).mean(); w_med = np.median(WSR[idx], axis=1).mean()
        out.append((cl, len(idx), e_best, e_med, w_best, w_med, w_best - e_best))
    return out

TAB9 = weighted_summary()

# ================= extra statistics for the prose ============================
EXTRA = {}

# ---- pooled IQR of iteration counts (several candidate definitions) ---------
iqr = {}
iqr_all = {}      # all runs, failures at their stored iteration count
iqr_pm = {}       # median over problems of the per-problem IQR (successful runs)
for ai, a in enumerate(algs):
    pool = []; pool_all = []; pm = []
    for pi in range(42):
        inf = INF[pi][ai]; it = IT[pi][ai]
        ok = (inf == 0) & (it > 0)
        s = it[ok]
        pool.extend(s.tolist())
        pool_all.extend(it.tolist())
        if len(s):
            q1, q3 = np.quantile(s, .25), np.quantile(s, .75)
            pm.append(q3 - q1)
    pool = np.array(pool); pool_all = np.array(pool_all)
    iqr[a] = (float(np.quantile(pool, .25)), float(np.quantile(pool, .75)),
              float(np.quantile(pool, .75) - np.quantile(pool, .25)))
    # prose claim: pooled IQR over ALL stored iteration values (failures included)
    iqr_all[a] = (float(np.quantile(pool_all, .25)), float(np.quantile(pool_all, .75)),
                  float(np.quantile(pool_all, .75) - np.quantile(pool_all, .25)))
    iqr_pm[a] = float(np.median(pm))
EXTRA['iqr'] = iqr
EXTRA['iqr_all'] = iqr_all
EXTRA['iqr_pm'] = iqr_pm

# ---- per-class mean SR for every algorithm (prose claims) -------------------
CLSMEAN = {}
for cl in CLASSES:
    idx = [probs.index(p) for p in probs if CLASS[p] == cl]
    CLSMEAN[cl] = {a: float(SR[algs.index(a), idx].mean()) for a in algs}
EXTRA['clsmean'] = CLSMEAN

# ---- class counts -----------------------------------------------------------
EXTRA['clscount'] = {cl: sum(1 for p in probs if CLASS[p] == cl) for cl in CLASSES}

# ---- CNPG-Max vs N5, CNPG-Avg vs N6: problems with differing statistics -----
def pair_diff(a1, a2):
    """Count problems on which the success rate or the median iteration count
    differs (the R1 convention: CPU time is too noisy to enter this count;
    both pairs give 38 of 42 under the old and the new MGH33 data)."""
    i, j = algs.index(a1), algs.index(a2)
    n = 0
    for pi in range(42):
        d = False
        if abs(SR[i, pi] - SR[j, pi]) > 1e-9:
            d = True
        a, b = MEDIT[i, pi], MEDIT[j, pi]
        if (np.isfinite(a) != np.isfinite(b)) or (np.isfinite(a) and abs(a - b) > 1e-9):
            d = True
        n += 1 if d else 0
    d = SR[i] - SR[j]
    return dict(any_stat=n, sr_better=int((d > 1e-9).sum()),
                sr_worse=int((d < -1e-9).sum()), sr_tied=int((np.abs(d) <= 1e-9).sum()))
EXTRA['max_vs_n5'] = pair_diff('CNPG-Max', 'EBNPG-N5')
EXTRA['avg_vs_n6'] = pair_diff('CNPG-Avg', 'EBNPG-N6')

# ---- EBNPG-N2 vs CNPG-Avg head to head --------------------------------------
def head2head(a1, a2):
    i, j = algs.index(a1), algs.index(a2)
    d = SR[i] - SR[j]
    better = [(probs[pi], float(SR[i, pi]), float(SR[j, pi])) for pi in range(42) if d[pi] > 1e-9]
    worse = [(probs[pi], float(SR[i, pi]), float(SR[j, pi])) for pi in range(42) if d[pi] < -1e-9]
    better.sort(key=lambda t: -(t[1] - t[2]))
    return dict(b=len(better), w=len(worse), t=int((np.abs(d) <= 1e-9).sum()),
                top=better[:4], worst=sorted(worse, key=lambda t: (t[1] - t[2]))[:4])
EXTRA['n2_vs_avg'] = head2head('EBNPG-N2', 'CNPG-Avg')
EXTRA['n6_vs_avg'] = head2head('EBNPG-N6', 'CNPG-Avg')

# ---- adaptive study: per-problem best variant, Euclidean vs weighted --------
WSR = np.full((42, 7), np.nan)
for pi, p in enumerate(probs):
    m = sio.loadmat(os.path.join(RESW, p + '_W_results.mat'))['prob_results']
    for a in range(7):
        e = m[a, 0]
        inf = np.asarray(gv(e, 'info_vals')).ravel().astype(float)
        WSR[pi, a] = 100.0 * (inf == 0).mean()
EBEST = np.zeros(42); WBEST = np.zeros(42)
EMED = np.zeros(42); WMED = np.zeros(42)
for pi in range(42):
    esr = np.array([SR[algs.index(v), pi] for v in VARIANTS])
    EBEST[pi] = esr.max(); WBEST[pi] = WSR[pi].max()
    EMED[pi] = np.median(esr); WMED[pi] = np.median(WSR[pi])
dd = WBEST - EBEST
improve = [(probs[pi], EBEST[pi], WBEST[pi]) for pi in range(42) if dd[pi] > 1e-9]
drop = [(probs[pi], EBEST[pi], WBEST[pi]) for pi in range(42) if dd[pi] < -1e-9]
improve.sort(key=lambda t: -(t[2] - t[1]))
drop.sort(key=lambda t: (t[2] - t[1]))
EXTRA['adapt'] = dict(
    n_improve=len(improve), improve=improve, n_drop=len(drop), drop=drop,
    drop_lo=float(dd[dd < -1e-9].min()) if len(drop) else 0.0,
    drop_hi=float(dd[dd < -1e-9].max()) if len(drop) else 0.0,
    both100=int(((EBEST >= 100 - 1e-9) & (WBEST >= 100 - 1e-9)).sum()),
    zlt1=(float(EBEST[probs.index('ZLT1')]), float(WBEST[probs.index('ZLT1')])),
    # per-variant overall mean SR, Euclidean vs weighted
    var_mean={v: (float(SR[algs.index(v)].mean()),
                  float(np.nanmean(WSR[:, VARIANTS.index(v)]))) for v in VARIANTS},
)
# NIC1 details
pi = probs.index('NIC1')
EXTRA['nic1'] = dict(e={v: float(SR[algs.index(v), pi]) for v in VARIANTS},
                     w={v: float(WSR[pi, VARIANTS.index(v)]) for v in VARIANTS})

# ---- QP ratio statistics (Table 7 prose) ------------------------------------
ratios = []
for p in common:
    a = 'MPG-Implicit'
    mit = MEDIT[algs.index(a), probs.index(p)]
    mqp = imp_qp[p]
    ratios.append(mqp / mit if mit > 0 else np.nan)
ratios = np.array([r for r in ratios if np.isfinite(r)])
imp_qp_med = float(np.median([imp_qp[p] for p in common]))
imp_it_med = float(np.median([MEDIT[algs.index('MPG-Implicit'), probs.index(p)] for p in common]))
EXTRA['qp'] = dict(med_qp=imp_qp_med, med_it=imp_it_med,
                   ratio_lo=float(ratios.min()), ratio_hi=float(ratios.max()),
                   ratio_med=float(np.median(ratios)),
                   n_gt2=int((ratios > 2).sum()), n_gt3=int((ratios > 3).sum()),
                   n_common=len(common),
                   overhead=100 * (imp_qp_med / imp_it_med - 1))

# ---- leading-group spread (conclusion claim) --------------------------------
srt = np.sort(AVG)[::-1]
EXTRA['spread'] = dict(top4=float(srt[0] - srt[3]), top7=float(srt[0] - srt[6]),
                       ebnpg_rob=(float(ROB[5:].min()), float(ROB[5:].max())),
                       mpg_rob=(float(ROB[:3].min()), float(ROB[:3].max())),
                       n2_vs_avg_ph=None)

# ---- MPG iteration range / EBNPG iteration range ----------------------------
EXTRA['medit_range'] = dict(
    eb=(float(MEDIT_MED[5:].min()), float(MEDIT_MED[5:].max())),
    mpg=(float(MEDIT_MED[:3].min()), float(MEDIT_MED[:3].max())))

# ---- N0 vs nonmonotone variants (class level) -------------------------------
EXTRA['n0_vs_nonmono'] = {cl: (CLSMEAN[cl]['EBNPG-N0'],
                               max(CLSMEAN[cl][v] for v in VARIANTS[1:]))
                          for cl in CLASSES}

# ---- weighted-cost wins for MPG-Explicit ------------------------------------
EXTRA['mpx_wins'] = {str(r): int(ORACLE[r][0][algs.index('MPG-Explicit')]) for r in [1, 5, 20]}
EXTRA['n2_rel20'] = float(ORACLE[20][1][algs.index('EBNPG-N2')])

# ================= dump =====================================================
out = dict(
    algs=algs, probs=probs, CLASS=CLASS,
    avg=AVG.tolist(), rob=ROB.tolist(), best=BEST.tolist(),
    medit=MEDIT_MED.tolist(), medtime=MEDTIME_MED.tolist(),
    avg_th=AVG_TH.tolist(), rob_th=ROB_TH.tolist(),
    avg_dn=AVG_DN.tolist(), rob_dn=ROB_DN.tolist(),
    avg_th_b=AVG_TH_B.tolist(), avg_dn_b=AVG_DN_B.tolist(),
    theta_med=float(np.median(succ_vals)), theta_q99=float(np.quantile(succ_vals, .99)),
    perit={a: perit[a] for a in algs},
    oracle={str(r): dict(wins=ORACLE[r][0].tolist(), medrel=ORACLE[r][1].tolist()) for r in ORACLE},
    cmh=[dict(alg=r[0], mean_d=r[1], chi2=r[2], p=r[3], p_w=r[4], b=r[5], w=r[6], t=r[7], p_h=holm[i])
         for i, r in enumerate(CMH)],
    qp=[dict(alg=r[0], qp=r[1], it=r[2], ratio=r[3]) for r in QP_ROWS],
    common_subset=common,
    kstat=KSTAT, ksplit=KSPLIT,
    krows=KROWS,
    tab2=TAB2, tab5=TAB5,
    tab9=[dict(cls=r[0], n=r[1], e_best=r[2], e_med=r[3], w_best=r[4], w_med=r[5], d=r[6]) for r in TAB9],
    extra=EXTRA,
)
json.dump(out, open(os.path.join(BASE, 'r2_tables.json'), 'w'), indent=1)

# ---------------- console summary ------------------------------------------
print('=== Table 3 ranking ===')
for k in np.argsort(-AVG):
    print('  %-13s avg=%6.2f rob=%5.1f best=%2d medit=%5.1f medtime=%.2f'
          % (algs[k], AVG[k], ROB[k], BEST[k], MEDIT_MED[k], MEDTIME_MED[k]))
print('\n=== Table 4 ===')
for k in np.argsort(-AVG_TH):
    print('  %-13s flag=%6.2f th=%6.2f dn=%6.2f | rob %5.1f %5.1f %5.1f | B: th=%6.2f dn=%6.2f'
          % (algs[k], AVG[k], AVG_TH[k], AVG_DN[k], ROB[k], ROB_TH[k], ROB_DN[k], AVG_TH_B[k], AVG_DN_B[k]))
print('\n-Theta among originally successful: median=%.2e q99=%.2e' % (np.median(succ_vals), np.quantile(succ_vals, .99)))
print('\n=== Table 2 ===')
for r in TAB2:
    print('  %-24s %s' % (r[0], ' | '.join(r[1:])))
print('\n=== Table 5 ===')
for r in TAB5:
    print('  %-24s %s' % (r[0], r[1]))
print('\n=== Table 9 ===')
for r in TAB9:
    print('  %-24s (%d) E: %5.1f %5.1f  W: %5.1f %5.1f  d=%+5.1f' % r)
print('\n=== Table 6 (CMH) ===')
for r, h in zip(CMH, holm):
    print('  %-13s meanD=%+6.2f b/w/t=%d/%d/%d chi2=%9.2f p=%.2e p_h=%.2e p_w=%.3f'
          % (r[0], r[1], r[5], r[6], r[7], r[2], r[3], h, r[4]))
print('\n=== Table 7 (QP, common subset %d) ===' % len(common))
for r in QP_ROWS:
    print('  %-13s qp=%6.1f it=%6.1f ratio=%.2f' % r)
print('\n=== Table 8 (oracle) ===')
for k, a in enumerate(algs):
    print('  %-13s ng=%.2f nh=%.2f wins=%s medrel20=%.2f'
          % (a, perit[a][0], perit[a][1],
             [int(ORACLE[r][0][k]) for r in [1, 5, 20]], ORACLE[20][1][k]))
print('\n=== Table 10 (fixed kernel, %d problems) ===' % nk)
print('  mean K0=%.2f K1=%.2f diff=%+.2f better=%d worse=%d tied=%d pool(B=%d,C=%d,p=%.3f) wil p=%.3f med0=%.1f med1=%.1f ndiff=%d'
      % (KSTAT['mean0'], KSTAT['mean1'], KSTAT['diff'], KSTAT['better'], KSTAT['worse'],
         KSTAT['tied'], B, C, POOL_P, WIL_P, KSTAT['med0'], KSTAT['med1'], KSTAT['ndiff']))
for s in KSPLIT:
    print('  split %-6s n=%2d K0=%s K1=%s diff=%s p=%s' % (s[0], s[1],
          ('%.2f' % s[2]) if s[2] is not None else '-', ('%.2f' % s[3]) if s[3] is not None else '-',
          ('%+.2f' % s[4]) if s[4] is not None else '-', ('%.3g' % s[7]) if s[1] else '-'))
print('\n=== EXTRA (prose statistics) ===')
print('class counts:', EXTRA['clscount'])
print('IQR (pooled, successful runs):')
for a in algs:
    q1, q3, w = iqr[a]
    print('  %-13s q1=%5.1f q3=%5.1f iqr=%5.1f | all-runs iqr=%5.1f | per-problem med IQR=%5.1f'
          % (a, q1, q3, w, iqr_all[a][2], iqr_pm[a]))
print('class means (per algorithm):')
for cl in CLASSES:
    print('  %-24s' % cl, ' '.join('%s=%.1f' % (short_name(a), CLSMEAN[cl][a]) for a in algs))
print('CNPG-Max vs N5  (any stat differs):', EXTRA['max_vs_n5'])
print('CNPG-Avg vs N6  (any stat differs):', EXTRA['avg_vs_n6'])
print('N2 vs CNPG-Avg:', EXTRA['n2_vs_avg']['b'], '/', EXTRA['n2_vs_avg']['w'], '/', EXTRA['n2_vs_avg']['t'],
      ' top:', EXTRA['n2_vs_avg']['top'][:3], ' worst:', EXTRA['n2_vs_avg']['worst'][:2])
print('N6 vs CNPG-Avg:', EXTRA['n6_vs_avg']['b'], '/', EXTRA['n6_vs_avg']['w'], '/', EXTRA['n6_vs_avg']['t'],
      ' top:', EXTRA['n6_vs_avg']['top'][:3], ' worst:', EXTRA['n6_vs_avg']['worst'][:2])
print('adapt: improve=%d drop=%d both100=%d zlt1=%s' %
      (EXTRA['adapt']['n_improve'], EXTRA['adapt']['n_drop'], EXTRA['adapt']['both100'], EXTRA['adapt']['zlt1']))
print('  improve list:', [(p, e, w) for p, e, w in EXTRA['adapt']['improve']])
print('  drop list   :', [(p, e, w) for p, e, w in EXTRA['adapt']['drop']])
print('  drop range  : %.1f..%.1f' % (EXTRA['adapt']['drop_lo'], EXTRA['adapt']['drop_hi']))
print('  variant mean (E, W):')
for v in VARIANTS:
    print('    %-9s %6.2f %6.2f  (%+.2f)' % (v, EXTRA['adapt']['var_mean'][v][0], EXTRA['adapt']['var_mean'][v][1],
          EXTRA['adapt']['var_mean'][v][1] - EXTRA['adapt']['var_mean'][v][0]))
print('  NIC1 E:', {k: v for k, v in EXTRA['nic1']['e'].items()})
print('  NIC1 W:', {k: v for k, v in EXTRA['nic1']['w'].items()})
print('QP:', EXTRA['qp'])
print('spread:', EXTRA['spread'])
print('medit range:', EXTRA['medit_range'])
print('n0 vs nonmono (class):')
for cl in CLASSES:
    print('    %-24s N0=%5.1f  best nonmono=%5.1f' % (cl, EXTRA['n0_vs_nonmono'][cl][0], EXTRA['n0_vs_nonmono'][cl][1]))
print('mpx wins:', EXTRA['mpx_wins'], ' N2 rel20=%.2f' % EXTRA['n2_rel20'])
print('\nsaved r2_tables.json')
