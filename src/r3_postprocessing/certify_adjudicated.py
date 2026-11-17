# Added for the EBNPG manuscript (third revision).
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-
"""Bracket the minimiser of the common proximal subproblem at the disputed points.

The manuscript evaluates the two common criteria at the *proximal point*
p_beta(x), i.e. at the minimiser of the common subproblem

    Theta(x,d) = max_i { <grad f_i(x), d> + h_i(x+d) - h_i(x) } + 1/2 ||d||^2,
    h_i(y)     = delta * || B_i^{-T} y ||_1 ,   B_i = A{i}(1:n,:),

over the box x + d in [l,u].  At the points recorded in
adjudicated_uncertified.csv the solver ladder of recompute_common_stationarity_r4.m
did not produce a verified minimiser, so the earlier record used the *returned*
point instead; at two of them the two points differ enough to change the verdict.
Six further points (method = -1 in retry_log.csv) had no verified solution at all.

Bounds.  Theta is 1-strongly convex in d on the box and 0 is feasible, so for
the minimiser d*

    Theta(d) - Theta* >= 1/2 ||d - d*||^2 ,        Theta* <= Theta(0) = 0 .

An upper bound U = Theta(dhat) is obtained at any box-feasible dhat, by
evaluating the model itself (not the solver's objective or epigraph variable).
A lower bound comes from any weights lam in the simplex and any z_i with
||z_i||_inf <= 1: because <q_i, y> <= h_i(y) with q_i = delta_i B_i^{-1} z_i,

    Phi(lam,z) = sum_i lam_i ( <q_i, x> - h_i(x) )
                 + min_{d in box} [ 1/2||d||^2 + <a,d> ],   a = sum_i lam_i (grad f_i + q_i)

satisfies Phi <= Theta*.  Hence, with r = sqrt(2 (U - L)),

    -Theta* in [-U, -L] ,      ||d*|| in [ max(0,||dhat||-r), ||dhat||+r ] ,

and a verdict at 1e-4 is recorded only when the whole interval lies on one side
of the tolerance; otherwise the point is marked undecided (-1) and counted as
not satisfying the criterion.  As a last resort a feasible trial step d_t
certifies that *both* criteria fail: with

    C = max_i ( ||grad f_i||_2 + delta_i sum_j || row_j(B_i^{-T}) ||_2 )

one has Theta(d) >= -C ||d|| for every d, so Theta(d_t) < -max(tol, C tol)
forces -Theta* > tol and ||d*|| > tol simultaneously.

The lower bound is maximised by a cutting-plane (Kelley) method with the master
LP solved in scaled units, plus a closed-form dual point built from the
subgradient of h_i.  Every iterate is dual feasible by construction, so every L
produced is valid; a solver status of "solved" is recorded for diagnosis only
and is never used as evidence.

Usage
    python certify_adjudicated.py <qp_export_dir> <adjudicated_uncertified.csv> <out.csv>

Requires numpy, scipy and osqp (pip install osqp).
"""
import sys, os, csv
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.optimize import linprog, minimize
import osqp

TOL = 1e-4
SET = dict(verbose=False, eps_abs=1e-11, eps_rel=1e-11,
           max_iter=400000, polish=True, polish_refine_iter=10,
           adaptive_rho=True, scaling=10)


# ------------------------------------------------------------------ data
def load_points(d):
    pts = []
    for f in sorted(os.listdir(d)):
        if not f.endswith('.mat'):
            continue
        D = sio.loadmat(os.path.join(d, f), squeeze_me=True, struct_as_record=False)
        n, m = int(D['n']), int(D['m'])
        l = np.atleast_1d(D['l']).astype(float).reshape(n)
        u = np.atleast_1d(D['u']).astype(float).reshape(n)
        # with m == 1 MATLAB collapses the cell array, and squeeze_me drops
        # singleton dimensions, so every block is reshaped from dimA.
        dimA = [int(x) for x in np.atleast_1d(D['dimA']).ravel()]
        Araw, braw = D['A'], D['b']
        if m == 1:
            Araw, braw = [Araw], [braw]
        A = [np.asarray(Araw[i], dtype=float).reshape(dimA[i], n) for i in range(m)]
        b = [np.asarray(braw[i], dtype=float).reshape(dimA[i]) for i in range(m)]
        ps = D['pts']
        if isinstance(ps, sio.matlab.mat_struct):
            ps = [ps]
        for q in (ps if ps is not None else []):
            pts.append(dict(prob=f[:-4], alg=str(q.alg), run=int(q.run), n=n, m=m,
                            l=l, u=u, A=A, b=b,
                            x=np.asarray(q.x, dtype=float).reshape(n),
                            JG=np.asarray(q.JG, dtype=float).reshape(m, n),
                            H=np.asarray(q.H, dtype=float).reshape(m)))
    return pts


def prep(pt):
    n, m = pt['n'], pt['m']
    pt['B'] = [np.asarray(pt['A'][i][:n, :], dtype=float) for i in range(m)]
    # the cost vector of the L1 block is delta_i * ones
    pt['delta'] = [float(np.mean(pt['b'][i])) for i in range(m)]
    pt['a'] = pt['l'] - pt['x']
    pt['bb'] = pt['u'] - pt['x']
    return pt


def theta(d, pt):
    """exact Theta at a point d of the box (no LP, no epigraph variable)."""
    y = pt['x'] + d
    v = np.empty(pt['m'])
    for i in range(pt['m']):
        z = np.linalg.solve(pt['B'][i].T, y)
        v[i] = pt['JG'][i] @ d + pt['delta'][i] * np.abs(z).sum() - pt['H'][i]
    return float(v.max() + 0.5 * (d @ d))


def lipschitz(pt):
    """C with Theta(d) >= -C ||d|| on the box."""
    C = 0.0
    for i in range(pt['m']):
        Binv = np.linalg.inv(pt['B'][i])
        C = max(C, np.linalg.norm(pt['JG'][i]) +
                pt['delta'][i] * float(np.sum(np.linalg.norm(Binv, axis=0))))
    return C


# ------------------------------------------------------------ QP reference
def build_qp(pt):
    """The QP in z = [tau; p; u1..um], used only to produce a candidate point."""
    n, m, x, JG, H = pt['n'], pt['m'], pt['x'], pt['JG'], pt['H']
    A, b, l, u = pt['A'], pt['b'], pt['l'], pt['u']
    dimA = [len(bi) for bi in b]
    sdim = sum(dimA)
    N = 1 + n + sdim
    s = max(1.0, np.abs(JG).max(), np.abs(H).max())
    P = sp.lil_matrix((N, N))
    for k in range(n):
        P[1 + k, 1 + k] = 1.0
    q = np.zeros(N)
    q[0] = s
    q[1:1 + n] = -x
    off, t = [], 1 + n
    for di in dimA:
        off.append(t)
        t += di
    Ai = sp.lil_matrix((m, N))
    bi = np.zeros(m)
    for i in range(m):
        Ai[i, 0] = -s
        Ai[i, 1:1 + n] = JG[i, :]
        Ai[i, off[i]:off[i] + dimA[i]] = b[i]
        bi[i] = H[i] + JG[i, :] @ x
    Ae = sp.lil_matrix((n * m, N))
    for i in range(m):
        for k in range(n):
            Ae[i * n + k, 1 + k] = -1.0
        Ae[i * n:(i + 1) * n, off[i]:off[i] + dimA[i]] = A[i].T
    be = np.zeros(n * m)
    lb = np.concatenate([[-np.inf], l, np.zeros(sdim)])
    ub = np.concatenate([[np.inf], u, np.full(sdim, np.inf)])
    Astack = sp.vstack([Ai.tocsc(), Ae.tocsc(), sp.eye(N, format='csc')]).tocsc()
    lo = np.concatenate([np.full(m, -np.inf), be, lb])
    up = np.concatenate([bi, be, ub])
    return dict(P=P.tocsc(), q=q, A=Astack, lo=lo, up=up)


def qp_solve(pt, solver):
    B = build_qp(pt)
    solver.setup(P=B['P'], q=B['q'], A=B['A'], l=B['lo'], u=B['up'], **SET)
    res = solver.solve()
    if res.x is None:
        return dict(status=str(res.info.status), dfeas=np.zeros(pt['n']))
    p = res.x[1:1 + pt['n']]
    d = p - pt['x']
    if not np.all(np.isfinite(d)):
        d = np.zeros(pt['n'])
    return dict(status=str(res.info.status), dfeas=np.clip(d, pt['a'], pt['bb']))


# ------------------------------------------------------------- dual bound
def phi(lam, W, pt, want_d=False):
    n, m = pt['n'], pt['m']
    a = np.zeros(n)
    const = 0.0
    for i in range(m):
        Bi, di = pt['B'][i], pt['delta'][i]
        u = np.linalg.solve(Bi.T, pt['x'])
        a += lam[i] * (pt['JG'][i] + di * np.linalg.solve(Bi, W[:, i]))
        const += lam[i] * (di * (W[:, i] @ u) - pt['H'][i])
    d = np.clip(-a, pt['a'], pt['bb'])
    val = const + 0.5 * (d @ d) + a @ d
    if want_d:
        return val, d, a
    return val


def kelley(pt, kmax=400, need=None, extra=()):
    """maximise Phi over (lam in the simplex) x (|W|_inf <= 1) by cutting planes.

    The primal side of the bracket is the best of a candidate pool: d = 0
    (always box feasible), the QP candidate in `extra`, and the inner
    minimisers met along the way.  The master LP is solved in scaled units so
    that the interior-point tolerances act on O(1) data.
    """
    n, m = pt['n'], pt['m']
    nv = m + m * n + 1                 # lam, W, theta (scaled)
    cuts = []
    Ubest, dhat = np.inf, None
    for cand in (np.zeros(n),) + tuple(extra):
        Uc = theta(cand, pt)
        if Uc < Ubest:
            Ubest, dhat = Uc, np.array(cand, dtype=float)

    lam = np.full(m, 1.0 / m)
    W = np.zeros((n, m))
    for i in range(m):
        z = np.linalg.solve(pt['B'][i].T, pt['x'])
        W[:, i] = np.sign(z) + (z == 0)

    Lbest = -np.inf
    for _ in range(kmax):
        val, dm, a = phi(lam, W, pt, want_d=True)
        y = pt['x'] + dm
        gl = np.empty(m)
        gw = np.zeros((n, m))
        for i in range(m):
            BTy = np.linalg.solve(pt['B'][i].T, y)
            gl[i] = pt['JG'][i] @ dm + pt['delta'][i] * (W[:, i] @ BTy) - pt['H'][i]
            gw[:, i] = lam[i] * pt['delta'][i] * BTy
        sc = max(1.0, abs(val), np.abs(gl).max(),
                 np.abs(gw).max() if gw.size else 0.0)
        cuts.append((lam.copy(), W.copy(), val, gl, gw, sc))
        Lbest = max(Lbest, val)
        Ud = theta(dm, pt)
        if Ud < Ubest:
            Ubest, dhat = Ud, dm.copy()
        if need is not None and dhat is not None and need(Lbest, dhat, Ubest):
            break
        if Ubest - Lbest <= 1e-15 * max(1.0, abs(Ubest)):
            break

        A_ub, b_ub = [], []
        for (l0, W0, p0, glk, gwk, s0) in cuts:
            row = np.zeros(nv)
            row[:m] = -glk / s0
            for i in range(m):
                row[m + i * n: m + (i + 1) * n] = -gwk[:, i] / s0
            row[-1] = 1.0
            A_ub.append(row)
            b_ub.append((p0 - glk @ l0 -
                         sum(gwk[:, i] @ W0[:, i] for i in range(m))) / s0)
        A_eq = np.zeros((1, nv))
        A_eq[0, :m] = 1.0
        bounds = [(0, None)] * m + [(-1, 1)] * (m * n) + [(None, None)]
        r = linprog(np.r_[np.zeros(nv - 1), -1.0], A_ub=np.array(A_ub),
                    b_ub=np.array(b_ub), A_eq=A_eq, b_eq=[1.0],
                    bounds=bounds, method='highs')
        if not r.success:
            break
        lam_new = np.clip(r.x[:m], 0, None)
        if lam_new.sum() <= 0:
            break
        lam = lam_new / lam_new.sum()
        W = np.clip(r.x[m:m + m * n].reshape(m, n).T, -1.0, 1.0)
    return Lbest, dhat, Ubest, len(cuts)


def tangent(pt, anchor):
    """Phi at the subgradient of h_i at x + anchor, maximised over the simplex.

    With W_i = sign(B_i^{-T} y) the inequality h_i(x+d) >= delta_i <W_i, B_i^{-T}(x+d)>
    holds for every d, so Phi(lam, W) is valid for any lam; the smallest bound
    in the direction of the objective is obtained by minimising
    || sum_i lam_i (grad f_i + q_i) ||_2 over the simplex.
    """
    n, m = pt['n'], pt['m']
    y = pt['x'] + anchor
    W = np.zeros((n, m))
    A = np.zeros((n, m))
    for i in range(m):
        z = np.linalg.solve(pt['B'][i].T, y)
        W[:, i] = np.sign(z) + (z == 0)
        A[:, i] = pt['JG'][i] + pt['delta'][i] * np.linalg.solve(pt['B'][i], W[:, i])

    def f(lam):
        return 0.5 * float(np.sum((A @ lam) ** 2))

    best = None
    for seed in [np.full(m, 1.0 / m)] + [np.eye(m)[i] for i in range(m)]:
        r = minimize(f, seed, method='SLSQP', bounds=[(0, 1)] * m,
                     constraints=[{'type': 'eq', 'fun': lambda l: l.sum() - 1.0}],
                     options=dict(ftol=1e-22, maxiter=800))
        if best is None or r.fun < best.fun:
            best = r
    lam = np.clip(best.x, 0, None)
    lam /= lam.sum()
    return phi(lam, W, pt)


# -------------------------------------------------------------- verdicts
def verdict(L, U, dhat, tol=TOL):
    """bracket both criteria; 1 = satisfies, 0 = does not, -1 = undecided."""
    nrm = float(np.linalg.norm(dhat))
    r = np.sqrt(2.0 * max(0.0, U - L))
    vt = 1 if -L <= tol else (0 if -U > tol else -1)
    vd = 1 if nrm + r <= tol else (0 if nrm - r > tol else -1)
    return vt, vd, r, nrm


def main():
    qp_dir, old_csv, out_csv = sys.argv[1], sys.argv[2], sys.argv[3]
    pts = [prep(p) for p in load_points(qp_dir)]
    print('exported points: %d' % len(pts))

    ret = {}
    for r in csv.DictReader(open(old_csv)):
        ret[(r['prob'], r['alg'], int(r['run']))] = (r['theta_model'], r['dnorm'])

    solver = osqp.OSQP()
    rows = []
    for k, pt in enumerate(pts):
        key = (pt['prob'], pt['alg'], pt['run'])
        q = qp_solve(pt, solver)

        def need(L, dh, U):
            vt, vd, _, _ = verdict(L, U, dh)
            return vt >= 0 and vd >= 0

        L, dhat, U, ncut = kelley(pt, need=need, extra=(q['dfeas'],))
        method = 'gap'
        if dhat is None:
            dhat = np.zeros(pt['n'])

        # closed-form dual point, used when the cutting planes leave a gap
        for anchor in (np.zeros(pt['n']), dhat):
            try:
                L2 = tangent(pt, anchor)
            except Exception:
                continue
            if np.isfinite(L2) and L2 > L:
                L = L2
        vt, vd, r, nrm = verdict(L, U, dhat)

        # feasible trial step: certifies that both criteria fail
        if vt < 0 or vd < 0:
            C = lipschitz(pt)
            if theta(dhat, pt) < -max(TOL, C * TOL):
                vt = vd = 0
                method = 'gap+trial'

        o = ret.get(key, ('', ''))
        rows.append(dict(
            prob=pt['prob'], alg=pt['alg'], run=pt['run'],
            method=method, status=q['status'],
            theta_ret=o[0], dnorm_ret=o[1],
            U='%.12g' % U, L='%.12g' % L, radius='%.6g' % r, dhat='%.12g' % nrm,
            neg_theta_lo='%.12g' % max(0.0, -U), neg_theta_hi='%.12g' % (-L),
            dnorm_lo='%.6g' % max(0.0, nrm - r), dnorm_hi='%.6g' % (nrm + r),
            pass_theta=vt, pass_dnorm=vd, cuts=ncut))
        if (k + 1) % 20 == 0:
            print('  bracketed %d/%d' % (k + 1, len(pts)), flush=True)

    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_t = sum(r['pass_theta'] == 1 for r in rows)
    n_d = sum(r['pass_dnorm'] == 1 for r in rows)
    n_u = sum(r['pass_theta'] < 0 or r['pass_dnorm'] < 0 for r in rows)
    rad = np.array([float(r['radius']) for r in rows])
    print('pass_theta %d  pass_dnorm %d  undecided %d  (n = %d)'
          % (n_t, n_d, n_u, len(rows)))
    print('radius: median %.2e  max %.2e' % (np.median(rad), rad.max()))
    print('wrote', out_csv)


if __name__ == '__main__':
    main()
