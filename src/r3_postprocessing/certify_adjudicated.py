# Added for the EBNPG manuscript (third revision).
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-
"""Certify the verdict at the disputed points and rebuild adjudicated.csv.

The manuscript evaluates the two common criteria at the *proximal point*
p_beta(x), i.e. at the minimiser of the common subproblem

    Theta(x,d) = max_i { <grad f_i(x), d> + h_i(x+d) - h_i(x) } + 1/2 ||d||^2,
    h_i(y)     = delta * || B_i^{-T} y ||_1 ,   B_i = A{i}(1:n,:),

over the box x + d in [l,u].  At the points recorded in
adjudicated_uncertified.csv the solver ladder of
recompute_common_stationarity_r4.m did not produce a certified minimiser, so
the earlier record used the *returned* point instead.  At two of them the two
points differ enough to change the verdict.

This script solves those subproblems again, independently of the ladder, and
accepts a candidate only with an optimality certificate.  Written as a QP in
z = [tau; p; u1..um] (p = x + d, u_i >= 0 the LP variables of h_i):

    min  1/2 p'p - x'p + s*tau
    s.t. <JG_i, p> + <b_i, u_i> - H_i - <JG_i, x> <= s*tau        i = 1..m
         A_i' u_i - p = 0                                        i = 1..m
         l <= p <= u,  u_i >= 0,  tau free

with s = max(1, max|JG|, max|H|), the scaling used by the ladder.  At the
solution Theta = 1/2 ||d||^2 + s*tau.

Certificate.  Theta is 1-strongly convex in d, so for any candidate d and the
minimiser d*

    1/2 ||d - d*||^2 <= Theta(d) - Theta(d*) <= dist(0, dTheta(d)) * ||d - d*||,

and dist(0, dTheta(d)) is the p-block of the KKT residual  P z + q + A'y.
Hence ||d - d*|| <= r := ||p-block||, and

    ||d*||  in  [ max(0, ||d|| - r), ||d|| + r ]
    -Theta* in  [ max(0, -Theta), -Theta + r^2 ] .

A verdict at 1e-4 is recorded only when the whole interval lies on one side;
the present data has no undecided point.

Usage
    python certify_adjudicated.py <qp_export_dir> <adjudicated_uncertified.csv> <out.csv>

Requires numpy, scipy and osqp (pip install osqp).
"""
import sys, os, csv
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import osqp

TOL = 1e-4
SET = dict(verbose=False, eps_abs=1e-11, eps_rel=1e-11,
           max_iter=400000, polish=True, polish_refine_iter=10,
           adaptive_rho=True, scaling=10)


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
        for q in ps:
            pts.append(dict(prob=f[:-4], alg=str(q.alg), run=int(q.run), n=n, m=m,
                            l=l, u=u, A=A, b=b,
                            x=np.asarray(q.x, dtype=float).reshape(n),
                            JG=np.asarray(q.JG, dtype=float).reshape(m, n),
                            H=np.asarray(q.H, dtype=float).reshape(m)))
    return pts


def build(pt):
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
    return dict(P=P.tocsc(), q=q, A=Astack, lo=lo, up=up, s=s, off=off, dimA=dimA)


def solve(pt, solver):
    B = build(pt)
    solver.setup(P=B['P'], q=B['q'], A=B['A'], l=B['lo'], u=B['up'], **SET)
    res = solver.solve()
    z, y = res.x, res.y
    if z is None:
        return None
    n = pt['n']
    tau, p = z[0], z[1:1 + n]
    d = p - pt['x']
    th = 0.5 * d @ d + B['s'] * tau
    stat = B['P'] @ z + B['q'] + B['A'].T @ y
    r = float(np.linalg.norm(stat[1:1 + n]))
    return dict(status=str(res.info.status), r=r, theta=th, dnorm=float(np.linalg.norm(d)))


def main():
    qp_dir, old_csv, out_csv = sys.argv[1], sys.argv[2], sys.argv[3]
    pts = load_points(qp_dir)
    print('disputed points: %d' % len(pts))
    solver = osqp.OSQP()
    C = {}
    for k, pt in enumerate(pts):
        C[(pt['prob'], pt['alg'], int(pt['run']))] = solve(pt, solver)
        if (k + 1) % 40 == 0:
            print('  solved %d/%d' % (k + 1, len(pts)), flush=True)

    rows, n_t, n_d, undec = [], 0, 0, 0
    for r in csv.DictReader(open(old_csv)):
        k = (r['prob'], r['alg'], int(r['run']))
        c = C[k]
        th = max(0.0, -c['theta'])
        dn = c['dnorm']
        rad = max(c['r'], 0.0)
        lo_t, hi_t = th, th + rad * rad
        lo_d, hi_d = max(0.0, dn - rad), dn + rad
        pt_ok = 1 if hi_t <= TOL else (0 if lo_t > TOL else -1)
        pd_ok = 1 if hi_d <= TOL else (0 if lo_d > TOL else -1)
        undec += (pt_ok < 0) + (pd_ok < 0)
        n_t += pt_ok == 1
        n_d += pd_ok == 1
        rows.append(dict(prob=r['prob'], alg=r['alg'], run=r['run'],
                         ret_theta=r['theta_model'], ret_dnorm=r['dnorm'],
                         theta_star='%.12g' % (-c['theta']), dnorm_star='%.12g' % dn,
                         radius='%.6g' % rad, status=c['status'],
                         pass_theta=pt_ok, pass_dnorm=pd_ok))

    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    radius = np.array([max(C[(r['prob'], r['alg'], int(r['run']))]['r'], 0.0) for r in rows])
    print('pass_theta %d, pass_dnorm %d, undecided %d' % (n_t, n_d, undec))
    print('certificate radius: median %.2e  max %.2e' % (np.median(radius), radius.max()))
    print('wrote', out_csv)
    if undec:
        print('WARNING: undecided verdicts present; lower the tolerance of the solver.')


if __name__ == '__main__':
    main()
