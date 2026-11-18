# Added for the EBNPG manuscript (third revision).
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-
"""Decide the common proximal criteria using exact fixed-witness bounds.

The model at a stored final point x is

    Phi_x(d) = max_i [JG_i d + h_i(x+d) - h_i(x)] + ||d||^2/2,
    h_i(y) = delta_i ||B_i^{-T}y||_1,  l <= x+d <= u.

Candidate generators include the existing QP, Kelley and tangent searches,
the box-feasible point p=0, projected dual vectors and, when needed, a concave
dual search in (lambda,Y=lambda*W). Solver states and candidate objectives
are diagnostic only. They are never used as optimality certificates.

Every fixed witness is checked using Fraction arithmetic. Exported binary64
x, JG, B, delta and box coefficients are interpreted as exact rational input
data; h_i(x) is recomputed from the support-function definition. The feasible
primal direction gives U=Phi_x(dhat). For lambda in the simplex and |W|<=1,
q_i=delta_i B_i^-1 W_i gives a lower bound L on min Phi_x. Strong convexity
then gives ||dhat-d*||<=sqrt(2(U-L)). Square-root endpoints are enclosed by
integer squaring, and decimal endpoints are rounded outwards.

These certificates bound the exported numerical subproblem. They do not
claim exact evaluation of transcendental derivatives before their binary64
export, nor individual certification of the inherited first-pass outputs.

A supporting-vector certificate at p=0 additionally verifies both activity
in the original maximum and feasibility in the unit cube. Large-value cases
may use a feasible-trial/Lipschitz lower bound on ||d*||. All 134 expected
keys must be covered. An undecided criterion aborts before final output is
replaced; it is never counted as a failed stationarity test.

Usage:
    python certify_adjudicated.py qp_export old_csv out_csv [witness_json]

The CSV and self-contained witness JSON are bound by a SHA-256 digest.
Requires numpy, scipy and osqp. Exact verification uses Python's standard
library fractions and integer arithmetic.
"""
import sys, os, csv, json, hashlib, math
from fractions import Fraction
from pathlib import Path
from functools import lru_cache
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
                            export_path=os.path.join(d, f),
                            l=l, u=u, A=A, b=b,
                            x=np.asarray(q.x, dtype=float).reshape(n),
                            JG=np.asarray(q.JG, dtype=float).reshape(m, n),
                            H=np.asarray(q.H, dtype=float).reshape(m)))
    return pts


def prep(pt):
    n, m = pt['n'], pt['m']
    pt['B'] = [np.asarray(pt['A'][i][:n, :], dtype=float) for i in range(m)]
    # the cost vector of the L1 block is delta_i * ones
    pt['delta'] = []
    for bi in pt['b']:
        if not np.all(bi == bi[0]):
            raise ValueError('support block must use a constant delta')
        pt['delta'].append(float(bi[0]))
    pt['a'] = pt['l'] - pt['x']
    pt['bb'] = pt['u'] - pt['x']
    return pt


def theta(d, pt):
    """Floating model evaluation used only to generate candidate witnesses."""
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
    """Produce a feasible primal candidate and raw multipliers, never a certificate."""
    B = build_qp(pt)
    solver.setup(P=B['P'], q=B['q'], A=B['A'], l=B['lo'], u=B['up'], **SET)
    result = solver.solve()
    direction = np.zeros(pt['n'])
    if result.x is not None:
        candidate = np.asarray(result.x[1:1 + pt['n']], dtype=float)
        if np.all(np.isfinite(candidate)):
            direction = np.clip(candidate, pt['l'], pt['u']) - pt['x']
    dual_y = None
    required = pt['m'] + pt['m'] * pt['n']
    if result.y is not None and np.all(np.isfinite(result.y[:required])):
        dual_y = np.asarray(result.y, dtype=float).copy()
    return dict(status=str(result.info.status), dfeas=direction, dual_y=dual_y)


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
    best_dual = None
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
        if val > Lbest:
            Lbest = val
            best_dual = (lam.copy(), W.copy(), 'kelley')
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
    return Lbest, dhat, Ubest, len(cuts), best_dual


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
    return phi(lam, W, pt), (lam.copy(), W.copy(), 'tangent')



# ------------------------------------------------------ exact fixed witnesses
# The certificate treats every exported binary64 coefficient as its exact
# rational value.  Support values H_i=h_i(x) are recomputed from B_i, delta_i
# and x rather than inheriting a rounded LP/support value.

def rational(value):
    if isinstance(value, Fraction):
        return value
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite witness coefficient")
    return Fraction.from_float(number)


def dot_exact(left, right):
    return sum((a * b for a, b in zip(left, right)), Fraction(0))


def matvec_exact(matrix, vector):
    return [dot_exact(row, vector) for row in matrix]


def transpose_exact(matrix):
    return list(map(list, zip(*matrix)))


@lru_cache(maxsize=64)
def inverse_exact(rows):
    """Gauss-Jordan inverse of the exact rational input matrix, cached per block."""
    dimension = len(rows)
    table = [
        list(row) + [Fraction(int(i == j)) for j in range(dimension)]
        for i, row in enumerate(rows)
    ]
    for column in range(dimension):
        pivot = max(range(column, dimension), key=lambda i: abs(table[i][column]))
        if table[pivot][column] == 0:
            raise ValueError("singular support matrix")
        table[column], table[pivot] = table[pivot], table[column]
        divisor = table[column][column]
        table[column] = [entry / divisor for entry in table[column]]
        for row in range(dimension):
            if row == column:
                continue
            factor = table[row][column]
            if factor:
                table[row] = [
                    entry - factor * pivot_entry
                    for entry, pivot_entry in zip(table[row], table[column])
                ]
    return tuple(tuple(row[dimension:]) for row in table)


def sqrt_interval(value, digits=50):
    """Rational outward bounds for sqrt(value), proved by integer squaring."""
    if value < 0:
        raise ArithmeticError("negative exact square")
    scale = 10 ** digits
    numerator = value.numerator * scale * scale
    integer = math.isqrt(numerator // value.denominator)
    lower = Fraction(integer, scale)
    if integer * integer * value.denominator == numerator:
        return lower, lower
    return lower, Fraction(integer + 1, scale)


def decimal_outward(value, upper, digits=40):
    """A fixed decimal endpoint rounded in the requested direction using integers."""
    scale = 10 ** digits
    scaled_numerator = value.numerator * scale
    integer = scaled_numerator // value.denominator
    if upper and scaled_numerator % value.denominator:
        integer += 1
    sign = "-" if integer < 0 else ""
    whole, fraction = divmod(abs(integer), scale)
    return "%s%d.%0*d" % (sign, whole, digits, fraction)


def encoded_floats(array):
    if isinstance(array, np.ndarray):
        return [encoded_floats(value) for value in array.tolist()]
    if isinstance(array, (list, tuple)):
        return [encoded_floats(value) for value in array]
    return float(array).hex()


class ExactModel:
    """Verify feasible witnesses and retain the strongest exact primal/dual pair."""

    def __init__(self, point):
        self.point = point
        self.n, self.m = point["n"], point["m"]
        self.x = list(map(rational, point["x"]))
        self.gradients = [list(map(rational, row)) for row in point["JG"]]
        self.lower = list(map(rational, point["l"]))
        self.upper = list(map(rational, point["u"]))
        self.delta = list(map(rational, point["delta"]))
        if any(value < 0 for value in self.delta):
            raise ValueError("support thresholds must be nonnegative")
        if any(lo > hi for lo, hi in zip(self.lower, self.upper)):
            raise ValueError("invalid box bounds")
        self.matrices = [
            tuple(tuple(map(rational, row)) for row in matrix)
            for matrix in point["B"]
        ]
        self.inverses = [inverse_exact(matrix) for matrix in self.matrices]
        self.shifted_lower = [lo - x for lo, x in zip(self.lower, self.x)]
        self.shifted_upper = [hi - x for hi, x in zip(self.upper, self.x)]
        self.H = [self.support(i, self.x) for i in range(self.m)]
        self.U, self.L = None, None
        self.direction = None
        self.lam, self.W = None, None
        self.upper_source, self.lower_source = None, None
        self.origin = None
        self.trial = None

    def support(self, objective, location):
        transformed = matvec_exact(transpose_exact(self.inverses[objective]), location)
        return self.delta[objective] * sum(map(abs, transformed), Fraction(0))

    def clip_direction(self, direction):
        return [
            min(max(rational(value), lo), hi)
            for value, lo, hi in zip(direction, self.shifted_lower, self.shifted_upper)
        ]

    def primal_value(self, direction):
        location = [x + d for x, d in zip(self.x, direction)]
        values = [
            dot_exact(gradient, direction) + self.support(i, location) - self.H[i]
            for i, gradient in enumerate(self.gradients)
        ]
        return max(values) + dot_exact(direction, direction) / 2

    def add_primal(self, direction, source):
        direction = self.clip_direction(direction)
        value = self.primal_value(direction)
        if self.U is None or value < self.U:
            self.U = value
            self.direction = direction
            self.upper_source = source

    def feasible_dual(self, lam, W):
        weights = [max(Fraction(0), rational(value)) for value in lam]
        total = sum(weights, Fraction(0))
        if total <= 0:
            raise ValueError("dual weights have zero mass")
        weights = [weight / total for weight in weights]
        cube = [
            [min(Fraction(1), max(Fraction(-1), rational(W[j][i])))
             for i in range(self.m)]
            for j in range(self.n)
        ]
        if sum(weights, Fraction(0)) != 1:
            raise ArithmeticError("simplex normalisation failed")
        return weights, cube

    def dual_value(self, weights, cube):
        coefficients = [Fraction(0)] * self.n
        constant = Fraction(0)
        for i in range(self.m):
            vector = self.delta[i]
            support_vector = [
                vector * value
                for value in matvec_exact(self.inverses[i], [cube[j][i] for j in range(self.n)])
            ]
            coefficients = [
                a + weights[i] * (gradient + support)
                for a, gradient, support in zip(coefficients, self.gradients[i], support_vector)
            ]
            constant += weights[i] * (dot_exact(support_vector, self.x) - self.H[i])
        inner_direction = self.clip_direction([-a for a in coefficients])
        value = (constant + dot_exact(coefficients, inner_direction)
                 + dot_exact(inner_direction, inner_direction) / 2)
        return value, inner_direction

    def add_dual(self, lam, W, source):
        weights, cube = self.feasible_dual(lam, W)
        value, inner_direction = self.dual_value(weights, cube)
        if self.L is None or value > self.L:
            self.L, self.lam, self.W = value, weights, cube
            self.lower_source = source
        self.add_primal(inner_direction, source + "_inner_primal")

    def add_origin_candidates(self):
        if any(lo > 0 or hi < 0 for lo, hi in zip(self.lower, self.upper)):
            return []
        origin_direction = [-x for x in self.x]
        self.add_primal(origin_direction, "origin")
        objective_values = [
            -dot_exact(gradient, self.x) - h
            for gradient, h in zip(self.gradients, self.H)
        ]
        maximum = max(objective_values)
        float_seeds = []
        for i in range(self.m):
            if self.delta[i] <= 0:
                continue
            target = [x - g for x, g in zip(self.x, self.gradients[i])]
            raw_cube = [
                value / self.delta[i]
                for value in matvec_exact(self.matrices[i], target)
            ]
            cube = [[Fraction(0)] * self.m for _ in range(self.n)]
            for j in range(self.n):
                cube[j][i] = min(Fraction(1), max(Fraction(-1), raw_cube[j]))
            weights = [Fraction(int(k == i)) for k in range(self.m)]
            source = "origin_objective_%d" % (i + 1)
            self.add_dual(weights, cube, source)
            active = objective_values[i] == maximum
            in_cube = max(map(abs, raw_cube)) <= 1
            if active and in_cube:
                self.origin = dict(
                    active_objective=i + 1,
                    objective_values=[str(value) for value in objective_values],
                    supporting_cube=[str(value) for value in raw_cube],
                    z_maxnorm=str(max(map(abs, raw_cube))),
                    exact=True,
                )
            float_seeds.append((
                np.array(list(map(float, weights))),
                np.array([[float(value) for value in row] for row in cube]),
                source,
            ))
        return float_seeds

    def endpoints(self):
        if self.U is None or self.L is None:
            return None
        gap = self.U - self.L
        if gap < 0:
            raise ArithmeticError("negative exact primal-dual gap")
        norm_squared = dot_exact(self.direction, self.direction)
        radius_squared = 2 * gap
        nlo, nhi = sqrt_interval(norm_squared)
        rlo, rhi = sqrt_interval(radius_squared)
        dlo, dhi = max(Fraction(0), nlo - rhi), nhi + rhi
        if self.trial is not None:
            dlo = max(dlo, self.trial["norm_lower"])
        return dict(
            U=self.U, L=self.L, gap=gap,
            radius_squared=radius_squared,
            radius_lo=rlo, radius_hi=rhi,
            norm_squared=norm_squared, norm_lo=nlo, norm_hi=nhi,
            neg_theta_lo=-self.U, neg_theta_hi=-self.L,
            dnorm_lo=dlo, dnorm_hi=dhi,
        )

    def classify(self, tol=Fraction(1, 10000)):
        bounds = self.endpoints()
        if bounds is None:
            return -1, -1
        criticality = (1 if bounds["neg_theta_hi"] <= tol
                       else 0 if bounds["neg_theta_lo"] > tol else -1)
        displacement = (1 if bounds["dnorm_hi"] <= tol
                        else 0 if bounds["dnorm_lo"] > tol else -1)
        return criticality, displacement

    def add_trial_certificate(self):
        if self.U is None or self.U >= 0:
            return
        constant = Fraction(0)
        for i in range(self.m):
            _, grad_upper = sqrt_interval(dot_exact(self.gradients[i], self.gradients[i]))
            row_norm_sum = Fraction(0)
            for row in transpose_exact(self.inverses[i]):
                _, row_upper = sqrt_interval(dot_exact(row, row))
                row_norm_sum += row_upper
            constant = max(constant, grad_upper + self.delta[i] * row_norm_sum)
        if constant > 0 and self.U < -max(Fraction(1, 10000), constant / 10000):
            self.trial = dict(
                U=self.U, lipschitz_upper=constant,
                norm_lower=-self.U / constant,
                direction=self.direction.copy(),
            )

    def witness(self):
        bounds = self.endpoints()
        record = dict(
            prob=self.point["prob"], alg=self.point["alg"], run=int(self.point["run"]),
            upper_source=self.upper_source, lower_source=self.lower_source,
            primal_direction=[str(value) for value in self.direction],
            lam=[str(value) for value in self.lam],
            W=[[str(value) for value in row] for row in self.W],
            U_fraction=str(self.U), L_fraction=str(self.L),
            gap_fraction=str(self.U - self.L),
            model=dict(
                x_hex=encoded_floats(self.point["x"]),
                gradients_hex=encoded_floats(self.point["JG"]),
                B_hex=encoded_floats(self.point["B"]),
                delta_hex=encoded_floats(self.point["delta"]),
                lower_hex=encoded_floats(self.point["l"]),
                upper_hex=encoded_floats(self.point["u"]),
                H_fraction=[str(value) for value in self.H],
                H_export_hex=encoded_floats(self.point["H"]),
            ),
            origin=self.origin,
        )
        if self.point.get("export_path"):
            export = Path(self.point["export_path"])
            record["export_sha256"] = hashlib.sha256(export.read_bytes()).hexdigest()
        if self.trial:
            record["trial"] = dict(
                U_fraction=str(self.trial["U"]),
                lipschitz_upper_fraction=str(self.trial["lipschitz_upper"]),
                norm_lower_fraction=str(self.trial["norm_lower"]),
                direction=[str(value) for value in self.trial["direction"]],
            )
        return record


def projected_osqp_dual(point, dual_y):
    if dual_y is None:
        return None
    n, m = point["n"], point["m"]
    raw_weights = np.maximum(np.asarray(dual_y[:m], dtype=float), 0.0)
    total = raw_weights.sum()
    if not np.isfinite(total) or total <= 0:
        return None
    weights = raw_weights / total
    nu = np.asarray(dual_y[m:m + m * n], dtype=float).reshape(m, n)
    cube = np.zeros((n, m))
    for i in range(m):
        if raw_weights[i] > 0 and point["delta"][i] > 0:
            cube[:, i] = np.clip(
                -(point["B"][i] @ nu[i]) / point["delta"][i] / raw_weights[i],
                -1.0, 1.0,
            )
    if not np.all(np.isfinite(cube)):
        return None
    return weights, cube, "projected_osqp_dual"


def refine_dual(point, seeds):
    """Concave dual search in (lambda,Y=lambda*W); final bounds are re-evaluated exactly."""
    n, m = point["n"], point["m"]
    delta = np.asarray(point["delta"])
    inverses = [np.linalg.solve(B, np.eye(n)) for B in point["B"]]
    transformed_x = np.array([delta[i] * inverses[i].T @ point["x"] for i in range(m)])
    constraint_rows = []
    for i in range(m):
        for j in range(n):
            for sign in (-1.0, 1.0):
                row = np.zeros(m + m * n)
                row[i] = -1
                row[m + i * n + j] = sign
                constraint_rows.append(row)
    matrix = np.array(constraint_rows)

    def objective(vector):
        weights = vector[:m]
        Y = vector[m:].reshape(m, n)
        a = weights @ point["JG"]
        for i in range(m):
            a += delta[i] * inverses[i] @ Y[i]
        direction = np.clip(-a, point["a"], point["bb"])
        value = (np.sum(Y * transformed_x) - weights @ point["H"]
                 + 0.5 * direction @ direction + a @ direction)
        location = point["x"] + direction
        gradient_weights = point["JG"] @ direction - point["H"]
        gradient_Y = np.array([delta[i] * inverses[i].T @ location for i in range(m)])
        return -float(value), -np.r_[gradient_weights, gradient_Y.ravel()]

    constraints = [
        {"type": "eq", "fun": lambda vector: vector[:m].sum() - 1,
         "jac": lambda vector: np.r_[np.ones(m), np.zeros(m * n)]},
        {"type": "ineq", "fun": lambda vector: -(matrix @ vector),
         "jac": lambda vector: -matrix},
    ]
    starts = []
    for weights, cube, _ in seeds:
        starts.append(np.r_[weights, (weights[:, None] * cube.T).ravel()])
    starts.append(np.r_[np.full(m, 1 / m), np.zeros(m * n)])
    best = None
    for start in starts:
        result = minimize(
            lambda vector: objective(vector)[0], start,
            jac=lambda vector: objective(vector)[1],
            method="SLSQP", constraints=constraints,
            bounds=[(0, 1)] * m + [(-1, 1)] * (m * n),
            options=dict(ftol=1e-18, maxiter=2500),
        )
        if not np.all(np.isfinite(result.x)):
            continue
        raw = np.maximum(result.x[:m], 0)
        if raw.sum() <= 0:
            continue
        weights = raw / raw.sum()
        Y = result.x[m:].reshape(m, n)
        cube = np.zeros((n, m))
        for i in range(m):
            if raw[i] > 0:
                cube[:, i] = np.clip(Y[i] / raw[i], -1, 1)
        value = phi(weights, cube, point)
        if best is None or value > best[0]:
            best = value, weights, cube
    if best is None:
        return None
    return best[1], best[2], "concave_dual_refinement"

# ------------------------------------------------------------- final record
def candidate_decided(L, direction, U):
    if not np.isfinite(L) or not np.isfinite(U) or L > U:
        return False
    radius = math.sqrt(2 * (U - L))
    norm = np.linalg.norm(direction)
    criticality = (-L <= TOL) or (-U > TOL)
    displacement = (norm + radius <= TOL) or (norm - radius > TOL)
    return criticality and displacement


def classify_point(point, solver):
    """Candidate solves stop once exact feasible witnesses decide both criteria."""
    model = ExactModel(point)
    model.add_primal(np.zeros(point["n"]), "zero_step")
    seeds = model.add_origin_candidates()
    status, cuts = "analytic_origin", 0
    if -1 not in model.classify():
        return model, status, cuts

    qp = qp_solve(point, solver)
    status = qp["status"]
    model.add_primal(qp["dfeas"], "osqp_candidate")
    dual = projected_osqp_dual(point, qp["dual_y"])
    if dual is not None:
        seeds.append(dual)
        model.add_dual(*dual)
    model.add_trial_certificate()
    if -1 not in model.classify():
        return model, status, cuts

    # Retain every old candidate source.  Its finite-precision objective is
    # used only for search, then the fixed witnesses are re-evaluated exactly.
    _, direction, _, cuts, dual = kelley(
        point, need=candidate_decided,
        extra=(qp["dfeas"], -point["x"]) if np.all(point["l"] <= 0) and np.all(point["u"] >= 0)
        else (qp["dfeas"],),
    )
    if direction is not None:
        model.add_primal(direction, "kelley_primal")
    if dual is not None:
        seeds.append(dual)
        model.add_dual(*dual)
    for anchor in (np.zeros(point["n"]), np.array(list(map(float, model.direction)))):
        try:
            _, tangent_dual = tangent(point, anchor)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            continue
        seeds.append(tangent_dual)
        model.add_dual(*tangent_dual)
    model.add_trial_certificate()
    if -1 not in model.classify():
        return model, status, cuts

    # A convex feasible domain for (lambda,Y) gives a better candidate without
    # treating an optimiser status or a returned objective as a proof.
    ordered_seeds = sorted(seeds, key=lambda item: phi(item[0], item[1], point), reverse=True)
    refined = refine_dual(point, ordered_seeds[:3])
    if refined is not None:
        model.add_dual(*refined)
    model.add_trial_certificate()
    if -1 in model.classify():
        key = point["prob"], point["alg"], point["run"]
        raise RuntimeError("undecided exact common criterion at %r" % (key,))
    return model, status, cuts


def expected_records(old_csv):
    original = {}
    with open(old_csv, newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            key = row["prob"], row["alg"], int(row["run"])
            if key in original:
                raise ValueError("duplicate old point %r" % (key,))
            original[key] = row["theta_model"], row["dnorm"]
    expected = set(original)
    retry = Path(old_csv).with_name("retry_log.csv")
    with retry.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if int(row["method"]) == -1:
                expected.add((row["prob"], row["alg"], int(row["run"])))
    if len(expected) != 134:
        raise ValueError("expected exactly 134 disputed points, got %d" % len(expected))
    return original, expected


def main():
    if len(sys.argv) not in (4, 5):
        raise SystemExit(
            "Usage: certify_adjudicated.py qp_export old_csv out_csv [witness_json]"
        )
    qp_dir, old_csv, out_csv = sys.argv[1:4]
    witness_path = (Path(sys.argv[4]) if len(sys.argv) == 5
                    else Path(out_csv).with_name("adjudication_witnesses.json"))
    points = [prep(point) for point in load_points(qp_dir)]
    old_records, expected = expected_records(old_csv)
    keys = [(point["prob"], point["alg"], point["run"]) for point in points]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("exported point keys do not match the 134 expected records")
    print("exported points: %d" % len(points), flush=True)
    rows, witnesses = [], []
    solver = osqp.OSQP()
    for index, point in enumerate(points, 1):
        model, status, cuts = classify_point(point, solver)
        bounds = model.endpoints()
        criticality, displacement = model.classify()
        key = point["prob"], point["alg"], point["run"]
        old_theta, old_direction = old_records.get(key, ("", ""))
        row = dict(
            prob=point["prob"], alg=point["alg"], run=point["run"],
            method="exact-gap+trial" if model.trial else "exact-gap",
            status=status, theta_ret=old_theta, dnorm_ret=old_direction,
            U=decimal_outward(bounds["U"], True),
            L=decimal_outward(bounds["L"], False),
            radius=decimal_outward(bounds["radius_hi"], True),
            dhat=decimal_outward(bounds["norm_hi"], True),
            neg_theta_lo=decimal_outward(bounds["neg_theta_lo"], False),
            neg_theta_hi=decimal_outward(bounds["neg_theta_hi"], True),
            dnorm_lo=decimal_outward(bounds["dnorm_lo"], False),
            dnorm_hi=decimal_outward(bounds["dnorm_hi"], True),
            pass_theta=criticality, pass_dnorm=displacement, cuts=cuts,
            lower_source=model.lower_source,
            exact_origin=int(model.origin is not None),
        )
        witness = model.witness()
        witness["pass_theta"], witness["pass_dnorm"] = criticality, displacement
        rows.append(row)
        witnesses.append(witness)
        print(
            "%3d/134 %-7s %-13s %3d  %d/%d  U=% .6e L=% .6e r=%.3e  %s"
            % (index, point["prob"], point["alg"], point["run"],
               criticality, displacement, float(model.U), float(model.L),
               float(bounds["radius_hi"]), model.lower_source), flush=True,
        )

    passed_criticality = sum(row["pass_theta"] == 1 for row in rows)
    passed_displacement = sum(row["pass_dnorm"] == 1 for row in rows)
    if any(row["pass_theta"] not in (0, 1) or row["pass_dnorm"] not in (0, 1) for row in rows):
        raise RuntimeError("an undecided point must not enter the final record")
    output = Path(out_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    document = dict(
        schema_version=1, certificate="exact-rational-primal-dual",
        csv_sha256=digest,
        interpretation=(
            "Exported binary64 x,JG,B,delta,box entries are exact rational model "
            "coefficients. H_i is recomputed exactly as delta*||B_i^-T*x||_1. "
            "All inequalities use exact rational arithmetic, with square-root "
            "endpoints enclosed by integer squaring at 50 decimal places."
        ),
        records=witnesses,
    )
    witness_path.parent.mkdir(parents=True, exist_ok=True)
    witness_temporary = witness_path.with_suffix(witness_path.suffix + ".tmp")
    witness_temporary.write_text(json.dumps(document, indent=2), encoding="utf8")
    temporary.replace(output)
    witness_temporary.replace(witness_path)
    print(
        "FINAL criticality=%d displacement=%d undecided=0 records=%d"
        % (passed_criticality, passed_displacement, len(rows)), flush=True,
    )
    print("CSV sha256 %s" % digest, flush=True)
    print("wrote %s and %s" % (output, witness_path), flush=True)


if __name__ == "__main__":
    main()

