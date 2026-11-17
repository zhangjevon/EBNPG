# Added for the EBNPG manuscript (third revision).
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-
"""Compare the rebuilt EBNPG-W results against the stored ones.

For every problem present in BOTH folders and every variant N0..N3,N5..N7 it
compares, in increasing order of strictness,
  (1) success rate over the 100 runs,
  (2) the medians (iter / ngev / nhev) over the successful runs,
  (3) the per-run info flag vector,
  (4) the per-run iteration / ngev / nhev vectors.
(1)-(2) are what the manuscript reports; (3)-(4) prove the reconstruction.
CPU time is reported but never compared (hardware dependent).
"""
import os
import sys
import glob
import numpy as np
import scipy.io as sio

_BASE = os.environ.get('EBPGM_DATA') or os.path.dirname(os.path.abspath(__file__))
BASE = _BASE
OLD = os.path.join(BASE, 'results_weighted')
NEW = os.path.join(BASE, 'results_w_rebuilt')
VARIANTS = ['N0', 'N1', 'N2', 'N3', 'N5', 'N6', 'N7']


def unwrap(v):
    while isinstance(v, np.ndarray) and v.dtype == object and v.size == 1:
        v = v.flat[0]
    return v


def vec(s, f):
    return np.asarray(unwrap(s[f]), dtype=float).ravel()


def sca(s, f):
    return float(np.asarray(unwrap(s[f])).squeeze())


def load(path):
    d = sio.loadmat(path)
    pr = d['prob_results']
    out = {}
    for k in range(pr.shape[0]):
        s = pr[k, 0]
        if s.size == 0 or not (s.dtype == object or s.dtype.names is not None):
            continue
        a = np.asarray(unwrap(s['algorithm'])).ravel()
        if a.size == 0:
            continue
        out[str(a[0])] = s
    return out


def rows(path):
    return load(path)


def main():
    oldf = sorted(glob.glob(os.path.join(OLD, '*_W_results.mat')))
    problems = sorted(os.path.basename(p)[:-len('_W_results.mat')] for p in oldf)
    nfail = 0
    ncmp = 0
    for p in problems:
        newp = os.path.join(NEW, p + '_W_results.mat')
        if not os.path.exists(newp):
            continue
        A = rows(os.path.join(OLD, p + '_W_results.mat'))
        B = rows(newp)
        print('=' * 78)
        print(p)
        print('%-13s %6s %6s | %8s %8s | %8s %8s | %8s %8s | %s' %
              ('variant', 'SR old', 'SR new', 'it old', 'it new',
               'ngev old', 'ngev new', 'nhev old', 'nhev new', 'per-run'))
        for v in VARIANTS:
            ka = 'EBNPG-W-' + v
            if ka not in A or ka not in B:
                print('  %-11s MISSING' % ka)
                continue
            a, b = A[ka], B[ka]
            sra, srb = sca(a, 'success_rate'), sca(b, 'success_rate')
            ia, ib = vec(a, 'info_vals'), vec(b, 'info_vals')
            ma = ia == 0
            mb = ib == 0
            oka = ma.sum() > 0
            it_a = np.median(vec(a, 'iter_vals')[ma]) if oka else float('nan')
            it_b = np.median(vec(b, 'iter_vals')[mb]) if mb.sum() > 0 else float('nan')
            ng_a = np.median(vec(a, 'ngev_vals')[ma]) if oka else float('nan')
            ng_b = np.median(vec(b, 'ngev_vals')[mb]) if mb.sum() > 0 else float('nan')
            nh_a = np.median(vec(a, 'nhev_vals')[ma]) if oka else float('nan')
            nh_b = np.median(vec(b, 'nhev_vals')[mb]) if mb.sum() > 0 else float('nan')

            # strict per-run identity
            exact = np.array_equal(ia, ib)
            if exact:
                for f in ('iter_vals', 'ngev_vals', 'nhev_vals'):
                    if not np.array_equal(vec(a, f), vec(b, f)):
                        exact = False
                        break
            tag = 'EXACT' if exact else ('flags' if np.array_equal(ia, ib) else 'diff')
            ncmp += 1
            ok = (sra == srb) and abs(it_a - it_b) < 1e-9 and \
                 abs(ng_a - ng_b) < 1e-9 and abs(nh_a - nh_b) < 1e-9
            if not ok:
                nfail += 1
            print('%-13s %6.0f %6.0f | %8.1f %8.1f | %8.1f %8.1f | %8.1f %8.1f | %s%s' %
                  (ka, sra, srb, it_a, it_b, ng_a, ng_b, nh_a, nh_b, tag,
                   '' if ok else '   <<< MISMATCH'))
        ta = np.median([sca(A['EBNPG-W-' + v], 'med_time')
                        for v in VARIANTS if 'EBNPG-W-' + v in A]) if A else float('nan')
        tb = np.median([sca(B['EBNPG-W-' + v], 'med_time')
                        for v in VARIANTS if 'EBNPG-W-' + v in B]) if B else float('nan')
        print('  median CPU over variants: stored %.3f s, rebuilt %.3f s' % (ta, tb))
    print('=' * 78)
    print('compared %d variant-rows, %d with a summary mismatch' % (ncmp, nfail))


if __name__ == '__main__':
    main()
