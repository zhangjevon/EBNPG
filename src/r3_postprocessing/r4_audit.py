# Added for the EBNPG manuscript (third revision).
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-
"""Audit of the points recovered in the R3 re-solve (crit_out_r4).

For each point the first pass left undecided, report
  * the certified value and the definition gap |Theta_def - Theta_qp|,
  * whether the decision at tolerance 1e-4 is robust, i.e. whether the value
    lies at least a factor MARGIN away from the threshold,
  * a cross-check against the inequality ||d_1||^2 <= 2(-Theta_1) used in the
    manuscript: if ||d_1|| > sqrt(2e-4) the criticality criterion cannot be
    satisfied, whatever the value of Theta.
"""
import os, sys, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'Code_Data', 'crit_out_r4', 'retry_log.csv')
TOL = 1e-4
MARGIN = 10.0          # decision called robust when |value - tol| exceeds this factor

rows = list(csv.DictReader(open(LOG)))
print('recovered points : %d' % len(rows))
meth = {}
for r in rows:
    meth[r['method']] = meth.get(r['method'], 0) + 1
print('by ladder stage  : %s' % meth)

cert = [r for r in rows if r['cert'] == '1']
uns = [r for r in rows if r['method'] == '-1']
print('certified        : %d' % len(cert))
print('unsolved         : %d' % len(uns))
if uns:
    print('   ' + ', '.join('%s/%s/%s' % (r['prob'], r['alg'], r['run']) for r in uns))
print()

# decision buckets
buckets = {'pass_robust': [], 'pass_marginal': [], 'fail_marginal': [], 'fail_robust': [], 'undecided': []}
for r in rows:
    if r['method'] == '-1':
        buckets['undecided'].append(r)
        continue
    v = abs(float(r['theta_qp']))          # = -Theta_qp
    if v <= TOL / MARGIN:
        buckets['pass_robust'].append(r)
    elif v <= TOL:
        buckets['pass_marginal'].append(r)
    elif v <= TOL * MARGIN:
        buckets['fail_marginal'].append(r)
    else:
        buckets['fail_robust'].append(r)

print('decision margin (factor %g around the 1e-4 threshold)' % MARGIN)
for k in ('pass_robust', 'pass_marginal', 'fail_marginal', 'fail_robust', 'undecided'):
    print('  %-14s %4d' % (k, len(buckets[k])))
print()

print('--- points within a factor %g of the threshold (need the cross-check)' % MARGIN)
print('%-6s %-13s %4s %4s %13s %13s %10s %4s %s' %
      ('prob', 'alg', 'run', 'info', 'theta_qp', 'theta_def', 'dnorm', 'cert', 'ineq'))
for k in ('pass_marginal', 'fail_marginal'):
    for r in sorted(buckets[k], key=lambda r: -abs(float(r['theta_qp']))):
        d = float(r['dnorm'])
        ineq = 'violated' if d * d > 2 * TOL and float(r['theta_qp']) > -TOL else 'ok'
        print('%-6s %-13s %4s %4s %13.6e %13.6e %10.3e %4s %s' %
              (r['prob'], r['alg'], r['run'], r['info_orig'],
               float(r['theta_qp']), float(r['theta_def']), d, r['cert'], ineq))
print()

# global inequality screen on the recovered set
bad = [r for r in rows if r['method'] != '-1'
       and float(r['dnorm']) ** 2 > 2 * TOL and float(r['theta_qp']) > -TOL]
print('inequality violations over the recovered set '
      '(||d||^2 > 2e-4 yet -Theta <= 1e-4): %d' % len(bad))
for r in bad:
    print('   %s/%s/%s  -Theta=%.4e  ||d||=%.4e' %
          (r['prob'], r['alg'], r['run'], float(r['theta_qp']), float(r['dnorm'])))
