# -*- coding: utf-8 -*-
"""Shared, fail-closed reader for common-stationarity certificates.

Certified MAT values are used directly.  Each remaining point must have exactly
one adjudication row, and its interval must decide the requested tolerance.
An unresolved point is an error, never an unsuccessful optimization run.
MAT files without a ``cert`` field remain readable: their finite values are
treated as the legacy certified values, but missing values still need evidence.
"""
import csv
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import scipy.io as sio


class AdjudicationError(ValueError):
    """Incomplete, inconsistent, or undecided stationarity evidence."""


def _decimal(value, label):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise AdjudicationError('%s is not a decimal: %r' % (label, value)) from exc
    if not result.is_finite():
        raise AdjudicationError('%s must be finite, got %r' % (label, value))
    return result


def _verdict(lo, hi, tolerance, label):
    if hi <= tolerance:
        return True
    if lo > tolerance:
        return False
    raise AdjudicationError('%s: interval [%s, %s] does not decide <= %s'
                            % (label, lo, hi, tolerance))


def _matrix(value, shape, label):
    result = np.asarray(value, dtype=float)
    if result.shape != shape:
        raise AdjudicationError('%s has shape %s, expected %s'
                                % (label, result.shape, shape))
    return result


class CommonCriteria:
    """Load all final-point values and validate complete adjudication coverage."""

    def __init__(self, crit_dir, problems, algorithms, runs=100,
                 expected_uncertified=134):
        self.directory = Path(crit_dir).resolve()
        self.problems = list(problems)
        self.algorithms = list(algorithms)
        self.runs = runs
        shape = (len(problems), len(algorithms), runs)
        self.criticality = np.full(shape, np.nan)
        self.displacement = np.full(shape, np.nan)
        self.info = np.full(shape, np.nan)
        self.certified = np.zeros(shape, dtype=bool)
        has_cert = []
        for pi, problem in enumerate(self.problems):
            path = self.directory / (problem + '_crit.mat')
            obj = sio.loadmat(path, squeeze_me=True, struct_as_record=False)['out']
            if hasattr(obj, 'algorithm'):
                names = [str(x).strip() for x in np.asarray(obj.algorithm).ravel()]
                if names != self.algorithms:
                    raise AdjudicationError('%s: algorithm names/order differ' % path)
            local_shape = shape[1:]
            c = _matrix(obj.crit, local_shape, '%s.crit' % problem)
            d = _matrix(obj.dnorm, local_shape, '%s.dnorm' % problem)
            info = _matrix(obj.info, local_shape, '%s.info' % problem)
            if not np.isfinite(info).all():
                raise AdjudicationError('%s: nonfinite algorithm success flag' % path)
            has_cert.append(hasattr(obj, 'cert'))
            if has_cert[-1]:
                raw = _matrix(obj.cert, local_shape, '%s.cert' % problem)
                if not np.isin(raw, [0, 1]).all():
                    raise AdjudicationError('%s: cert must contain only 0 or 1' % path)
                ok = raw == 1
                if np.any(ok & (~np.isfinite(c) | ~np.isfinite(d))):
                    raise AdjudicationError('%s: certified value is nonfinite' % path)
            else:
                ok = np.isfinite(c) & np.isfinite(d)
            if np.any(ok & (d < 0)):
                raise AdjudicationError('%s: negative certified displacement' % path)
            self.criticality[pi] = np.where(ok, c, np.nan)
            self.displacement[pi] = np.where(ok, d, np.nan)
            self.info[pi] = info
            self.certified[pi] = ok
        if any(has_cert) and not all(has_cert):
            raise AdjudicationError('Cannot mix legacy MAT files with final cert-bearing files')
        self.legacy = not any(has_cert)
        missing_indices = np.argwhere(~self.certified)
        self.expected_keys = {
            (self.problems[i], self.algorithms[j], int(k) + 1)
            for i, j, k in missing_indices
        }
        if not self.legacy and expected_uncertified is not None:
            if len(self.expected_keys) != expected_uncertified:
                raise AdjudicationError('Expected %d uncertified final points, found %d'
                                        % (expected_uncertified, len(self.expected_keys)))
        self.rows = {}
        self._indices = {name: i for i, name in enumerate(self.problems)}
        self._alg_indices = {name: i for i, name in enumerate(self.algorithms)}
        path = self.directory / 'adjudicated.csv'
        if not path.is_file():
            if self.expected_keys or not self.legacy:
                raise FileNotFoundError('Required adjudication file is absent: %s' % path)
        else:
            self._read_rows(path)
            self._read_witnesses(path)
        # Validate both final-report criteria before any output is written.
        self.pass_mask('crit', '1e-4')
        self.pass_mask('dnorm', '1e-4')

    def _read_rows(self, path):
        required = {'prob', 'alg', 'run', 'neg_theta_lo', 'neg_theta_hi',
                    'dnorm_lo', 'dnorm_hi', 'pass_theta', 'pass_dnorm'}
        with path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            if not required.issubset(reader.fieldnames or []):
                raise AdjudicationError('%s: missing columns %s'
                                        % (path, sorted(required - set(reader.fieldnames or []))))
            for number, row in enumerate(reader, start=2):
                try:
                    run = int(row['run'])
                except (ValueError, TypeError) as exc:
                    raise AdjudicationError('%s:%d: invalid run' % (path, number)) from exc
                key = (row['prob'], row['alg'], run)
                if key in self.rows:
                    raise AdjudicationError('%s:%d: duplicate key %s' % (path, number, key))
                if key not in self.expected_keys:
                    raise AdjudicationError('%s:%d: unexpected/already-certified key %s'
                                            % (path, number, key))
                bounds = {}
                for field, prefix, flag in [('crit', 'neg_theta', 'pass_theta'),
                                            ('dnorm', 'dnorm', 'pass_dnorm')]:
                    lo = _decimal(row[prefix + '_lo'], '%s %s_lo' % (key, prefix))
                    hi = _decimal(row[prefix + '_hi'], '%s %s_hi' % (key, prefix))
                    if lo > hi or (field == 'dnorm' and lo < 0):
                        raise AdjudicationError('%s: invalid %s interval [%s, %s]'
                                                % (key, field, lo, hi))
                    verdict = _verdict(lo, hi, Decimal('1e-4'), '%s %s' % (key, field))
                    if row[flag] not in ('0', '1') or (row[flag] == '1') != verdict:
                        raise AdjudicationError('%s: %s is inconsistent with the 1e-4 interval'
                                                % (key, flag))
                    bounds[field] = (lo, hi)
                self.rows[key] = bounds
        missing = self.expected_keys - self.rows.keys()
        if missing:
            raise AdjudicationError('%s: %d uncovered uncertified points; first: %s'
                                    % (path, len(missing), sorted(missing)[:3]))

    def _read_witnesses(self, csv_path):
        """Bind interval decisions to the published per-point witness archive."""
        path = self.directory / 'adjudication_witnesses.json'
        with path.open(encoding='utf-8') as stream:
            document = json.load(stream)
        if document.get('schema_version') != 1:
            raise AdjudicationError('%s: unsupported witness schema version' % path)
        if document.get('certificate') != 'exact-rational-primal-dual':
            raise AdjudicationError('%s: unexpected certificate type' % path)
        digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        if document.get('csv_sha256') != digest:
            raise AdjudicationError('%s: csv_sha256 does not match adjudicated.csv' % path)
        keys = set()
        records = document.get('records')
        if not isinstance(records, list):
            raise AdjudicationError('%s: records must be a list' % path)
        for record in records:
            try:
                key = (record['prob'], record['alg'], int(record['run']))
            except (KeyError, TypeError, ValueError) as exc:
                raise AdjudicationError('%s: invalid witness key' % path) from exc
            if key in keys:
                raise AdjudicationError('%s: duplicate witness key %s' % (path, key))
            keys.add(key)
        if keys != self.expected_keys:
            raise AdjudicationError('%s: witness keys do not exactly cover the uncertified points'
                                    % path)
        self.witness_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.csv_sha256 = digest

    def pass_mask(self, field, tolerance, mode='A'):
        """Re-decide every bracket at this tolerance, independently of flags."""
        if field not in ('crit', 'dnorm') or mode not in ('A', 'B'):
            raise ValueError('field must be crit/dnorm and mode must be A/B')
        tol = _decimal(tolerance, 'tolerance')
        if tol < 0:
            raise ValueError('tolerance must be nonnegative')
        values = self.criticality if field == 'crit' else self.displacement
        result = self.certified & np.isfinite(values) & (values <= float(tol))
        for key, intervals in self.rows.items():
            lo, hi = intervals[field]
            verdict = _verdict(lo, hi, tol, '%s %s' % (key, field))
            p, a, run = key
            result[self._indices[p], self._alg_indices[a], run - 1] = verdict
        if mode == 'B':
            result &= self.info == 0
        return result

    def rates(self, field, tolerance, mode='A'):
        return 100.0 * self.pass_mask(field, tolerance, mode).sum(axis=2) / self.runs

    def diagnostics(self):
        covered = ~self.certified
        return {
            'total_points': int(self.certified.size),
            'certified_values': int(self.certified.sum()),
            'adjudicated_points': len(self.rows),
            'passes_criticality': int((self.pass_mask('crit', '1e-4') & covered).sum()),
            'passes_displacement': int((self.pass_mask('dnorm', '1e-4') & covered).sum()),
            'undecided_points': 0,
            'coverage': 'one unique adjudication key for every uncertified point',
            'csv_sha256': getattr(self, 'csv_sha256', None),
            'witness_sha256': getattr(self, 'witness_sha256', None),
            'decision_rule': 'upper <= tolerance: pass; lower > tolerance: fail; '
                             'otherwise raise an error; recomputed for each tolerance',
        }
