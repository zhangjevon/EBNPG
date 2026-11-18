"""Focused regression checks for the earlier false exact-origin and gap bugs."""
import importlib.util
from pathlib import Path
import unittest
from fractions import Fraction
import numpy as np

module_path = Path(__file__).with_name("certify_adjudicated.py")
spec = importlib.util.spec_from_file_location("certificate_under_test", module_path)
cert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cert)


def point(gradients, x=1.0):
    m = len(gradients)
    return dict(n=1, m=m, prob="regression", alg="synthetic", run=1,
                x=np.array([x]), l=np.array([-2.]), u=np.array([2.]),
                JG=np.array(gradients, dtype=float).reshape(m, 1),
                B=[np.eye(1) for _ in range(m)], delta=[1.] * m,
                H=np.array([abs(x)] * m))


class CertificateRegressions(unittest.TestCase):
    def test_inactive_feasible_cube_does_not_certify_origin(self):
        model = cert.ExactModel(point([0., -2.]))
        model.add_primal(np.zeros(1), "zero_step")
        model.add_origin_candidates()
        self.assertIsNone(model.origin)
        self.assertEqual(model.primal_value([-Fraction(1)]), Fraction(3, 2))
        self.assertEqual(model.U, 0)

    def test_active_feasible_cube_certifies_origin_exactly(self):
        model = cert.ExactModel(point([0.], x=0.125))
        model.add_primal(np.zeros(1), "zero_step")
        model.add_origin_candidates()
        self.assertIsNotNone(model.origin)
        self.assertEqual(model.origin["active_objective"], 1)
        self.assertEqual(model.U, model.L)
        self.assertEqual(model.direction, [-Fraction(1, 8)])

    def test_negative_exact_gap_is_an_error(self):
        model = cert.ExactModel(point([0.]))
        model.U, model.L, model.direction = Fraction(-1), Fraction(0), [Fraction(0)]
        with self.assertRaises(ArithmeticError):
            model.endpoints()

    def test_square_root_enclosure_is_outward(self):
        low, high = cert.sqrt_interval(Fraction(2), digits=30)
        self.assertLess(low * low, 2)
        self.assertGreater(high * high, 2)
        self.assertEqual(cert.sqrt_interval(Fraction(4)), (Fraction(2), Fraction(2)))

    def test_projection_produces_exact_feasible_dual(self):
        model = cert.ExactModel(point([0., -2.]))
        weights, cube = model.feasible_dual([-3., 7.], [[4., -9.]])
        self.assertEqual(sum(weights), 1)
        self.assertTrue(all(weight >= 0 for weight in weights))
        self.assertTrue(all(abs(value) <= 1 for row in cube for value in row))
        model.add_primal(np.zeros(1), "zero")
        model.add_dual(weights, cube, "projection")
        self.assertLessEqual(model.L, model.U)


if __name__ == "__main__":
    unittest.main(verbosity=2)

