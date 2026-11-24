import math
import unittest

import numpy as np
import sympy as sp
from scipy.linalg import expm

import stepsize_strategies as sss
from stepsize_strategies import _core
from stepsize_strategies._expr import parse_function

try:
    import pandas  # noqa: F401
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ScalarStrategyTests(unittest.TestCase):
    def test_picard_reaches_end_of_interval(self):
        sol = sss.picard("x", x0=1, a=1, b=1)
        self.assertTrue(sol.reached_end)
        self.assertAlmostEqual(sol.t[-1], 1.0)
        self.assertAlmostEqual(sol.h.sum(), 1.0)
        self.assertEqual(sol.x.shape, (sol.steps + 1,))

    def test_error_analysis_converges(self):
        sol = sss.error_analysis("x", x0=1, a=1, b=1, delta=1e-6)
        self.assertAlmostEqual(sol.x[-1], math.e, places=2)

    def test_error_analysis_step_matches_formula(self):
        # x' = t + x: f_t + f_x f = 1 + t + x, maximal at the upper corner of the rectangle
        sol = sss.error_analysis("t + x", x0=1, a=1, b=1, delta=1e-3)
        m0 = 1 + (0 + 1) + (1 + 1)
        self.assertAlmostEqual(sol.h[0], math.sqrt(2 * 1e-3 / m0))

    def test_h_min_stops_integration(self):
        sol = sss.error_analysis("x", x0=1, a=1, b=1, delta=1e-6, h_min=0.5)
        self.assertTrue(sol.stopped_by_h_min)
        self.assertEqual(sol.steps, 0)

    def test_step_limit(self):
        with self.assertRaises(sss.ComputationError):
            sss.error_analysis("x", x0=1, a=1, b=1, delta=1e-12, max_steps=50)

    def test_picard_error_shrinks_the_rectangle(self):
        # b_1 = min(1, 0.5 * 2 * 0.5**2) = 0.25, so the second step is 0.25 / 1.75
        sol = sss.picard_error("x", x0=1, a=1, b=1, h_min=1e-3)
        self.assertAlmostEqual(sol.h[0], 0.5)
        self.assertAlmostEqual(sol.h[1], 0.25 / 1.75)
        self.assertEqual(sol.steps, 3)
        self.assertTrue(sol.stopped_by_h_min)

    def test_inputs_as_text_and_sympy(self):
        t, x = sp.symbols("t x", real=True)
        a = sss.picard("t - x^2", x0="0,5", a="pi/4", b=1)
        b = sss.picard(t - x**2, x0=0.5, a=math.pi / 4, b=1)
        np.testing.assert_allclose(a.x, b.x)
        self.assertAlmostEqual(a.t[-1], math.pi / 4)

    def test_python_functions_are_refused(self):
        with self.assertRaises(TypeError):
            sss.picard(lambda t, x: x, x0=1, a=1, b=1)

    def test_unknown_variable(self):
        with self.assertRaises(sss.ExpressionError):
            sss.picard("x + y", x0=1, a=1, b=1)

    def test_legacy_flag_is_accepted(self):
        sol = sss.picard("t*x", x0=1, a=1, b=1, legacy=True)
        self.assertTrue(sol.parameters["legacy"])


class SystemStrategyTests(unittest.TestCase):
    def test_linear_local_error_uses_matrix_exponential(self):
        A = [[0.0, 1.0], [-1.0, 0.0]]
        sol = sss.linear_system(A, [1, 0], a=1, b=1, delta=1e-3)
        h = sol.h[0]
        y0 = np.array([1.0, 0.0])
        expected = np.linalg.norm(expm(np.array(A) * h) @ y0 - (y0 + h * np.array(A) @ y0))
        self.assertAlmostEqual(sol.local_error[0], expected)
        self.assertAlmostEqual(sol.t[-1], 1.0)
        self.assertEqual(sol.x.shape, (sol.steps + 1, 2))

    def test_linear_step_matches_formula(self):
        A = [[0, 1, 0], [0, 0, 1], [-1, -3, -3]]
        sol = sss.linear_system(A, [1, 0, 0], a=1, b=1, delta=5e-3)
        alpha, beta, n = 3.0, 2.0, 3
        self.assertAlmostEqual(sol.h[0], math.sqrt(2 * 5e-3 / beta) / (alpha * n**1.25))

    def test_linear_legacy_differs_for_non_diagonal_matrix(self):
        A = [[0.0, 1.0], [-1.0, 0.0]]
        new = sss.linear_system(A, [1, 0], a=1, b=1, delta=1e-3)
        old = sss.linear_system(A, [1, 0], a=1, b=1, delta=1e-3, legacy=True)
        self.assertNotAlmostEqual(new.local_error[0], old.local_error[0])

    def test_nonlinear_total_derivative_is_exact(self):
        A = [[0.0, 1.0], [-1.0, 0.3]]
        names = _core.symbol_table(2)
        model = _core.build_nonlinear_model(
            A, [parse_function("0.1*x**2+t*y", names), parse_function("0.2*y*x+sin(t)", names)]
        )
        t0, X = 0.7, np.array([0.4, -0.3])
        flow = lambda t, Y: np.array(A) @ Y + model.phi(t, *Y)  # noqa: E731
        h = 1e-6
        numeric = (model.phi(t0 + h, *(X + h * flow(t0, X))) - model.phi(t0 - h, *(X - h * flow(t0, X)))) / (2 * h)
        np.testing.assert_allclose(model.dphi(t0, *X), numeric, rtol=1e-6)

    def test_nonlinear_system_runs(self):
        sol = sss.nonlinear_system([[1, 0], [0, -0.5]], ["-0.5*x*y", "0.25*x*y"], [2, 1],
                                   a=5, b=1, delta=5e-3)
        self.assertTrue(sol.reached_end)
        self.assertAlmostEqual(sol.t[-1], 5.0)
        self.assertIsNone(sol.local_error)
        self.assertTrue(np.all(np.isfinite(sol.x)))

    def test_nonlinear_variable_aliases(self):
        a = sss.nonlinear_system([[1, 0], [0, -0.5]], ["-0.5*x*y", "0.25*x*y"], [2, 1], a=1, b=1, delta=5e-3)
        b = sss.nonlinear_system([[1, 0], [0, -0.5]], ["-0.5*x1*x2", "0.25*x1*x2"], [2, 1], a=1, b=1, delta=5e-3)
        np.testing.assert_allclose(a.x, b.x)

    def test_size_checks(self):
        with self.assertRaises(ValueError):
            sss.linear_system([[1, 2, 3], [4, 5, 6]], [1, 0], a=1, b=1, delta=1e-3)
        with self.assertRaises(ValueError):
            sss.linear_system([[1, 0], [0, 1]], [1, 0, 0], a=1, b=1, delta=1e-3)
        with self.assertRaises(ValueError):
            sss.nonlinear_system([[1, 0], [0, 1]], ["x*y"], [1, 0], a=1, b=1, delta=1e-3)


class SolutionTests(unittest.TestCase):
    def setUp(self):
        self.sol = sss.linear_system([[0, 1], [-1, -0.3]], [1, 0], a=2, b=1, delta=5e-3)

    def test_summary(self):
        info = self.sol.summary()
        self.assertEqual(info["steps"], self.sol.steps)
        self.assertTrue(info["reached_end"])
        self.assertEqual(len(info["x_final"]), 2)
        self.assertIn("largest_local_error", info)
        self.assertIn("steps=", repr(self.sol))

    @unittest.skipUnless(HAS_PANDAS, "pandas is not installed")
    def test_dataframe(self):
        df = self.sol.to_dataframe()
        self.assertEqual(list(df.columns), ["step", "t", "h", "x1", "x2", "norm", "local_error"])
        self.assertEqual(len(df), self.sol.steps)

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is not installed")
    def test_plot(self):
        ax = self.sol.plot()
        self.assertEqual(len(ax.get_lines()), 2)


if __name__ == "__main__":
    unittest.main()
