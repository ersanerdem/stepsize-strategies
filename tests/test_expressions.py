import math
import unittest

from stepsize_strategies._core import symbol_table
from stepsize_strategies._expr import ExpressionError, parse_function, parse_number


class SafeExpressionTests(unittest.TestCase):
    def setUp(self):
        self.vars = symbol_table(1)

    def test_valid_expressions(self):
        for text in ["x*(1-x)", "sin(t)+x^2", "-2*t*x", "exp(-x)/(1+t**2)", "sqrt(x**2+1)", "2*pi*x", "0,5*x"]:
            parse_function(text, self.vars)

    def test_code_injection_is_rejected(self):
        for text in [
            "__import__('os').system('id')",
            "x.__class__",
            "open('/etc/passwd')",
            "eval(1)",
            "lambda: 1",
            "x; import os",
            "9**9**9**9**9",
            "y+1",
        ]:
            with self.assertRaises(ExpressionError, msg=text):
                parse_function(text, self.vars)

    def test_numbers(self):
        self.assertAlmostEqual(parse_number("pi/2"), math.pi / 2)
        self.assertEqual(parse_number("1e-3"), 0.001)
        self.assertEqual(parse_number("0,007"), 0.007)
        for bad in ["", "x", "1/0", "inf", "nan", "1,2,3"]:
            with self.assertRaises(ExpressionError):
                parse_number(bad)

    def test_integer_constants_stay_exact(self):
        self.assertEqual(str(parse_function("2*x", self.vars)), "2*x")


if __name__ == "__main__":
    unittest.main()
