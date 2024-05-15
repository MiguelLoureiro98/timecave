import unittest
import numpy as np
from timecave.data_generation._utils import (
    _nonlin_func,
    _generate_random_parameters,
    _get_arma_parameters,
    _generate_seeds,
)
from timecave.data_generation.frequency_modulation import FrequencyModulationLinear
import random


class TestUtils(unittest.TestCase):

    def test_nonlin_func(self):

        # Exceptions
        self.assertRaises(ValueError, _nonlin_func, 1.1, 1.0)
        self.assertRaises(ValueError, _nonlin_func, -1, 1.0)
        self.assertRaises(TypeError, _nonlin_func, 1, 1)

        # Funcionality
        x1 = 1.0
        self.assertEqual(_nonlin_func(0, x1), np.cos(x1))
        self.assertEqual(_nonlin_func(1, x1), np.sin(x1))
        self.assertEqual(_nonlin_func(2, x1), np.tanh(x1))
        self.assertEqual(_nonlin_func(3, x1), np.arctan(x1))
        self.assertEqual(_nonlin_func(4, x1), np.exp(-x1 / 10000))

    def test_get_arma_parameters(self):

        # Exceptions
        self.assertRaises(ValueError, _get_arma_parameters, 2, 1.0)
        self.assertRaises(TypeError, _get_arma_parameters, 2.0, 1.2)  # lags type
        self.assertRaises(TypeError, _get_arma_parameters, 2, "1.2")  # max_root type
        self.assertRaises(TypeError, _get_arma_parameters, 2, 1.2, 1.2)  # seed type

        # Funcionality
        lags = 2
        max_root = 1.2
        arr = _get_arma_parameters(2, 1.2)
        self.assertEqual(len(arr), lags)

    def test_generate_random_parameters(self):
        d1 = {}
        d2 = {"a": [0, 1]}
        d3 = {"b": (0, 1)}
        d4 = {"c": 1}
        d5 = {"a": [0, 1], "b": (0, 1), "c": 1}

        # Exceptions
        self.assertRaises(
            TypeError, _generate_random_parameters, 2, 1.0
        )  # non-float seed
        self.assertRaises(ValueError, _generate_random_parameters, d1)  # empty dict
        self.assertRaises(TypeError, _generate_random_parameters, "1")  # non-dict param

        # Funcionality
        seed = 1
        r1 = _generate_random_parameters(d2, seed)
        self.assertEqual(len(r1.items()), len(r1.items()))
        random.seed(seed)
        e1 = random.uniform(d2["a"][0], d2["a"][1])
        self.assertAlmostEqual(r1["a"], e1)

        r2 = _generate_random_parameters(d3, seed)
        self.assertEqual(len(r2.items()), len(r2.items()))
        random.seed(seed)
        e2 = random.choice(d3["b"])
        self.assertAlmostEqual(r2["b"], e2)

        r3 = _generate_random_parameters(d4, seed)
        self.assertEqual(len(r3.items()), len(r3.items()))
        self.assertAlmostEqual(r3["c"], d4["c"])

        r4 = _generate_random_parameters(d5, seed)
        self.assertEqual(len(r4.items()), len(r4.items()))

    def test_generate_seeds(self):
        # Exceptions
        self.assertRaises(TypeError, _generate_seeds, 2, 1.2)  # max_root type
        self.assertRaises(TypeError, _generate_seeds, 2.2, 1)  # seed type

        # Funcionality
        seed, num_seeds = 2, 4
        l = _generate_seeds(seed, num_seeds)
        self.assertEqual(len(l), num_seeds)


if __name__ == "__main__":
    unittest.main()
