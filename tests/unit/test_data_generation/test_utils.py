import unittest
import numpy as np
from tsvalidation.data_generation._utils import (
    _nonlin_func,
    _generate_random_parameters,
    _get_arma_parameters,
    _generate_seeds,
)
from tsvalidation.data_generation.frequency_modulation import FrequencyModulationLinear
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
        lags = 2
        max_root = 1.2
        seed = 1
        random.seed(seed)
        arr = _generate_random_parameters(d2, seed)
        self.assertEqual(len(arr), lags)


if __name__ == "__main__":
    unittest.main()
