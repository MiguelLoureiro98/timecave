"""
This file contains tests targetting the 'weights' module.
"""

import unittest
from timecave.validation_methods.weights import (
    constant_weights,
    linear_weights,
    exponential_weights,
)
import numpy as np


class TestWeights(unittest.TestCase):

    def setUp(self) -> None:

        self.splits = 5
        self.integer_param = 2
        self.gap = 1
        self.compensation = 1

        self.constant_w = np.array([1, 1, 1, 1, 1])
        self.linear_w = np.array([2 / 30, 4 / 30, 6 / 30, 8 / 30, 10 / 30])
        self.linear_w_gap = np.array([0.1, 0.2, 0.3, 0.4])
        self.linear_w_gap_comp = np.array([1 / 6, 2 / 6, 3 / 6])
        self.exponential_w = np.array([1 / 31, 2 / 31, 4 / 31, 8 / 31, 16 / 31])
        self.exponential_w_gap = np.array([1 / 15, 2 / 15, 4 / 15, 8 / 15])
        self.exponential_w_gap_comp = np.array([1 / 7, 2 / 7, 4 / 7])

        return

    def tearDown(self) -> None:

        del self.splits
        del self.integer_param
        del self.gap
        del self.compensation
        del self.constant_w
        del self.linear_w
        del self.linear_w_gap
        del self.exponential_w
        del self.exponential_w_gap

        return

    def test_weights(self):
        """
        Test built-in weighting functions.
        """

        # Exceptions
        self.assertRaises(TypeError, linear_weights, 5, 0, 0, {"slope": "a"})
        self.assertRaises(ValueError, linear_weights, 5, 0, 0, {"slope": -1})
        self.assertRaises(TypeError, exponential_weights, 5, 0, 0, {"base": "a"})
        self.assertRaises(ValueError, exponential_weights, 5, 0, 0, {"base": -1})

        # Functionality
        constant_w = constant_weights(self.splits)
        constant_w_gap = constant_weights(self.splits, self.gap)
        constant_w_gap_compensation = constant_weights(
            self.splits, self.gap, self.compensation
        )
        linear_w = linear_weights(self.splits, params={"slope": self.integer_param})
        linear_w_gap = linear_weights(
            self.splits, self.gap, params={"slope": self.integer_param}
        )
        linear_w_gap_compensation = linear_weights(
            self.splits,
            self.gap,
            self.compensation,
            params={"slope": self.integer_param},
        )
        exponential_w = exponential_weights(
            self.splits, params={"base": self.integer_param}
        )
        exponential_w_gap = exponential_weights(
            self.splits, self.gap, params={"base": self.integer_param}
        )
        exponential_w_gap_compensation = exponential_weights(
            self.splits,
            self.gap,
            self.compensation,
            params={"base": self.integer_param},
        )

        np.testing.assert_array_equal(constant_w, self.constant_w)
        np.testing.assert_array_equal(constant_w_gap, self.constant_w[:-1])
        np.testing.assert_array_equal(constant_w_gap_compensation, self.constant_w[:-2])
        np.testing.assert_array_equal(linear_w, self.linear_w)
        np.testing.assert_array_equal(linear_w_gap, self.linear_w_gap)
        np.testing.assert_array_equal(linear_w_gap_compensation, self.linear_w_gap_comp)
        np.testing.assert_array_equal(exponential_w, self.exponential_w)
        np.testing.assert_array_equal(exponential_w_gap, self.exponential_w_gap)
        np.testing.assert_array_equal(
            exponential_w_gap_compensation, self.exponential_w_gap_comp
        )

        return


if __name__ == "__main__":

    unittest.main()
