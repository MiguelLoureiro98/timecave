"""
This file contains unit tests targetting the 'CV' module.
"""

import unittest
from tsvalidation.validation_methods.CV import Block_CV, hv_Block_CV
from tsvalidation.validation_methods.weights import linear_weights, exponential_weights
import numpy as np


class TestCV(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        cls.test_array_simple = np.arange(1, 11);
        cls.test_array_simple_odd = np.arange(1, 16);
        cls.test_array_high_freq = np.arange(1, 11, 0.01);
        cls.specific_array_1 = np.arange(1, 12);
        cls.specific_array_2 = np.arange(1, 13);
        cls.simple_freq = 1;
        cls.high_freq = 100;
        cls.n_splits = 5;
        cls.h_simple = 1;
        cls.v_simple = 1;
        cls.h_large = 50;
        cls.v_large = 50;

        cls.Block1 = Block_CV(cls.n_splits, cls.test_array_simple, cls.simple_freq);
        cls.Block2 = Block_CV(cls.n_splits, cls.test_array_simple_odd, cls.simple_freq);
        cls.Block3 = Block_CV(cls.n_splits, cls.test_array_high_freq, cls.high_freq);
        cls.Block1_linear = Block_CV(cls.n_splits, cls.test_array_simple, cls.simple_freq, weight_function=linear_weights, params={"slope": 2});
        cls.Block2_linear = Block_CV(cls.n_splits, cls.test_array_simple_odd, cls.simple_freq, weight_function=linear_weights, params={"slope": 2});
        cls.Block3_linear = Block_CV(cls.n_splits, cls.test_array_high_freq, cls.high_freq, weight_function=linear_weights, params={"slope": 2});
        cls.Block1_exponential = Block_CV(cls.n_splits, cls.test_array_simple, cls.simple_freq, weight_function=exponential_weights, params={"base": 2});
        cls.Block2_exponential = Block_CV(cls.n_splits, cls.test_array_simple_odd, cls.simple_freq, weight_function=exponential_weights, params={"base": 2});
        cls.Block3_exponential = Block_CV(cls.n_splits, cls.test_array_high_freq, cls.high_freq, weight_function=exponential_weights, params={"base": 2});

        cls.hvBlock1 = hv_Block_CV(cls.test_array_simple, cls.simple_freq, cls.h_simple, cls.v_simple);
        cls.hvBlock2 = hv_Block_CV(cls.test_array_simple_odd, cls.simple_freq, cls.h_simple, cls.v_simple);
        cls.hvBlock3 = hv_Block_CV(cls.test_array_high_freq, cls.high_freq, cls.h_large, cls.v_large);
        cls.hvBlock_limit = hv_Block_CV(cls.test_array_simple, cls.simple_freq, 2, 2);

        return;

    @classmethod
    def tearDownClass(cls) -> None:

        del cls.test_array_simple;
        del cls.test_array_simple_odd;
        del cls.test_array_high_freq;
        del cls.specific_array_1;
        del cls.specific_array_2;
        del cls.simple_freq;
        del cls.high_freq;
        del cls.n_splits;
        del cls.Block1;
        del cls.Block2;
        del cls.Block3;
        del cls.Block1_linear;
        del cls.Block2_linear;
        del cls.Block3_linear;
        del cls.Block1_exponential;
        del cls.Block2_exponential;
        del cls.Block3_exponential;
        del cls.hvBlock1;
        del cls.hvBlock2;
        del cls.hvBlock3;
        del cls.hvBlock_limit;

        return;

    def test_initialisation(self):

        """
        Test the class constructors and checks.
        """

        pass

    def test_split(self):

        """
        Test the 'split' method.
        """

        pass

    def test_info(self):

        """
        Test the 'info' method.
        """

        pass

    def test_statistics(self):

        """
        Test the 'statistics' method.
        """

        pass

    def test_plot(self):

        """
        Test the 'plot' method.
        """

        pass


if __name__ == "__main__":

    unittest.main();
