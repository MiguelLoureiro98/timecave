"""
This file contains unit tests targetting the 'CV' module.
"""

import unittest
from tsvalidation.validation_methods.CV import Block_CV, hv_Block_CV
import numpy as np


class TestCV(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        cls.test_array_simple = np.arange(1, 11)
        cls.test_array_simple_odd = np.arange(1, 16)
        cls.test_array_high_freq = np.arange(1, 11, 0.01)
        cls.simple_freq = 1
        cls.high_freq = 100

        return

    @classmethod
    def tearDownClass(cls) -> None:

        del cls.test_array_simple
        del cls.test_array_simple_odd
        del cls.test_array_high_freq
        del cls.simple_freq
        del cls.high_freq

        return

    def test_initialisation(self):

        pass

    def test_split(self):

        pass

    def test_info(self):

        pass

    def test_statistics(self):

        pass

    def test_plot(self):

        pass


if __name__ == "__main__":

    unittest.main()
