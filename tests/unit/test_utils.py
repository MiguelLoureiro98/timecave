"""
This file contains tests targetting the 'utils' module.
"""

import unittest
from tsvalidation.utils import Nyquist_min_samples, heuristic_min_samples, true_test_indices
import numpy as np

class TestUtils(unittest.TestCase):

    def setUp(self) -> None:
        
        self.fs1 = 100;
        self.fs2 = 1.5;
        self.freq_limit1 = 6;
        self.freq_limit2 = 7;
        self.Nyquist_min1 = np.ceil((2 / self.freq_limit1) * self.fs1);
        self.Nyquist_min2 = np.ceil((2 / self.freq_limit2) * self.fs1);
        self.heuristic_min1 = np.ceil((10 / self.freq_limit1) * self.fs1);
        self.heuristic_min2 = np.ceil((10 / self.freq_limit2) * self.fs1);
        self.heuristic_max1 = np.ceil((20 / self.freq_limit1) * self.fs1);
        self.heuristic_max2 = np.ceil((20 / self.freq_limit2) * self.fs1);

        return;

    def tearDown(self) -> None:

        del self.fs1;
        del self.fs2;
        del self.freq_limit1;
        del self.freq_limit2;
        del self.Nyquist_min1;
        del self.Nyquist_min2;
        del self.heuristic_min1;
        del self.heuristic_min2;
        del self.heuristic_max1;
        del self.heuristic_max2;

        return;
    
    def test_frequency(self) -> None:

        """
        Test the 'Nyquist_min_samples' and the 'heuristic_min_samples' functions.
        """

        Nyquist_samples1 = Nyquist_min_samples(self.fs1, self.freq_limit1);
        Nyquist_samples2 = Nyquist_min_samples(self.fs1, self.freq_limit2);

        heuristic_samples1 = heuristic_min_samples(self.fs1, self.freq_limit1);
        heuristic_samples2 = heuristic_min_samples(self.fs1, self.freq_limit2);

        # Exceptions
        self.assertRaises(TypeError, Nyquist_min_samples, "a", 1);
        self.assertRaises(TypeError, Nyquist_min_samples, 1, "a");
        self.assertRaises(ValueError, Nyquist_min_samples, -1, 1);
        self.assertRaises(ValueError, Nyquist_min_samples, 1, -1);
        self.assertRaises(Warning, Nyquist_min_samples, self.fs2, self.freq_limit1);

        self.assertRaises(TypeError, heuristic_min_samples, "a", 1);
        self.assertRaises(TypeError, heuristic_min_samples, 1, "a");
        self.assertRaises(ValueError, heuristic_min_samples, -1, 1);
        self.assertRaises(ValueError, heuristic_min_samples, 1, -1);
        self.assertRaises(Warning, heuristic_min_samples, self.fs2, self.freq_limit1);
        self.assertRaises(Warning, heuristic_min_samples, 150, self.freq_limit1);

        # Functionality
        self.assertEqual(Nyquist_samples1, self.Nyquist_min1);
        self.assertEqual(Nyquist_samples2, self.Nyquist_min2);

        self.assertEqual(heuristic_samples1["Min_samples"], self.heuristic_min1);
        self.assertEqual(heuristic_samples1["Max_samples"], self.heuristic_max1);
        self.assertEqual(heuristic_samples2["Min_samples"], self.heuristic_min2);
        self.assertEqual(heuristic_samples2["Max_samples"], self.heuristic_max2);
    
        return;

    def test_indices(self) -> None:

        """
        Test the 'true_test_indices' function.
        """

        pass

if __name__ == "__main__":

    unittest.main();