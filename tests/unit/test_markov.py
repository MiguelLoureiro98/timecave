"""
This file contains unit tests regarding the 'MarkovCV' class.
"""

import unittest
from tsvalidation.validation_methods.markov import MarkovCV
import numpy as np


class TestMarkovCV(unittest.TestCase):

    def setUp(self) -> None:
        self.ts1 = np.arange(10)
        self.ts2 = np.ones(10)

        self.Markov
        return

    def test_initialisation(self) -> None:
        """
        Test the class constructors and checks.
        """

        # Exceptions
        self.assertRaises(TypeError, MarkovCV, self.ts1, "1", 1)
        self.assertRaises(TypeError, MarkovCV, self.ts1, 1, "1")
        self.assertRaises(ValueError, MarkovCV, self.ts1, -1, 1)

        with self.assertWarns(UserWarning):
            MarkovCV(self.ts1, 1, 1).sampling_freq

        # Attribute values
        self.assertEquals


if __name__ == "__main__":

    unittest.main()
