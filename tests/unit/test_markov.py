"""
This file contains unit tests regarding the 'MarkovCV' class.
"""

import unittest
from tsvalidation.validation_methods.markov import MarkovCV
import numpy as np
from itertools import combinations
from typing import List


def _checks_union_train_val(method: MarkovCV):
    all_training_patterns = []
    all_validation_patterns = []
    for training, validation in method.split():
        all_training_patterns = np.concatenate((all_training_patterns, training))
        all_validation_patterns = np.concatenate((all_validation_patterns, validation))
    return np.sort(all_training_patterns), np.sort(all_validation_patterns)


class TestMarkovCV(unittest.TestCase):

    def setUp(self) -> None:
        self.ts1 = np.arange(10)
        self.ts2 = np.ones(10)
        self.ts2 = np.ones(1000)

        self.m0 = MarkovCV(self.ts1, 0, 1)
        self.m1 = MarkovCV(self.ts1, 1, 1)
        self.m1_dup = MarkovCV(self.ts1, 1, 1)
        self.m2 = MarkovCV(self.ts2, 1, 1)
        self.m3 = MarkovCV(self.ts2, 2, 1)
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
        self.assert_(not bool(self.m0._suo))
        self.assert_(not bool(self.m0._suo))
        self.assertEqual(self.m0._seed, 1)
        self.assertEqual(self.m0._p, 0)
        self.assertEqual(self.m0._m, 1)
        self.assertEqual(self.m0.n_splits, 2 * 1)
        self.assertEqual(self.m1._m, 2)
        self.assertEqual(self.m1.n_splits, 2 * 2)
        self.assertEqual(self.m2._m, 2)
        self.assertEqual(self.m2.n_splits, 2 * 2)
        self.assertEqual(self.m3._m, 3)
        self.assertEqual(self.m3.n_splits, 2 * 3)

        self.assertEqual(self.m0._n_samples, len(self.ts1))
        self.assertEqual(self.m1._n_samples, len(self.ts1))
        self.assertEqual(self.m2._n_samples, len(self.ts2))
        self.assertEqual(self.m3._n_samples, len(self.ts2))

    def test_markov_iteration(self) -> None:
        np.testing.assert_array_equal(  # checking if the seed is working
            self.m1._markov_iteration(self.m1._n_samples),
            self.m1_dup._markov_iteration(self.m1_dup._n_samples),
        )
        o0 = self.m0._markov_iteration(self.m0._n_samples)
        self.assertEqual(self.m0._n_samples, len(o0))
        o1 = self.m1._markov_iteration(self.m1._n_samples)
        self.assertEqual(self.m1._n_samples, len(o1))
        o2 = self.m2._markov_iteration(self.m2._n_samples)
        self.assertEqual(self.m2._n_samples, len(o2))
        o3 = self.m3._markov_iteration(self.m3._n_samples)
        self.assertEqual(self.m3._n_samples, len(o3))

    def _are_disjoint_arrays(self, arrays: List[np.array]):
        for array1, array2 in combinations(arrays, 2):
            if set(array1.flatten()).intersection(array2.flatten()):
                return False
        return True

    def test_markov_partitions(self) -> None:

        self.m0._markov_partitions()
        self.assert_(bool(self.m0._suo))
        self.assert_(bool(self.m0._sue))
        self.m1._markov_partitions()
        self.assert_(bool(self.m1._suo))
        self.assert_(bool(self.m1._sue))
        self.m2._markov_partitions()
        self.assert_(bool(self.m2._suo))
        self.assert_(bool(self.m2._sue))
        self.m3._markov_partitions()
        self.assert_(bool(self.m3._suo))
        self.assert_(bool(self.m3._sue))

        self.assertEqual(len(self.m0._suo.items()), self.m0._m)
        self.assertEqual(len(self.m1._suo.items()), self.m1._m)
        self.assertEqual(len(self.m2._suo.items()), self.m2._m)
        self.assertEqual(len(self.m3._suo.items()), self.m3._m)

        # check if sets are disjoints
        su0 = list(self.m0._suo.values())
        su0.extend(list(self.m0._sue.values()))
        self.assert_(self._are_disjoint_arrays(su0))

        su1 = list(self.m1._suo.values())
        su1.extend(list(self.m1._sue.values()))
        self.assert_(self._are_disjoint_arrays(su1))

        su2 = list(self.m2._suo.values())
        su2.extend(list(self.m2._sue.values()))
        self.assert_(self._are_disjoint_arrays(su2))

        su3 = list(self.m3._suo.values())
        su3.extend(list(self.m3._sue.values()))
        self.assert_(self._are_disjoint_arrays(su3))

    def test_split(self) -> None:
        # checking no sample is left behind
        all_train, all_val = _checks_union_train_val(self.m0)
        np.testing.assert_array_equal(all_train, np.arange(self.m0._n_samples))
        np.testing.assert_array_equal(all_val, np.arange(self.m0._n_samples))

        all_train, all_val = _checks_union_train_val(self.m1)
        np.testing.assert_array_equal(all_train, np.arange(self.m1._n_samples))
        np.testing.assert_array_equal(all_val, np.arange(self.m1._n_samples))

        all_train, all_val = _checks_union_train_val(self.m2)
        np.testing.assert_array_equal(all_train, np.arange(self.m2._n_samples))
        np.testing.assert_array_equal(all_val, np.arange(self.m2._n_samples))

        all_train, all_val = _checks_union_train_val(self.m3)
        np.testing.assert_array_equal(all_train, np.arange(self.m3._n_samples))
        np.testing.assert_array_equal(all_val, np.arange(self.m3._n_samples))

        return

    def test_info(self) -> None:
        """
        Test the 'info' method.
        """

        self.m0.info()
        self.m1.info()
        self.m2.info()
        self.m3.info()

        return

    """
    def test_plot(self) -> None:
        height = 2
        width = 10

        self.m0.plot(height, width)
        self.m1.plot(height, width)
        self.m2.plot(height, width)
        self.m3.plot(height, width)

        return"""


if __name__ == "__main__":

    unittest.main()
