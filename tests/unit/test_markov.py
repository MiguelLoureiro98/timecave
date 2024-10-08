"""
This file contains unit tests regarding the 'MarkovCV' class.
"""

import unittest
from timecave.validation_methods.markov import MarkovCV
import numpy as np
from itertools import combinations
from typing import List


def _are_disjoint_arrays(arrays: List[np.array]):
    """
    Checks if arrays are disjoint.
    """
    for array1, array2 in combinations(arrays, 2):
        if set(array1.flatten()).intersection(array2.flatten()):
            return False
    return True


def _checks_union_train_val(method: MarkovCV):
    """
    Concatenates training and validation patterns from a MarkovCV object.
    """

    all_training_patterns = []
    all_validation_patterns = []
    for training, validation, _ in method.split():
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
        assert not bool(self.m0._suo)
        assert not bool(self.m0._suo)
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
        """
        Test the 't_markov_iteration' method.
        """
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

    def test_markov_partitions(self) -> None:
        """
        Test the '_markov_partitions' method.
        """

        self.m0._markov_partitions()
        assert bool(self.m0._suo)
        assert bool(self.m0._sue)
        self.m1._markov_partitions()
        assert bool(self.m1._suo)
        assert bool(self.m1._sue)
        self.m2._markov_partitions()
        assert bool(self.m2._suo)
        assert bool(self.m2._sue)
        self.m3._markov_partitions()
        assert bool(self.m3._suo)
        assert bool(self.m3._sue)

        self.assertEqual(len(self.m0._suo.items()), self.m0._m)
        self.assertEqual(len(self.m1._suo.items()), self.m1._m)
        self.assertEqual(len(self.m2._suo.items()), self.m2._m)
        self.assertEqual(len(self.m3._suo.items()), self.m3._m)

        # check if sets are disjoints
        su0 = list(self.m0._suo.values())
        su0.extend(list(self.m0._sue.values()))
        assert _are_disjoint_arrays(su0)

        su1 = list(self.m1._suo.values())
        su1.extend(list(self.m1._sue.values()))
        assert _are_disjoint_arrays(su1)

        su2 = list(self.m2._suo.values())
        su2.extend(list(self.m2._sue.values()))
        assert _are_disjoint_arrays(su2)

        su3 = list(self.m3._suo.values())
        su3.extend(list(self.m3._sue.values()))
        assert _are_disjoint_arrays(su3)

    def test_split(self) -> None:
        """
        Test the 'split' method.
        """
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

    def test_statistics(self) -> None:
        """
        Test the 'statistics' method.
        """
        columns = [
            "Mean",
            "Median",
            "Min",
            "Max",
            "Variance",
            "P2P_amplitude",
            "Trend_slope",
            "Strength_of_trend",
            "Mean_crossing_rate",
            "Median_crossing_rate",
        ]

        m1_full_stats, m1_training_stats, m1_validation_stats = self.m1.statistics()

        self.assertListEqual(columns, m1_full_stats.columns.tolist())
        self.assertListEqual(columns, m1_training_stats.columns.tolist())
        self.assertListEqual(columns, m1_validation_stats.columns.tolist())

        self.assertEqual(m1_full_stats.shape[0], 1)
        self.assertEqual(m1_training_stats.shape[0], self.m1.n_splits)
        self.assertEqual(m1_validation_stats.shape[0], self.m1.n_splits)

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

    def test_plot(self) -> None:
        """
        Test the 'plot' method.
        """
        height = 2
        width = 10

        self.m0.plot(height, width)
        self.m1.plot(height, width)
        self.m2.plot(height, width)
        self.m3.plot(height, width)

        return


if __name__ == "__main__":

    unittest.main()
