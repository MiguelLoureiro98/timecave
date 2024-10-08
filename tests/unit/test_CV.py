"""
This file contains unit tests targetting the 'CV' module.
"""

import unittest
from timecave.validation_methods.CV import BlockCV, hvBlockCV, AdaptedhvBlockCV
from timecave.validation_methods.weights import linear_weights, exponential_weights
import numpy as np


class TestCV(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        cls.test_array_simple = np.arange(1, 11);
        cls.test_array_simple_odd = np.arange(1, 14);
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

        cls.Block1 = BlockCV(cls.n_splits, cls.test_array_simple, cls.simple_freq);
        cls.Block2 = BlockCV(cls.n_splits, cls.test_array_simple_odd, cls.simple_freq);
        cls.Block3 = BlockCV(cls.n_splits, cls.test_array_high_freq, cls.high_freq);
        cls.Block4 = BlockCV(cls.n_splits, cls.specific_array_1, cls.simple_freq);
        cls.Block5 = BlockCV(cls.n_splits, cls.specific_array_2, cls.simple_freq);
        cls.Block1_linear = BlockCV(cls.n_splits, cls.test_array_simple, cls.simple_freq, weight_function=linear_weights, params={"slope": 2});
        cls.Block2_linear = BlockCV(cls.n_splits, cls.test_array_simple_odd, cls.simple_freq, weight_function=linear_weights, params={"slope": 2});
        cls.Block3_linear = BlockCV(cls.n_splits, cls.test_array_high_freq, cls.high_freq, weight_function=linear_weights, params={"slope": 2});
        cls.Block1_exponential = BlockCV(cls.n_splits, cls.test_array_simple, cls.simple_freq, weight_function=exponential_weights, params={"base": 2});
        cls.Block2_exponential = BlockCV(cls.n_splits, cls.test_array_simple_odd, cls.simple_freq, weight_function=exponential_weights, params={"base": 2});
        cls.Block3_exponential = BlockCV(cls.n_splits, cls.test_array_high_freq, cls.high_freq, weight_function=exponential_weights, params={"base": 2});

        cls.hvBlock1 = hvBlockCV(cls.test_array_simple, cls.simple_freq, cls.h_simple, cls.v_simple);
        cls.hvBlock2 = hvBlockCV(cls.test_array_simple_odd, cls.simple_freq, cls.h_simple, cls.v_simple);
        cls.hvBlock3 = hvBlockCV(cls.test_array_high_freq, cls.high_freq, cls.h_large, cls.v_large);
        cls.hvBlock_limit = hvBlockCV(cls.test_array_simple, cls.simple_freq, 2, 2);

        cls.adapted1 = AdaptedhvBlockCV(cls.n_splits, cls.test_array_simple, cls.simple_freq, cls.h_simple);
        cls.adapted2 = AdaptedhvBlockCV(cls.n_splits, cls.test_array_simple_odd, cls.simple_freq, cls.h_simple);
        cls.adapted3 = AdaptedhvBlockCV(cls.n_splits, cls.test_array_high_freq, cls.high_freq, cls.h_large);

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
        del cls.Block4;
        del cls.Block5;
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
        del cls.adapted1;
        del cls.adapted2;
        del cls.adapted3;

        return;

    def test_initialisation(self):

        """
        Test the class constructors and checks.
        """

        # Exceptions
        self.assertRaises(TypeError, hvBlockCV, self.test_array_simple, self.simple_freq, 0.5);
        self.assertRaises(TypeError, hvBlockCV, self.test_array_simple, self.simple_freq, 1, 0.5);
        self.assertRaises(ValueError, hvBlockCV, self.test_array_simple, self.simple_freq, -1);
        self.assertRaises(ValueError, hvBlockCV, self.test_array_simple, self.simple_freq, 1, -1);
        self.assertRaises(ValueError, hvBlockCV, self.test_array_simple, self.simple_freq, 3, 2);
        self.assertRaises(ValueError, hvBlockCV, self.test_array_simple_odd, self.simple_freq, 3, 4);
        self.assertRaises(TypeError, AdaptedhvBlockCV, 5, self.test_array_simple, self.simple_freq, 0.5);
        self.assertRaises(ValueError, AdaptedhvBlockCV, 5, self.test_array_simple, self.simple_freq, -1);
        self.assertRaises(ValueError, AdaptedhvBlockCV, 5, self.test_array_simple, self.simple_freq, 10);

        # Functionality
        self.assertEqual(self.Block1.n_splits, self.n_splits);
        self.assertEqual(self.Block2.n_splits, self.n_splits);
        self.assertEqual(self.Block3.n_splits, self.n_splits);
        self.assertEqual(self.Block1_linear.n_splits, self.n_splits);
        self.assertEqual(self.Block2_linear.n_splits, self.n_splits);
        self.assertEqual(self.Block3_linear.n_splits, self.n_splits);
        self.assertEqual(self.Block1_exponential.n_splits, self.n_splits);
        self.assertEqual(self.Block2_exponential.n_splits, self.n_splits);
        self.assertEqual(self.Block3_exponential.n_splits, self.n_splits);
        self.assertEqual(self.hvBlock1.n_splits, self.test_array_simple.shape[0]);
        self.assertEqual(self.hvBlock2.n_splits, self.test_array_simple_odd.shape[0]);
        self.assertEqual(self.hvBlock3.n_splits, self.test_array_high_freq.shape[0]);
        self.assertEqual(self.hvBlock_limit.n_splits, self.test_array_simple.shape[0]);
        self.assertEqual(self.adapted1.n_splits, self.n_splits);
        self.assertEqual(self.adapted2.n_splits, self.n_splits);
        self.assertEqual(self.adapted3.n_splits, self.n_splits);

        self.assertEqual(self.Block1.sampling_freq, self.simple_freq);
        self.assertEqual(self.Block2.sampling_freq, self.simple_freq);
        self.assertEqual(self.Block3.sampling_freq, self.high_freq);
        self.assertEqual(self.Block1_linear.sampling_freq, self.simple_freq);
        self.assertEqual(self.Block2_linear.sampling_freq, self.simple_freq);
        self.assertEqual(self.Block3_linear.sampling_freq, self.high_freq);
        self.assertEqual(self.Block1_exponential.sampling_freq, self.simple_freq);
        self.assertEqual(self.Block2_exponential.sampling_freq, self.simple_freq);
        self.assertEqual(self.Block3_exponential.sampling_freq, self.high_freq);
        self.assertEqual(self.hvBlock1.sampling_freq, self.simple_freq);
        self.assertEqual(self.hvBlock2.sampling_freq, self.simple_freq);
        self.assertEqual(self.hvBlock3.sampling_freq, self.high_freq);
        self.assertEqual(self.hvBlock_limit.sampling_freq, self.simple_freq);
        self.assertEqual(self.adapted1.sampling_freq, self.simple_freq);
        self.assertEqual(self.adapted2.sampling_freq, self.simple_freq);
        self.assertEqual(self.adapted3.sampling_freq, self.high_freq);

        return;

    def test_split(self):

        """
        Test the 'split' method.
        """

        # Definitions
        indices1 = np.arange(0, 10).tolist();
        indices2 = np.arange(0, 13).tolist();
        indices3 = np.arange(0, 1000).tolist();
        indices4 = np.arange(0, 11).tolist();
        indices5 = np.arange(0, 12).tolist();

        const_w = np.ones(5);
        linear_w = np.array([2/30, 4/30, 6/30, 8/30, 10/30]);
        exponential_w = np.array([1/31, 2/31, 4/31, 8/31, 16/31]);

        blockCV_val1 = [indices1[ind:ind+2] for ind in indices1 if ind % 2 == 0];
        blockCV_val2 = [[0, 1, 2],
                          [3, 4, 5],
                          [6, 7, 8],
                          [9, 10],
                          [11, 12]];
        blockCV_val3 = [indices3[ind:ind+200] for ind in indices3 if ind % 200 == 0];
        blockCV_val4 = [[0, 1, 2], 
                          [3, 4],
                          [5, 6],
                          [7, 8],
                          [9, 10]];
        blockCV_val5 = [[0, 1, 2], 
                          [3, 4, 5],
                          [6, 7],
                          [8, 9],
                          [10, 11]];
        blockCV1 = [[ind for ind in indices1 if ind not in blockCV_val1[i]] for i in range(5)];
        blockCV2 = [[ind for ind in indices2 if ind not in blockCV_val2[i]] for i in range(5)];
        blockCV3 = [[ind for ind in indices3 if ind not in blockCV_val3[i]] for i in range(5)];
        blockCV4 = [[ind for ind in indices4 if ind not in blockCV_val4[i]] for i in range(5)];
        blockCV5 = [[ind for ind in indices5 if ind not in blockCV_val5[i]] for i in range(5)];

        hvblock_val1 = [[0, 1],
                         [0, 1, 2],
                         [1, 2, 3],
                         [2, 3, 4],
                         [3, 4, 5],
                         [4, 5, 6],
                         [5, 6, 7],
                         [6, 7, 8],
                         [7, 8, 9],
                         [8, 9]];
        hvblock_val2 = [[0, 1],
                         [0, 1, 2],
                         [1, 2, 3],
                         [2, 3, 4],
                         [3, 4, 5],
                         [4, 5, 6],
                         [5, 6, 7],
                         [6, 7, 8],
                         [7, 8, 9],
                         [8, 9, 10],
                         [9, 10, 11],
                         [10, 11, 12],
                         [11, 12]];
        hvblock_val3 = [indices3[np.fmax(i - self.v_large, 0) : np.fmin(i + self.v_large + 1, len(indices3))] for i in indices3];
        hvblock_val_limit = [[0, 1, 2],
                              [0, 1, 2, 3],
                              [0, 1, 2, 3, 4],
                              [1, 2, 3, 4, 5],
                              [2, 3, 4, 5, 6],
                              [3, 4, 5, 6, 7],
                              [4, 5, 6, 7, 8],
                              [5, 6, 7, 8, 9],
                              [6, 7, 8, 9],
                              [7, 8, 9]];
        
        h_list1 = [indices1[np.fmax(i - self.v_simple - self.h_simple, 0) : np.fmin(i + self.v_simple + self.h_simple + 1, len(indices1))] for i in indices1];
        h_list2 = [indices2[np.fmax(i - self.v_simple - self.h_simple, 0) : np.fmin(i + self.v_simple + self.h_simple + 1, len(indices2))] for i in indices2];
        h_list3 = [indices3[np.fmax(i - self.v_large - self.h_large, 0) : np.fmin(i + self.v_large + self.h_large + 1, len(indices3))] for i in indices3];
        h_list4 = [indices1[np.fmax(i - 2 - 2, 0) : np.fmin(i + 2 + 2 + 1, len(indices1))] for i in indices1];
        
        hvblock1 = [[ind for ind in indices1 if ind not in h_list1[i]] for i in range(len(h_list1))];
        hvblock2 = [[ind for ind in indices2 if ind not in h_list2[i]] for i in range(len(h_list2))];
        hvblock3 = [[ind for ind in indices3 if ind not in h_list3[i]] for i in range(len(h_list3))];
        hvblock_limit = [[ind for ind in indices1 if ind not in h_list4[i]] for i in range(len(h_list4))];

        adapted_val1 = [indices1[ind:ind+2] for ind in indices1 if ind % 2 == 0];
        adapted_val2 = [[0, 1, 2],
                          [3, 4, 5],
                          [6, 7, 8],
                          [9, 10],
                          [11, 12]];
        adapted_val3 = [indices3[ind:ind+200] for ind in indices3 if ind % 200 == 0];

        adapted_hlist1 = [indices1[np.fmax(ind-self.h_simple, 0) : np.fmin(ind+2+self.h_simple, len(indices1))] for ind in indices1 if ind % 2 == 0];
        adapted_hlist2 = [[0, 1, 2, 3],
                          [2, 3, 4, 5, 6],
                          [5, 6, 7, 8, 9],
                          [8, 9, 10, 11],
                          [10, 11, 12]];
        adapted_hlist3 = [indices3[np.fmax(ind-self.h_large, 0) : np.fmin(ind+200+self.h_large, len(indices3))] for ind in indices3 if ind % 200 == 0];

        adapted1 = [[ind for ind in indices1 if ind not in adapted_hlist1[i]] for i in range(len(adapted_hlist1))];
        adapted2 = [[ind for ind in indices2 if ind not in adapted_hlist2[i]] for i in range(len(adapted_hlist2))];
        adapted3 = [[ind for ind in indices3 if ind not in adapted_hlist3[i]] for i in range(len(adapted_hlist3))];
        
        # Tests - Block CV
        for ind, (train, val, w) in enumerate(self.Block1.split()):

            self.assertListEqual(train.tolist(), blockCV1[ind]);
            self.assertListEqual(val.tolist(), blockCV_val1[ind]);
            self.assertEqual(w, const_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Block2.split()):

            self.assertListEqual(train.tolist(), blockCV2[ind]);
            self.assertListEqual(val.tolist(), blockCV_val2[ind]);
            self.assertEqual(w, const_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Block3.split()):

            self.assertListEqual(train.tolist(), blockCV3[ind]);
            self.assertListEqual(val.tolist(), blockCV_val3[ind]);
            self.assertEqual(w, const_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Block4.split()):

            self.assertListEqual(train.tolist(), blockCV4[ind]);
            self.assertListEqual(val.tolist(), blockCV_val4[ind]);
            self.assertEqual(w, const_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Block5.split()):

            self.assertListEqual(train.tolist(), blockCV5[ind]);
            self.assertListEqual(val.tolist(), blockCV_val5[ind]);
            self.assertEqual(w, const_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Block1_linear.split()):

            self.assertListEqual(train.tolist(), blockCV1[ind]);
            self.assertListEqual(val.tolist(), blockCV_val1[ind]);
            self.assertEqual(w, linear_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Block2_linear.split()):

            self.assertListEqual(train.tolist(), blockCV2[ind]);
            self.assertListEqual(val.tolist(), blockCV_val2[ind]);
            self.assertEqual(w, linear_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Block3_linear.split()):

            self.assertListEqual(train.tolist(), blockCV3[ind]);
            self.assertListEqual(val.tolist(), blockCV_val3[ind]);
            self.assertEqual(w, linear_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Block1_exponential.split()):

            self.assertListEqual(train.tolist(), blockCV1[ind]);
            self.assertListEqual(val.tolist(), blockCV_val1[ind]);
            self.assertEqual(w, exponential_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Block2_exponential.split()):

            self.assertListEqual(train.tolist(), blockCV2[ind]);
            self.assertListEqual(val.tolist(), blockCV_val2[ind]);
            self.assertEqual(w, exponential_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Block3_exponential.split()):

            self.assertListEqual(train.tolist(), blockCV3[ind]);
            self.assertListEqual(val.tolist(), blockCV_val3[ind]);
            self.assertEqual(w, exponential_w[ind]);
        
        # Tests - hv Block CV
        for ind, (train, val, _) in enumerate(self.hvBlock1.split()):

            self.assertListEqual(train.tolist(), hvblock1[ind]);
            self.assertListEqual(val.tolist(), hvblock_val1[ind]);
        
        for ind, (train, val, _) in enumerate(self.hvBlock2.split()):

            self.assertListEqual(train.tolist(), hvblock2[ind]);
            self.assertListEqual(val.tolist(), hvblock_val2[ind]);
        
        for ind, (train, val, _) in enumerate(self.hvBlock3.split()):

            self.assertListEqual(train.tolist(), hvblock3[ind]);
            self.assertListEqual(val.tolist(), hvblock_val3[ind]);
        
        for ind, (train, val, _) in enumerate(self.hvBlock_limit.split()):

            self.assertListEqual(train.tolist(), hvblock_limit[ind]);
            self.assertListEqual(val.tolist(), hvblock_val_limit[ind]);

        # Tests - Adapted hv Block CV
        for ind, (train, val, _) in enumerate(self.adapted1.split()):

            self.assertListEqual(train.tolist(), adapted1[ind]);
            self.assertListEqual(val.tolist(), adapted_val1[ind]);
        
        for ind, (train, val, _) in enumerate(self.adapted2.split()):

            self.assertListEqual(train.tolist(), adapted2[ind]);
            self.assertListEqual(val.tolist(), adapted_val2[ind]);
        
        for ind, (train, val, _) in enumerate(self.adapted3.split()):

            self.assertListEqual(train.tolist(), adapted3[ind]);
            self.assertListEqual(val.tolist(), adapted_val3[ind]);

        return;

    def test_info(self):

        """
        Test the 'info' method.
        """

        self.Block1.info();
        self.Block2.info();
        self.Block3.info();
        self.Block4.info();
        self.Block5.info();

        self.Block1_linear.info();
        self.Block2_linear.info();
        self.Block3_linear.info();

        self.Block1_exponential.info();
        self.Block2_exponential.info();
        self.Block3_exponential.info();

        self.hvBlock1.info();
        self.hvBlock2.info();
        self.hvBlock3.info();
        self.hvBlock_limit.info();

        self.adapted1.info();
        self.adapted2.info();
        self.adapted3.info();

        return;

    def test_statistics(self):

        """
        Test the 'statistics' method.
        """

        # Exceptions
        two_sample_series = np.zeros(shape=(2,));
        small_series = np.zeros(shape=(5,));

        block_2samp = BlockCV(2, two_sample_series, 1);
        hv2samp = hvBlockCV(two_sample_series, 1);
        block_small = BlockCV(5, small_series, 1);

        self.assertRaises(ValueError, block_2samp.statistics);
        self.assertRaises(ValueError, hv2samp.statistics);
        self.assertRaises(ValueError, block_small.statistics);

        # Functionality
        columns = 13
        column_list = [
            "Mean",
            "Median",
            "Min",
            "Max",
            "Variance",
            "P2P_amplitude",
            "Trend_slope",
            "Spectral_centroid",
            "Spectral_rolloff",
            "Spectral_entropy",
            "Strength_of_trend",
            "Mean_crossing_rate",
            "Median_crossing_rate",
        ]

        block1_full_stats, block1_train_stats, block1_val_stats = self.Block1.statistics();
        block2_full_stats, block2_train_stats, block2_val_stats = self.Block2.statistics();
        hvblock1_full_stats, hvblock1_train_stats, hvblock1_val_stats = self.hvBlock1.statistics();
        hvblock2_full_stats, hvblock2_train_stats, hvblock2_val_stats = self.hvBlock2.statistics();
        adapted1_full_stats, adapted1_train_stats, adapted1_val_stats = self.adapted1.statistics();
        adapted2_full_stats, adapted2_train_stats, adapted2_val_stats = self.adapted2.statistics();
        
        self.assertEqual(block1_full_stats.shape[1], columns);
        self.assertListEqual(block1_full_stats.columns.tolist(), column_list);

        self.assertEqual(block1_full_stats.shape[0], 1);
        self.assertEqual(block1_train_stats.shape[0], 5);
        self.assertEqual(block1_val_stats.shape[0], 5);
        self.assertEqual(block2_full_stats.shape[0], 1);
        self.assertEqual(block2_train_stats.shape[0], 5);
        self.assertEqual(block2_val_stats.shape[0], 5);

        self.assertEqual(hvblock1_full_stats.shape[0], 1);
        self.assertEqual(hvblock1_train_stats.shape[0], self.test_array_simple.shape[0]);
        self.assertEqual(hvblock1_val_stats.shape[0], self.test_array_simple.shape[0]);
        self.assertEqual(hvblock2_full_stats.shape[0], 1);
        self.assertEqual(hvblock2_train_stats.shape[0], self.test_array_simple_odd.shape[0]);
        self.assertEqual(hvblock2_val_stats.shape[0], self.test_array_simple_odd.shape[0]);

        self.assertEqual(adapted1_full_stats.shape[0], 1);
        self.assertEqual(adapted1_train_stats.shape[0], 5);
        self.assertEqual(adapted1_val_stats.shape[0], 5);
        self.assertEqual(adapted2_full_stats.shape[0], 1);
        self.assertEqual(adapted2_train_stats.shape[0], 5);
        self.assertEqual(adapted2_val_stats.shape[0], 5);

        return;

    def test_plot(self):

        """
        Test the 'plot' method.
        """

        height = 10;
        width = 10;

        self.Block1.plot(height, width);
        self.Block2.plot(height, width);
        self.Block3.plot(height, width);
        self.Block4.plot(height, width);
        self.Block5.plot(height, width);

        self.Block1_linear.plot(height, width);
        self.Block2_linear.plot(height, width);
        self.Block3_linear.plot(height, width);

        self.Block1_exponential.plot(height, width);
        self.Block2_exponential.plot(height, width);
        self.Block3_exponential.plot(height, width);

        self.hvBlock1.plot(height, width);
        self.hvBlock2.plot(height, width);
        self.hvBlock_limit.plot(height, width);

        self.adapted1.plot(height, width);
        self.adapted2.plot(height, width);
        self.adapted3.plot(height, width);

        return;


if __name__ == "__main__":

    unittest.main();
