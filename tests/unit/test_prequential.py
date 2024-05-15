"""
This file contains unit tests targetting the 'Prequential' module.
"""

import unittest
from timecave.validation_methods.prequential import Growing_Window, Rolling_Window
from timecave.validation_methods.weights import linear_weights, exponential_weights
import numpy as np


class TestPrequential(unittest.TestCase):

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

        cls.Growing1 = Growing_Window(cls.n_splits, cls.test_array_simple, cls.simple_freq);
        cls.Growing2 = Growing_Window(cls.n_splits, cls.test_array_simple_odd, cls.simple_freq);
        cls.Growing3 = Growing_Window(cls.n_splits, cls.test_array_high_freq, cls.high_freq);
        cls.Growing4 = Growing_Window(cls.n_splits, cls.specific_array_1, cls.simple_freq);
        cls.Growing5 = Growing_Window(cls.n_splits, cls.specific_array_2, cls.simple_freq);
        cls.Growing1_gap = Growing_Window(cls.n_splits, cls.test_array_simple, cls.simple_freq, gap=1);
        cls.Growing2_gap = Growing_Window(cls.n_splits, cls.test_array_simple_odd, cls.simple_freq, gap=2);
        cls.Growing3_gap = Growing_Window(cls.n_splits, cls.test_array_high_freq, cls.high_freq, gap=3);
        cls.Growing1_linear = Growing_Window(cls.n_splits, cls.test_array_simple, cls.simple_freq, weight_function=linear_weights, params={"slope": 2});
        cls.Growing2_linear = Growing_Window(cls.n_splits, cls.test_array_simple_odd, cls.simple_freq, weight_function=linear_weights, params={"slope": 2});
        cls.Growing3_linear = Growing_Window(cls.n_splits, cls.test_array_high_freq, cls.high_freq, weight_function=linear_weights, params={"slope": 2});
        cls.Growing1_exponential = Growing_Window(cls.n_splits, cls.test_array_simple, cls.simple_freq, weight_function=exponential_weights, params={"base": 2});
        cls.Growing2_exponential = Growing_Window(cls.n_splits, cls.test_array_simple_odd, cls.simple_freq, weight_function=exponential_weights, params={"base": 2});
        cls.Growing3_exponential = Growing_Window(cls.n_splits, cls.test_array_high_freq, cls.high_freq, weight_function=exponential_weights, params={"base": 2});             

        cls.Rolling1 = Rolling_Window(cls.n_splits, cls.test_array_simple, cls.simple_freq);
        cls.Rolling2 = Rolling_Window(cls.n_splits, cls.test_array_simple_odd, cls.simple_freq);
        cls.Rolling3 = Rolling_Window(cls.n_splits, cls.test_array_high_freq, cls.high_freq);
        cls.Rolling4 = Rolling_Window(cls.n_splits, cls.specific_array_1, cls.simple_freq);
        cls.Rolling5 = Rolling_Window(cls.n_splits, cls.specific_array_2, cls.simple_freq);
        cls.Rolling1_gap = Rolling_Window(cls.n_splits, cls.test_array_simple, cls.simple_freq, gap=1);
        cls.Rolling2_gap = Rolling_Window(cls.n_splits, cls.test_array_simple_odd, cls.simple_freq, gap=2);
        cls.Rolling3_gap = Rolling_Window(cls.n_splits, cls.test_array_high_freq, cls.high_freq, gap=3);
        cls.Rolling1_linear = Rolling_Window(cls.n_splits, cls.test_array_simple, cls.simple_freq, weight_function=linear_weights, params={"slope": 2});
        cls.Rolling2_linear = Rolling_Window(cls.n_splits, cls.test_array_simple_odd, cls.simple_freq, weight_function=linear_weights, params={"slope": 2});
        cls.Rolling3_linear = Rolling_Window(cls.n_splits, cls.test_array_high_freq, cls.high_freq, weight_function=linear_weights, params={"slope": 2});
        cls.Rolling1_exponential = Rolling_Window(cls.n_splits, cls.test_array_simple, cls.simple_freq, weight_function=exponential_weights, params={"base": 2});
        cls.Rolling2_exponential = Rolling_Window(cls.n_splits, cls.test_array_simple_odd, cls.simple_freq, weight_function=exponential_weights, params={"base": 2});
        cls.Rolling3_exponential = Rolling_Window(cls.n_splits, cls.test_array_high_freq, cls.high_freq, weight_function=exponential_weights, params={"base": 2});       

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
        del cls.Growing1;
        del cls.Growing2;
        del cls.Growing3;
        del cls.Growing4;
        del cls.Growing5;
        del cls.Growing1_gap;
        del cls.Growing2_gap;
        del cls.Growing3_gap;
        del cls.Growing1_linear;
        del cls.Growing2_linear;
        del cls.Growing3_linear;
        del cls.Growing1_exponential;
        del cls.Growing2_exponential;
        del cls.Growing3_exponential;
        del cls.Rolling1;
        del cls.Rolling2;
        del cls.Rolling3;
        del cls.Rolling4;
        del cls.Rolling5;
        del cls.Rolling1_gap;
        del cls.Rolling2_gap;
        del cls.Rolling3_gap;
        del cls.Rolling1_linear;
        del cls.Rolling2_linear;
        del cls.Rolling3_linear;
        del cls.Rolling1_exponential;
        del cls.Rolling2_exponential;
        del cls.Rolling3_exponential;

        return;

    def test_initialisation(self):

        """
        Test the class constructors and checks.
        """

        # Exceptions
        self.assertRaises(TypeError, Growing_Window, self.n_splits, self.test_array_simple, self.simple_freq, gap=0.5);
        self.assertRaises(ValueError, Growing_Window, self.n_splits, self.test_array_simple, self.simple_freq, gap=-1);
        self.assertRaises(ValueError, Growing_Window, self.n_splits, self.test_array_simple, self.simple_freq, gap=4);

        self.assertRaises(TypeError, Rolling_Window, self.n_splits, self.test_array_simple, self.simple_freq, gap=0.5);
        self.assertRaises(ValueError, Rolling_Window, self.n_splits, self.test_array_simple, self.simple_freq, gap=-1);
        self.assertRaises(ValueError, Rolling_Window, self.n_splits, self.test_array_simple, self.simple_freq, gap=4);

        # Functionality
        self.assertEqual(self.Growing1.n_splits, self.n_splits);
        self.assertEqual(self.Growing2.n_splits, self.n_splits);
        self.assertEqual(self.Growing3.n_splits, self.n_splits);
        self.assertEqual(self.Growing1_gap.n_splits, self.n_splits);
        self.assertEqual(self.Growing2_gap.n_splits, self.n_splits);
        self.assertEqual(self.Growing3_gap.n_splits, self.n_splits);
        self.assertEqual(self.Growing1_linear.n_splits, self.n_splits);
        self.assertEqual(self.Growing2_linear.n_splits, self.n_splits);
        self.assertEqual(self.Growing3_linear.n_splits, self.n_splits);
        self.assertEqual(self.Growing1_exponential.n_splits, self.n_splits);
        self.assertEqual(self.Growing2_exponential.n_splits, self.n_splits);
        self.assertEqual(self.Growing3_exponential.n_splits, self.n_splits);

        self.assertEqual(self.Growing1.sampling_freq, self.simple_freq);
        self.assertEqual(self.Growing2.sampling_freq, self.simple_freq);
        self.assertEqual(self.Growing3.sampling_freq, self.high_freq);
        self.assertEqual(self.Growing1_gap.sampling_freq, self.simple_freq);
        self.assertEqual(self.Growing2_gap.sampling_freq, self.simple_freq);
        self.assertEqual(self.Growing3_gap.sampling_freq, self.high_freq);
        self.assertEqual(self.Growing1_linear.sampling_freq, self.simple_freq);
        self.assertEqual(self.Growing2_linear.sampling_freq, self.simple_freq);
        self.assertEqual(self.Growing3_linear.sampling_freq, self.high_freq);
        self.assertEqual(self.Growing1_exponential.sampling_freq, self.simple_freq);
        self.assertEqual(self.Growing2_exponential.sampling_freq, self.simple_freq);
        self.assertEqual(self.Growing3_exponential.sampling_freq, self.high_freq);

        self.assertEqual(self.Rolling1.n_splits, self.n_splits);
        self.assertEqual(self.Rolling2.n_splits, self.n_splits);
        self.assertEqual(self.Rolling3.n_splits, self.n_splits);
        self.assertEqual(self.Rolling1_gap.n_splits, self.n_splits);
        self.assertEqual(self.Rolling2_gap.n_splits, self.n_splits);
        self.assertEqual(self.Rolling3_gap.n_splits, self.n_splits);
        self.assertEqual(self.Rolling1_linear.n_splits, self.n_splits);
        self.assertEqual(self.Rolling2_linear.n_splits, self.n_splits);
        self.assertEqual(self.Rolling3_linear.n_splits, self.n_splits);
        self.assertEqual(self.Rolling1_exponential.n_splits, self.n_splits);
        self.assertEqual(self.Rolling2_exponential.n_splits, self.n_splits);
        self.assertEqual(self.Rolling3_exponential.n_splits, self.n_splits);

        self.assertEqual(self.Rolling1.sampling_freq, self.simple_freq);
        self.assertEqual(self.Rolling2.sampling_freq, self.simple_freq);
        self.assertEqual(self.Rolling3.sampling_freq, self.high_freq);
        self.assertEqual(self.Rolling1_gap.sampling_freq, self.simple_freq);
        self.assertEqual(self.Rolling2_gap.sampling_freq, self.simple_freq);
        self.assertEqual(self.Rolling3_gap.sampling_freq, self.high_freq);
        self.assertEqual(self.Rolling1_linear.sampling_freq, self.simple_freq);
        self.assertEqual(self.Rolling2_linear.sampling_freq, self.simple_freq);
        self.assertEqual(self.Rolling3_linear.sampling_freq, self.high_freq);
        self.assertEqual(self.Rolling1_exponential.sampling_freq, self.simple_freq);
        self.assertEqual(self.Rolling2_exponential.sampling_freq, self.simple_freq);
        self.assertEqual(self.Rolling3_exponential.sampling_freq, self.high_freq);

        return;

    def test_split(self):

        """
        Test the 'split' method.
        """

        # Definitions
        indices1 = np.arange(0, 10).tolist();
        #indices2 = np.arange(0, 11)#.tolist(); # add new arrays for 12 and 11!
        indices3 = np.arange(0, 1000).tolist();
        #exp = Growing_Window(5, indices2, self.simple_freq);
        #print(exp._splitting_ind);
        #for (t, v, _) in exp.split():

        #    print(t);
        #    print(v);

        const_w = np.ones(4);
        linear_w = np.array([0.1, 0.2, 0.3, 0.4]);
        exponential_w = np.array([1/15, 2/15, 4/15, 8/15]);

        growing_train1 = [indices1[:ind] for ind in indices1 if ind != 0 and ind % 2 == 0];
        growing_train2 = [[0, 1, 2], 
                          [0, 1, 2, 3, 4, 5],
                          [0, 1, 2, 3, 4, 5, 6, 7, 8],
                          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]];
        growing_train3 = [indices3[:ind] for ind in indices3 if ind != 0 and ind % 200 == 0];
        growing_train4 = [[0, 1, 2], 
                          [0, 1, 2, 3, 4],
                          [0, 1, 2, 3, 4, 5, 6],
                          [0, 1, 2, 3, 4, 5, 6, 7, 8]];
        growing_train5 = [[0, 1, 2], 
                          [0, 1, 2, 3, 4, 5],
                          [0, 1, 2, 3, 4, 5, 6, 7],
                          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]];
        growing_val1 = [indices1[train[-1]+1:train[-1]+3] for train in growing_train1];
        growing_val2 = [[3, 4, 5],
                        [6, 7, 8],
                        [9, 10],
                        [11, 12]];
        growing_val3 = [indices3[train[-1]+1:train[-1]+201] for train in growing_train3];
        growing_val4 = [[3, 4],
                        [5, 6],
                        [7, 8],
                        [9, 10]];
        growing_val5 = [[3, 4, 5],
                        [6, 7],
                        [8, 9],
                        [10, 11]];
        growing_train1_gap = [indices1[:ind] for ind in indices1 if ind != 0 and ind % 2 == 0 and ind < 8];
        growing_train2_gap = [[0, 1, 2],
                              [0, 1, 2, 3, 4, 5]];
        growing_train3_gap = [indices3[:ind] for ind in indices3 if ind != 0 and ind % 200 == 0 and ind < 400];
        growing_val1_gap = [indices1[train[-1]+3:train[-1]+5] for train in growing_train1_gap];
        growing_val2_gap = [[9, 10],
                            [11, 12]];
        growing_val3_gap = [indices3[train[-1]+601:train[-1]+801] for train in growing_train3_gap];

        rolling_train1 = [indices1[ind:ind+2] for ind in indices1 if ind % 2 == 0 and ind < 8];
        rolling_train2 = [[0, 1, 2],
                          [3, 4, 5],
                          [6, 7, 8],
                          [9, 10]];
        rolling_train3 = [indices3[ind:ind+200] for ind in indices3 if ind % 200 == 0 and ind < 800];
        rolling_train4 = [[0, 1, 2], 
                          [3, 4],
                          [5, 6],
                          [7, 8]];
        rolling_train5 = [[0, 1, 2], 
                          [3, 4, 5],
                          [6, 7],
                          [8, 9]];
        rolling_val1 = [indices1[train[-1]+1:train[-1]+3] for train in rolling_train1];
        rolling_val2 = [[3, 4, 5],
                        [6, 7, 8],
                        [9, 10],
                        [11, 12]];
        rolling_val3 = [indices3[train[-1]+1:train[-1]+201] for train in rolling_train3];
        rolling_val4 = [[3, 4],
                        [5, 6],
                        [7, 8],
                        [9, 10]];
        rolling_val5 = [[3, 4, 5],
                        [6, 7],
                        [8, 9],
                        [10, 11]];
        rolling_train1_gap = [indices1[ind:ind+2] for ind in indices1 if ind % 2 == 0 and ind < 6];
        rolling_train2_gap = [[0, 1, 2],
                              [3, 4, 5]];
        rolling_train3_gap = [indices3[:ind] for ind in indices3 if ind != 0 and ind % 200 == 0 and ind < 400];
        rolling_val1_gap = [indices1[train[-1]+3:train[-1]+5] for train in rolling_train1_gap];
        rolling_val2_gap = [[9, 10],
                            [11, 12]];
        rolling_val3_gap = [indices3[train[-1]+601:train[-1]+801] for train in rolling_train3_gap];

        # Tests - Growing window
        for ind, (train, val, w) in enumerate(self.Growing1.split()):

            self.assertListEqual(train.tolist(), growing_train1[ind]);
            self.assertListEqual(val.tolist(), growing_val1[ind]);
            self.assertEqual(w, const_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Growing2.split()):

            self.assertListEqual(train.tolist(), growing_train2[ind]);
            self.assertListEqual(val.tolist(), growing_val2[ind]);
            self.assertEqual(w, const_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Growing3.split()):

            self.assertListEqual(train.tolist(), growing_train3[ind]);
            self.assertListEqual(val.tolist(), growing_val3[ind]);
            self.assertEqual(w, const_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Growing4.split()):

            self.assertListEqual(train.tolist(), growing_train4[ind]);
            self.assertListEqual(val.tolist(), growing_val4[ind]);
            self.assertEqual(w, const_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Growing5.split()):

            self.assertListEqual(train.tolist(), growing_train5[ind]);
            self.assertListEqual(val.tolist(), growing_val5[ind]);
            self.assertEqual(w, const_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Growing1_gap.split()):

            self.assertListEqual(train.tolist(), growing_train1_gap[ind]);
            self.assertListEqual(val.tolist(), growing_val1_gap[ind]);
            self.assertEqual(w, const_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Growing2_gap.split()):

            self.assertListEqual(train.tolist(), growing_train2_gap[ind]);
            self.assertListEqual(val.tolist(), growing_val2_gap[ind]);
            self.assertEqual(w, const_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Growing3_gap.split()):

            self.assertListEqual(train.tolist(), growing_train3_gap[ind]);
            self.assertListEqual(val.tolist(), growing_val3_gap[ind]);
            self.assertEqual(w, const_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Growing1_linear.split()):

            self.assertListEqual(train.tolist(), growing_train1[ind]);
            self.assertListEqual(val.tolist(), growing_val1[ind]);
            self.assertEqual(w, linear_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Growing2_linear.split()):

            self.assertListEqual(train.tolist(), growing_train2[ind]);
            self.assertListEqual(val.tolist(), growing_val2[ind]);
            self.assertEqual(w, linear_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Growing3_linear.split()):

            self.assertListEqual(train.tolist(), growing_train3[ind]);
            self.assertListEqual(val.tolist(), growing_val3[ind]);
            self.assertEqual(w, linear_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Growing1_exponential.split()):

            self.assertListEqual(train.tolist(), growing_train1[ind]);
            self.assertListEqual(val.tolist(), growing_val1[ind]);
            self.assertEqual(w, exponential_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Growing2_exponential.split()):

            self.assertListEqual(train.tolist(), growing_train2[ind]);
            self.assertListEqual(val.tolist(), growing_val2[ind]);
            self.assertEqual(w, exponential_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Growing3_exponential.split()):

            self.assertListEqual(train.tolist(), growing_train3[ind]);
            self.assertListEqual(val.tolist(), growing_val3[ind]);
            self.assertEqual(w, exponential_w[ind]);
        
        # Tests - Rolling window
        for ind, (train, val, w) in enumerate(self.Rolling1.split()):

            self.assertListEqual(train.tolist(), rolling_train1[ind]);
            self.assertListEqual(val.tolist(), rolling_val1[ind]);
            self.assertEqual(w, const_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Rolling2.split()):

            self.assertListEqual(train.tolist(), rolling_train2[ind]);
            self.assertListEqual(val.tolist(), rolling_val2[ind]);
            self.assertEqual(w, const_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Rolling3.split()):

            self.assertListEqual(train.tolist(), rolling_train3[ind]);
            self.assertListEqual(val.tolist(), rolling_val3[ind]);
            self.assertEqual(w, const_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Rolling4.split()):

            self.assertListEqual(train.tolist(), rolling_train4[ind]);
            self.assertListEqual(val.tolist(), rolling_val4[ind]);
            self.assertEqual(w, const_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Rolling5.split()):

            self.assertListEqual(train.tolist(), rolling_train5[ind]);
            self.assertListEqual(val.tolist(), rolling_val5[ind]);
            self.assertEqual(w, const_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Rolling1_gap.split()):

            self.assertListEqual(train.tolist(), rolling_train1_gap[ind]);
            self.assertListEqual(val.tolist(), rolling_val1_gap[ind]);
            self.assertEqual(w, const_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Rolling2_gap.split()):

            self.assertListEqual(train.tolist(), rolling_train2_gap[ind]);
            self.assertListEqual(val.tolist(), rolling_val2_gap[ind]);
            self.assertEqual(w, const_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Rolling3_gap.split()):

            self.assertListEqual(train.tolist(), rolling_train3_gap[ind]);
            self.assertListEqual(val.tolist(), rolling_val3_gap[ind]);
            self.assertEqual(w, const_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Rolling1_linear.split()):

            self.assertListEqual(train.tolist(), rolling_train1[ind]);
            self.assertListEqual(val.tolist(), rolling_val1[ind]);
            self.assertEqual(w, linear_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Rolling2_linear.split()):

            self.assertListEqual(train.tolist(), rolling_train2[ind]);
            self.assertListEqual(val.tolist(), rolling_val2[ind]);
            self.assertEqual(w, linear_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Rolling3_linear.split()):

            self.assertListEqual(train.tolist(), rolling_train3[ind]);
            self.assertListEqual(val.tolist(), rolling_val3[ind]);
            self.assertEqual(w, linear_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Rolling1_exponential.split()):

            self.assertListEqual(train.tolist(), rolling_train1[ind]);
            self.assertListEqual(val.tolist(), rolling_val1[ind]);
            self.assertEqual(w, exponential_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Rolling2_exponential.split()):

            self.assertListEqual(train.tolist(), rolling_train2[ind]);
            self.assertListEqual(val.tolist(), rolling_val2[ind]);
            self.assertEqual(w, exponential_w[ind]);
        
        for ind, (train, val, w) in enumerate(self.Rolling3_exponential.split()):

            self.assertListEqual(train.tolist(), rolling_train3[ind]);
            self.assertListEqual(val.tolist(), rolling_val3[ind]);
            self.assertEqual(w, exponential_w[ind]);

        return;

    def test_info(self):

        """
        Test the 'info' method.
        """

        self.Growing1.info();
        self.Growing2.info();
        self.Growing3.info();
        self.Growing1_gap.info();
        self.Growing2_gap.info();
        self.Growing3_gap.info();
        self.Growing1_linear.info();
        self.Growing2_linear.info();
        self.Growing3_linear.info();
        self.Growing1_exponential.info();
        self.Growing2_exponential.info();
        self.Growing3_exponential.info();

        self.Rolling1.info();
        self.Rolling2.info();
        self.Rolling3.info();
        self.Rolling1_gap.info();
        self.Rolling2_gap.info();
        self.Rolling3_gap.info();
        self.Rolling1_linear.info();
        self.Rolling2_linear.info();
        self.Rolling3_linear.info();
        self.Rolling1_exponential.info();
        self.Rolling2_exponential.info();
        self.Rolling3_exponential.info();

        return;

    def test_statistics(self):
        
        """
        Test the 'statistics' method.
        """

        # Exceptions
        two_sample_series = np.zeros(shape=(2,));
        growing2samp = Growing_Window(2, two_sample_series, 1);
        rolling2samp = Rolling_Window(2, two_sample_series, 1);

        small_series = np.zeros(shape=(5,));
        growing5samp = Growing_Window(5, small_series, 1);
        rolling5samp = Rolling_Window(5, small_series, 1);

        self.assertRaises(ValueError, growing2samp.statistics);
        self.assertRaises(ValueError, rolling2samp.statistics);
        self.assertRaises(ValueError, growing5samp.statistics);
        self.assertRaises(ValueError, rolling5samp.statistics);

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

        growing1_full_stats, growing1_train_stats, growing1_val_stats = self.Growing1.statistics();
        rolling1_full_stats, rolling1_train_stats, rolling1_val_stats = self.Rolling1.statistics();

        growing2_full_stats, growing2_train_stats, growing2_val_stats = self.Growing2.statistics();
        rolling2_full_stats, rolling2_train_stats, rolling2_val_stats = self.Rolling2.statistics();

        growing_gap_full_stats, growing_gap_train_stats, growing_gap_val_stats = self.Growing1_gap.statistics();
        rolling_gap_full_stats, rolling_gap_train_stats, rolling_gap_val_stats = self.Rolling1_gap.statistics();

        self.assertEqual(growing1_full_stats.shape[1], columns);
        self.assertListEqual(growing1_full_stats.columns.tolist(), column_list);

        self.assertEqual(growing1_full_stats.shape[0], 1);
        self.assertEqual(growing1_train_stats.shape[0], 4);
        self.assertEqual(growing1_val_stats.shape[0], 4);
        self.assertEqual(rolling1_full_stats.shape[0], 1);
        self.assertEqual(rolling1_train_stats.shape[0], 4);
        self.assertEqual(rolling1_val_stats.shape[0], 4);

        self.assertEqual(growing2_full_stats.shape[0], 1);
        self.assertEqual(growing2_train_stats.shape[0], 4);
        self.assertEqual(growing2_val_stats.shape[0], 4);
        self.assertEqual(rolling2_full_stats.shape[0], 1);
        self.assertEqual(rolling2_train_stats.shape[0], 4);
        self.assertEqual(rolling2_val_stats.shape[0], 4);

        self.assertEqual(growing_gap_full_stats.shape[0], 1);
        self.assertEqual(growing_gap_train_stats.shape[0], 3);
        self.assertEqual(growing_gap_val_stats.shape[0], 3);
        self.assertEqual(rolling_gap_full_stats.shape[0], 1);
        self.assertEqual(rolling_gap_train_stats.shape[0], 3);
        self.assertEqual(rolling_gap_val_stats.shape[0], 3);

        return;

    def test_plot(self):

        """
        Test the 'plot' method.
        """

        height = 10;
        width = 10;

        self.Growing1.plot(height, width);
        self.Growing2.plot(height, width);
        self.Growing3.plot(height, width);
        self.Growing1_gap.plot(height, width);
        self.Growing2_gap.plot(height, width);
        self.Growing3_gap.plot(height, width);
        self.Growing1_linear.plot(height, width);
        self.Growing2_linear.plot(height, width);
        self.Growing3_linear.plot(height, width);
        self.Growing1_exponential.plot(height, width);
        self.Growing2_exponential.plot(height, width);
        self.Growing3_exponential.plot(height, width);

        self.Rolling1.plot(height, width);
        self.Rolling2.plot(height, width);
        self.Rolling3.plot(height, width);
        self.Rolling1_gap.plot(height, width);
        self.Rolling2_gap.plot(height, width);
        self.Rolling3_gap.plot(height, width);
        self.Rolling1_linear.plot(height, width);
        self.Rolling2_linear.plot(height, width);
        self.Rolling3_linear.plot(height, width);
        self.Rolling1_exponential.plot(height, width);
        self.Rolling2_exponential.plot(height, width);
        self.Rolling3_exponential.plot(height, width);

        return;


if __name__ == "__main__":

    unittest.main()
