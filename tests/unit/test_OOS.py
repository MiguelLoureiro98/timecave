"""
This file contains unit tests targetting the 'OOS' module.
"""

import unittest
from tsvalidation.validation_methods.OOS import Holdout, Repeated_Holdout, Rolling_Origin_Update, Rolling_Origin_Recalibration, Fixed_Size_Rolling_Window
import numpy as np

class TestOOS(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        
        cls.test_array_simple = np.arange(1, 11);
        cls.test_array_simple_odd = np.arange(1, 16);
        cls.test_array_high_freq = np.arange(1, 11, 0.01);
        cls.simple_freq = 1;
        cls.high_freq = 100;
        cls.Holdout1 = Holdout(cls.test_array_simple, cls.simple_freq);
        cls.Holdout2 = Holdout(cls.test_array_simple_odd, cls.simple_freq);
        cls.Holdout3 = Holdout(cls.test_array_high_freq, cls.high_freq, validation_size=0.5);
        cls.Repeated_Holdout1 = Repeated_Holdout(cls.test_array_simple, cls.simple_freq, 2);
        cls.Repeated_Holdout2 = Repeated_Holdout(cls.test_array_simple_odd, cls.simple_freq, 5, [7, 10]);
        cls.Repeated_Holdout3 = Repeated_Holdout(cls.test_array_high_freq, cls.high_freq, 10, [0.5, 0.6]);
        cls.Update1 = Rolling_Origin_Update(cls.test_array_simple, cls.simple_freq);
        cls.Update2 = Rolling_Origin_Update(cls.test_array_simple_odd, cls.simple_freq, origin=9);
        cls.Update3 = Rolling_Origin_Update(cls.test_array_high_freq, cls.high_freq, origin=0.5);
        cls.Recalibration1 = Rolling_Origin_Recalibration(cls.test_array_simple, cls.simple_freq);
        cls.Recalibration2 = Rolling_Origin_Recalibration(cls.test_array_simple_odd, cls.simple_freq, origin=9);
        cls.Recalibration3 = Rolling_Origin_Recalibration(cls.test_array_high_freq, cls.high_freq, origin=0.5);
        cls.Window1 = Fixed_Size_Rolling_Window(cls.test_array_simple, cls.simple_freq);
        cls.Window2 = Fixed_Size_Rolling_Window(cls.test_array_simple_odd, cls.simple_freq, origin=9);
        cls.Window3 = Fixed_Size_Rolling_Window(cls.test_array_high_freq, cls.high_freq, origin=0.5);

        return;

    @classmethod
    def tearDownClass(cls) -> None:
        
        del cls.test_array_simple;
        del cls.test_array_simple_odd;
        del cls.test_array_high_freq;
        del cls.simple_freq;
        del cls.high_freq;
        del cls.Holdout1;
        del cls.Holdout2;
        del cls.Holdout3;
        del cls.Repeated_Holdout1;
        del cls.Repeated_Holdout2;
        del cls.Repeated_Holdout3;
        del cls.Update1;
        del cls.Update2;
        del cls.Update3;
        del cls.Recalibration1;
        del cls.Recalibration2;
        del cls.Recalibration3;
        del cls.Window1;
        del cls.Window2;
        del cls.Window3;

        return;

    def test_initialisation(self) -> None:

        """
        Test the class constructors and checks.
        """

        # Exceptions
        self.assertRaises(TypeError, Holdout, [0.1, 0.2, 0.3], 1);
        self.assertRaises(ValueError, Holdout, np.zeros(shape=(2, 2)), 1);
        self.assertRaises(ValueError, Holdout, np.array([1]), 1);
        self.assertRaises(TypeError, Holdout, self.test_array_simple, "a");
        self.assertRaises(ValueError, Holdout, self.test_array_simple, -1);

        self.assertRaises(TypeError, Holdout, self.test_array_simple, 1, 1);
        self.assertRaises(ValueError, Holdout, self.test_array_simple, 1, -0.5);
        self.assertRaises(ValueError, Holdout, self.test_array_simple, 1, 1.2);

        self.assertRaises(TypeError, Repeated_Holdout, self.test_array_simple, 1, 0.5);
        self.assertRaises(ValueError, Repeated_Holdout, self.test_array_simple, 1, -2);
        self.assertRaises(TypeError, Repeated_Holdout, self.test_array_simple, 1, 2, 2);
        self.assertRaises(ValueError, Repeated_Holdout, self.test_array_simple, 1, 2, [0.1, 0.2, 0.3]);
        self.assertRaises(TypeError, Repeated_Holdout, self.test_array_simple, 1, 2, ["a", 0.5]);
        self.assertRaises(ValueError, Repeated_Holdout, self.test_array_simple, 1, 2, [0.9, 0.5]);
        self.assertRaises(TypeError, Repeated_Holdout, self.test_array_simple, 1, 2, [0.1, 0.2], 0.1);

        self.assertRaises(TypeError, Rolling_Origin_Update, self.test_array_simple, 1, "a");
        self.assertRaises(ValueError, Rolling_Origin_Update, self.test_array_simple, 1, -0.1);
        self.assertRaises(ValueError, Rolling_Origin_Update, self.test_array_simple, 1, 1.1);
        self.assertRaises(ValueError, Rolling_Origin_Update, self.test_array_simple, 1, 0);
        self.assertRaises(ValueError, Rolling_Origin_Update, self.test_array_simple, 1, 20);

        self.assertRaises(TypeError, Rolling_Origin_Recalibration, self.test_array_simple, 1, "a");
        self.assertRaises(ValueError, Rolling_Origin_Recalibration, self.test_array_simple, 1, -0.1);
        self.assertRaises(ValueError, Rolling_Origin_Recalibration, self.test_array_simple, 1, 1.1);
        self.assertRaises(ValueError, Rolling_Origin_Recalibration, self.test_array_simple, 1, 0);
        self.assertRaises(ValueError, Rolling_Origin_Recalibration, self.test_array_simple, 1, 20);

        self.assertRaises(TypeError, Fixed_Size_Rolling_Window, self.test_array_simple, 1, "a");
        self.assertRaises(ValueError, Fixed_Size_Rolling_Window, self.test_array_simple, 1, -0.1);
        self.assertRaises(ValueError, Fixed_Size_Rolling_Window, self.test_array_simple, 1, 1.1);
        self.assertRaises(ValueError, Fixed_Size_Rolling_Window, self.test_array_simple, 1, 0);
        self.assertRaises(ValueError, Fixed_Size_Rolling_Window, self.test_array_simple, 1, 20);

        # Attribute correctness
        self.assertEqual(self.Holdout1.n_splits, 2);
        self.assertEqual(self.Holdout1.sampling_freq, self.simple_freq);
        self.assertEqual(self.Repeated_Holdout2.n_splits, 5);
        self.assertEqual(self.Repeated_Holdout1.sampling_freq, self.simple_freq);
        self.assertEqual(self.Update1.n_splits, 3);
        self.assertEqual(self.Update2.n_splits, 5);
        self.assertEqual(self.Update3.n_splits, 500);
        self.assertEqual(self.Update1.sampling_freq, self.simple_freq);
        self.assertEqual(self.Recalibration1.n_splits, 3);
        self.assertEqual(self.Recalibration2.n_splits, 5);
        self.assertEqual(self.Recalibration3.n_splits, 500);
        self.assertEqual(self.Recalibration1.sampling_freq, self.simple_freq);
        self.assertEqual(self.Window1.n_splits, 3);
        self.assertEqual(self.Window2.n_splits, 5);
        self.assertEqual(self.Window3.n_splits, 500);
        self.assertEqual(self.Window1.sampling_freq, self.simple_freq);

        return;

    def test_split(self) -> None:

        """
        Test the 'split' methods.
        """

        # Holdout
        holdout1_train = [0, 1, 2, 3, 4, 5, 6];
        holdout1_val = [7, 8, 9];
        holdout2_train = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        holdout2_val = [10, 11, 12, 13, 14];
        holdout3_train = np.arange(0, 500).tolist();
        holdout3_val = np.arange(500, 1000).tolist();
        indices = np.arange(0, 1000).tolist();
        holdout1_split = self.Holdout1.split();
        holdout2_split = self.Holdout2.split();
        holdout3_split = self.Holdout3.split();
        train1, val1 = next(holdout1_split);
        train2, val2 = next(holdout2_split);
        train3, val3 = next(holdout3_split);

        self.assertEqual(train1.tolist(), holdout1_train);
        self.assertEqual(val1.tolist(), holdout1_val);
        self.assertEqual(train2.tolist(), holdout2_train);
        self.assertEqual(val2.tolist(), holdout2_val);
        self.assertEqual(train3.tolist(), holdout3_train);
        self.assertEqual(val3.tolist(), holdout3_val);

        # Repeated Holdout
        holdout1_lower = int(np.round(0.7 * 10));
        holdout1_upper = int(np.round(0.8 * 10));
        holdout2_lower = 7;
        holdout2_upper = 10;
        holdout3_lower = int(np.round(0.5 * 1000));
        holdout3_upper = int(np.round(0.6 * 1000));

        for (_, val) in self.Repeated_Holdout1.split():

            self.assertGreaterEqual(val[0], holdout1_lower);
            self.assertLessEqual(val[0], holdout1_upper);

        for (_, val) in self.Repeated_Holdout2.split():

            self.assertGreaterEqual(val[0], holdout2_lower);
            self.assertLessEqual(val[0], holdout2_upper);

        for (_, val) in self.Repeated_Holdout3.split():

            self.assertGreaterEqual(val[0], holdout3_lower);
            self.assertLessEqual(val[0], holdout3_upper);

        # Validation sets
        rolling1_val = [[7, 8, 9], [8, 9], [9]];
        rolling2_val = [[10, 11, 12, 13, 14], [11, 12, 13, 14], [12, 13, 14], [13, 14], [14]];
        rolling3_val = [holdout3_val[ind:] for ind in range(500)];

        # Rolling Origin Update
        update1_train = [0, 1, 2, 3, 4, 5, 6];
        update2_train = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        update3_train = holdout3_train;

        for ind, (train, val) in enumerate(self.Update1.split()):

            self.assertListEqual(train.tolist(), update1_train);
            self.assertListEqual(val.tolist(), rolling1_val[ind]);

        for ind, (train, val) in enumerate(self.Update2.split()):

            self.assertListEqual(train.tolist(), update2_train);
            self.assertListEqual(val.tolist(), rolling2_val[ind]);

        for ind, (train, val) in enumerate(self.Update3.split()):

            self.assertListEqual(train.tolist(), update3_train);
            self.assertListEqual(val.tolist(), rolling3_val[ind]);

        # Rolling Origin Recalibration
        rec1_train = [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7, 8]];
        rec2_train = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], \
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], \
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], \
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]];
        rec3_train = [indices[:500+ind] for ind in range(500)];

        for ind, (train, val) in enumerate(self.Recalibration1.split()):

            self.assertListEqual(train.tolist(), rec1_train[ind]);
            self.assertListEqual(val.tolist(), rolling1_val[ind]);

        for ind, (train, val) in enumerate(self.Recalibration2.split()):

            self.assertListEqual(train.tolist(), rec2_train[ind]);
            self.assertListEqual(val.tolist(), rolling2_val[ind]);

        for ind, (train, val) in enumerate(self.Recalibration3.split()):

            self.assertListEqual(train.tolist(), rec3_train[ind]);
            self.assertListEqual(val.tolist(), rolling3_val[ind]);

        # Fixed-size Rolling Window
        window1_train = [[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8]];
        window2_train = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
                         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], \
                         [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], \
                         [3, 4, 5, 6, 7, 8, 9, 10, 11, 12], \
                         [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]];
        window3_train = [indices[ind:500+ind] for ind in range(500)];

        for ind, (train, val) in enumerate(self.Window1.split()):

            self.assertListEqual(train.tolist(), window1_train[ind]);
            self.assertListEqual(val.tolist(), rolling1_val[ind]);

        for ind, (train, val) in enumerate(self.Window2.split()):

            self.assertListEqual(train.tolist(), window2_train[ind]);
            self.assertListEqual(val.tolist(), rolling2_val[ind]);

        for ind, (train, val) in enumerate(self.Window3.split()):

            self.assertListEqual(train.tolist(), window3_train[ind]);
            self.assertListEqual(val.tolist(), rolling3_val[ind]);

        return;

    def test_info(self) -> None:

        """
        Test the 'info' method.
        """

        self.Holdout3.info();
        self.Repeated_Holdout3.info();
        self.Update3.info();
        self.Recalibration3.info();
        self.Window3.info();

        return;

    def test_statistics(self) -> None:

        """
        Test the 'statistics' method.
        """

        # Exceptions
        two_sample_series = np.zeros(shape=(2,));
        holdout_2sample = Holdout(two_sample_series, 1);
        repeated_holdout_2sample = Repeated_Holdout(two_sample_series, 1, 2);
        update_2sample = Rolling_Origin_Update(two_sample_series, 1);
        rec_2sample = Rolling_Origin_Recalibration(two_sample_series, 1);
        window_2sample = Fixed_Size_Rolling_Window(two_sample_series, 1);

        self.assertRaises(ValueError, holdout_2sample.statistics);
        self.assertRaises(ValueError, repeated_holdout_2sample.statistics);
        self.assertRaises(ValueError, update_2sample.statistics);
        self.assertRaises(ValueError, rec_2sample.statistics);
        self.assertRaises(ValueError, window_2sample.statistics);

        # Functionality
        columns = 13;
        column_list = ["Mean", "Median", "Min", "Max", "Variance", "P2P_amplitude", \
                       "Trend_slope", "Spectral_centroid", "Spectral_rolloff", "Spectral_entropy", \
                       "Strength_of_trend", "Mean_crossing_rate", "Median_crossing_rate"];

        # Holdout
        holdout1_full_stats, holdout1_training_stats, holdout1_validation_stats = self.Holdout1.statistics();
        holdout2_full_stats, holdout2_training_stats, holdout2_validation_stats = self.Holdout2.statistics();
        holdout3_full_stats, holdout3_training_stats, holdout3_validation_stats = self.Holdout3.statistics();

        self.assertEqual(holdout1_full_stats.shape[1], columns);
        self.assertListEqual(column_list, holdout1_full_stats.columns.tolist());

        self.assertEqual(holdout1_full_stats.shape[0], 1);
        self.assertEqual(holdout1_training_stats.shape[0], 1);
        self.assertEqual(holdout1_validation_stats.shape[0], 1);
        self.assertEqual(holdout2_full_stats.shape[0], 1);
        self.assertEqual(holdout2_training_stats.shape[0], 1);
        self.assertEqual(holdout2_validation_stats.shape[0], 1);
        self.assertEqual(holdout3_full_stats.shape[0], 1);
        self.assertEqual(holdout3_training_stats.shape[0], 1);
        self.assertEqual(holdout3_validation_stats.shape[0], 1);

        # Repeated Holdout
        repeated_holdout1_full_stats, repeated_holdout1_training_stats, repeated_holdout1_validation_stats = self.Repeated_Holdout1.statistics();
        repeated_holdout2_full_stats, repeated_holdout2_training_stats, repeated_holdout2_validation_stats = self.Repeated_Holdout2.statistics();
        repeated_holdout3_full_stats, repeated_holdout3_training_stats, repeated_holdout3_validation_stats = self.Repeated_Holdout3.statistics();

        self.assertEqual(repeated_holdout1_full_stats.shape[0], 1);
        self.assertEqual(repeated_holdout1_training_stats.shape[0], 2);
        self.assertEqual(repeated_holdout1_validation_stats.shape[0], 2);
        self.assertEqual(repeated_holdout2_full_stats.shape[0], 1);
        self.assertEqual(repeated_holdout2_training_stats.shape[0], 5);
        self.assertEqual(repeated_holdout2_validation_stats.shape[0], 5);
        self.assertEqual(repeated_holdout3_full_stats.shape[0], 1);
        self.assertEqual(repeated_holdout3_training_stats.shape[0], 10);
        self.assertEqual(repeated_holdout3_validation_stats.shape[0], 10);

        # Rolling Origin Update
        update1_full_stats, update1_training_stats, update1_validation_stats = self.Update1.statistics();
        update2_full_stats, update2_training_stats, update2_validation_stats = self.Update2.statistics();
        update3_full_stats, update3_training_stats, update3_validation_stats = self.Update3.statistics();

        self.assertEqual(update1_full_stats.shape[0], 1);
        self.assertEqual(update1_training_stats.shape[0], 1);
        self.assertEqual(update1_validation_stats.shape[0], 2);
        self.assertEqual(update2_full_stats.shape[0], 1);
        self.assertEqual(update2_training_stats.shape[0], 1);
        self.assertEqual(update2_validation_stats.shape[0], 4);
        self.assertEqual(update3_full_stats.shape[0], 1);
        self.assertEqual(update3_training_stats.shape[0], 1);
        self.assertEqual(update3_validation_stats.shape[0], 499);

        # Rolling Origin Recalibration
        rec1_full_stats, rec1_training_stats, rec1_validation_stats = self.Recalibration1.statistics();
        rec2_full_stats, rec2_training_stats, rec2_validation_stats = self.Recalibration2.statistics();
        rec3_full_stats, rec3_training_stats, rec3_validation_stats = self.Recalibration3.statistics();

        self.assertEqual(rec1_full_stats.shape[0], 1);
        self.assertEqual(rec1_training_stats.shape[0], 3);
        self.assertEqual(rec1_validation_stats.shape[0], 2);
        self.assertEqual(rec2_full_stats.shape[0], 1);
        self.assertEqual(rec2_training_stats.shape[0], 5);
        self.assertEqual(rec2_validation_stats.shape[0], 4);
        self.assertEqual(rec3_full_stats.shape[0], 1);
        self.assertEqual(rec3_training_stats.shape[0], 500);
        self.assertEqual(rec3_validation_stats.shape[0], 499);

        # Fixed-size Rolling Window
        window1_full_stats, window1_training_stats, window1_validation_stats = self.Window1.statistics();
        window2_full_stats, window2_training_stats, window2_validation_stats = self.Window2.statistics();
        window3_full_stats, window3_training_stats, window3_validation_stats = self.Window3.statistics();

        self.assertEqual(window1_full_stats.shape[0], 1);
        self.assertEqual(window1_training_stats.shape[0], 3);
        self.assertEqual(window1_validation_stats.shape[0], 2);
        self.assertEqual(window2_full_stats.shape[0], 1);
        self.assertEqual(window2_training_stats.shape[0], 5);
        self.assertEqual(window2_validation_stats.shape[0], 4);
        self.assertEqual(window3_full_stats.shape[0], 1);
        self.assertEqual(window3_training_stats.shape[0], 500);
        self.assertEqual(window3_validation_stats.shape[0], 499);

        # TODO: add 'reset_index' to feature data frames after concatenation takes place (just for convenience).

        return;

    def test_plot(self) -> None:

        """
        Test the 'plot' method.
        """

        height = 10;
        width = 10;

        self.Holdout1.plot(height, width);
        self.Holdout2.plot(height, width);
        self.Holdout3.plot(height, width);
        self.Repeated_Holdout1.plot(height, width);
        self.Repeated_Holdout2.plot(height, width);
        self.Repeated_Holdout3.plot(height + 10, width + 10);
        self.Update1.plot(height, width);
        self.Update2.plot(height, width);
        self.Recalibration1.plot(height, width);
        self.Recalibration2.plot(height, width);
        self.Window1.plot(height, width);
        self.Window2.plot(height, width);

        return;

if __name__ == "__main__":

    unittest.main();