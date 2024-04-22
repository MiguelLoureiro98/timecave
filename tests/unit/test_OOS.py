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
        holdout1_lower = 0.7;
        holdout1_upper = 0.8;
        holdout2_lower = 7;
        holdout2_upper = 10;
        holdout3_lower = 0.5;
        holdout3_upper = 0.6;

        for (train, val) in self.Repeated_Holdout1.split():

            pass

        # Validation sets
        rolling1_val = [[7, 8, 9], [8, 9], [9]];
        rolling2_val = [[10, 11, 12, 13, 14], [11, 12, 13, 14], [12, 13, 14], [13, 14], [14]];
        rolling3_val = [holdout3_val[ind:] for ind in range(500)];

        # Rolling Origin Update
        update1_train = [0, 1, 2, 3, 4, 5, 6];
        update2_train = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        update3_train = holdout3_train;

        for (train, val) in self.Update1.split():

            self.assertEqual(train, update1_train);
            self.assertEqual(val, rolling1_val);

        for (train, val) in self.Update2.split():

            self.assertEqual(train, update2_train);
            self.assertEqual(val, rolling2_val);

        for (train, val) in self.Update3.split():

            self.assertEqual(train, update3_train);
            self.assertEqual(val, rolling3_val);

        # Rolling Origin Recalibration
        rec1_train = [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7, 8]];
        rec2_train = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], \
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], \
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], \
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]];
        rec3_train = [indices[:500+ind] for ind in range(500)];

        for (train, val) in self.Recalibration1.split():

            self.assertEqual(train, rec1_train);
            self.assertEqual(val, rolling1_val);

        for (train, val) in self.Recalibration2.split():

            self.assertEqual(train, rec2_train);
            self.assertEqual(val, rolling2_val);

        for (train, val) in self.Recalibration3.split():

            self.assertEqual(train, rec3_train);
            self.assertEqual(val, rolling3_val);

        # Fixed-size Rolling Window
        window1_train = [[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8]];
        window2_train = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
                         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], \
                         [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], \
                         [3, 4, 5, 6, 7, 8, 9, 10, 11, 12], \
                         [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]];
        window3_train = [indices[ind:500+ind] for ind in range(500)];

        for (train, val) in self.Window1.split():

            self.assertEqual(train, window1_train);
            self.assertEqual(val, rolling1_val);

        for (train, val) in self.Window2.split():

            self.assertEqual(train, window2_train);
            self.assertEqual(val, rolling2_val);

        for (train, val) in self.Window3.split():

            self.assertEqual(train, window3_train);
            self.assertEqual(val, rolling3_val);

        return;

    def test_info(self) -> None:

        pass

    def test_statistics(self) -> None:

        pass

    def test_plot(self) -> None:

        pass

if __name__ == "__main__":

    unittest.main();