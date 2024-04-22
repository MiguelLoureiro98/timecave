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

        

        pass

    def test_info(self) -> None:

        pass

    def test_statistics(self) -> None:

        pass

    def test_plot(self) -> None:

        pass

if __name__ == "__main__":

    unittest.main();