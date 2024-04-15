"""
This file contains tests targetting the 'data_characteristics' module.
"""

import unittest
from tsvalidation.data_characteristics import get_features, strength_of_trend, mean_crossing_rate, median_crossing_rate
import numpy as np
import pandas as pd

class TestDataChar(unittest.TestCase):

    def setUp(self) -> None:

        self.ts_numpy = np.array([1, 2, 3, 4, 5, -5, -4, -3, -2, -1]);
        self.ts_pandas = pd.Series([1, 2, 3, 4, 5, -5, -4, -3, -2, -1]);
        self.std_numpy = self.ts_numpy.std();
        self.diff_std_numpy = np.diff(self.ts_numpy).std();
        self.std_pandas = self.ts_pandas.std();
        self.diff_std_pandas = self.ts_pandas.diff().dropna().std();

        return;
    
    def tearDown(self) -> None:

        del self.ts_numpy;
        del self.ts_pandas;
        del self.std_numpy;
        del self.diff_std_numpy;
        del self.std_pandas;
        del self.diff_std_pandas;

        return;

    def test_sot(self) -> None:

        """
        Test the 'strength_of_trend' function.
        """

        res_numpy = strength_of_trend(self.ts_numpy);
        res_pandas = strength_of_trend(self.ts_pandas);

        # Exceptions
        self.assertRaises(TypeError, strength_of_trend, "a");

        # Functionality
        self.assertEqual(res_numpy, self.std_numpy / self.diff_std_numpy);
        self.assertEqual(res_pandas, self.std_pandas / self.diff_std_pandas);

        return;

    def test_crossing_rates(self) -> None:

        """
        Test the 'mean_crossing_rate' and 'median_crossing_rate' functions.
        """

        real_crossing_rate = 1 / 9;
        res_mean_numpy = mean_crossing_rate(self.ts_numpy);
        res_mean_pandas = mean_crossing_rate(self.ts_pandas);
        res_median_numpy = median_crossing_rate(self.ts_numpy);
        res_median_pandas = median_crossing_rate(self.ts_pandas);

        # Exceptions
        self.assertRaises(TypeError, mean_crossing_rate, "a");
        self.assertRaises(TypeError, median_crossing_rate, "a");

        # Functionality
        self.assertEqual(res_mean_numpy, real_crossing_rate);
        self.assertEqual(res_mean_pandas, real_crossing_rate);
        self.assertEqual(res_median_numpy, real_crossing_rate);
        self.assertEqual(res_median_pandas, real_crossing_rate);

        return;

    def test_all(self) -> None:

        """
        Test the 'get_features' function.
        """

        n_features = 13;
        mean = 0.0;
        median = 0.0;
        minimum = -5;
        maximum = 5;
        var_numpy = self.ts_numpy.var();
        #var_pandas = self.ts_pandas.var();
        p2p = 10;

        res_numpy = get_features(self.ts_numpy, 1000);
        res_pandas = get_features(self.ts_pandas, 1000);

        # Exceptions
        self.assertRaises(TypeError, get_features, "a", 1);
        self.assertRaises(TypeError, get_features, self.ts_numpy, "a");
        self.assertRaises(ValueError, get_features, self.ts_numpy, -1);
        self.assertRaises(ValueError, get_features, self.ts_numpy, 0);

        # Functionality
        self.assertEqual(res_numpy.shape[1], n_features);
        self.assertEqual(res_pandas.shape[1], n_features);

        self.assertEqual(res_numpy["0_Mean"].to_numpy().item(), mean);
        self.assertEqual(res_numpy["0_Median"].to_numpy().item(), median);
        self.assertEqual(res_numpy["0_Min"].to_numpy().item(), minimum);
        self.assertEqual(res_numpy["0_Max"].to_numpy().item(), maximum);
        self.assertEqual(res_numpy["0_Peak to peak distance"].to_numpy().item(), p2p);
        self.assertEqual(res_numpy["0_Variance"].to_numpy().item(), var_numpy);
        self.assertEqual(res_pandas["0_Variance"].to_numpy().item(), var_numpy);

        sloped_series = np.array([1, 2, 3, 4, 5]);
        res_slope = get_features(sloped_series, 1000);
        self.assertAlmostEqual(res_slope["Trend_slope"].to_numpy().item(), 1);

        dt = 0.001;
        fs = 1 / dt;
        t = np.arange(0, 10, dt);
        sinusoid = np.sin(2 * np.pi * 2 * t);
        sinusoid2 = sinusoid + np.sin(2 * np.pi * 4 * t);
        centroid1 = 2.0;
        centroid2 = 3.0;
        roll_off1 = 2.0;
        roll_off2 = 4.0;
        entropy1 = 0.0;
        entropy2 = 0.0813802;
        freq_feat1 = get_features(sinusoid, fs);
        freq_feat2 = get_features(sinusoid2, fs);

        self.assertAlmostEqual(freq_feat1["Spectral_centroid"].to_numpy().item(), centroid1);
        self.assertAlmostEqual(freq_feat2["Spectral_centroid"].to_numpy().item(), centroid2);
        self.assertAlmostEqual(freq_feat1["Spectral_rolloff"].to_numpy().item(), roll_off1);
        self.assertAlmostEqual(freq_feat2["Spectral_rolloff"].to_numpy().item(), roll_off2);
        self.assertAlmostEqual(freq_feat1["Spectral_entropy"].to_numpy().item(), entropy1);
        self.assertAlmostEqual(freq_feat2["Spectral_entropy"].to_numpy().item(), entropy2);

        return;

if __name__ == "__main__":

    unittest.main();