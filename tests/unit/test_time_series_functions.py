import unittest
import numpy as np
from tsvalidation.data_generation.time_series_functions import (
    sinusoid_ts,
    scaled_right_indicator_ts,
    scaled_unit_impulse_function_ts,
    frequency_varying_sinusoid_ts,
    indicator_ts,
    linear_ts,
    exponential_ts,
    arma_ts,
    nonlinear_ar_ts,
)
from tsvalidation.data_generation.frequency_modulation import FrequencyModulationLinear


class TestTimeSeriesFunctions(unittest.TestCase):

    def test_sinusoid_ts(self):
        n1 = 8 + 1
        frequency = 1 / (2 * np.pi)
        amplitude = 1
        ts1 = sinusoid_ts(
            n1, 4 * np.pi, frequency=frequency, amplitude=amplitude, phase=0
        )
        ts2 = sinusoid_ts(
            n1, 4 * np.pi, frequency=frequency, amplitude=amplitude, phase=-2 * np.pi
        )
        result = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0])

        # Exception
        self.assertRaises(ValueError, sinusoid_ts, -1, 1)
        self.assertRaises(TypeError, sinusoid_ts, "a", 1)
        self.assertRaises(ValueError, sinusoid_ts, 1, -1)
        self.assertRaises(TypeError, sinusoid_ts, 1, "a")

        # Functionality
        self.assertEqual(len(ts1), n1)
        self.assertTrue(np.all(np.abs(ts1) <= amplitude))
        np.testing.assert_array_almost_equal(ts1, result)
        np.testing.assert_array_almost_equal(ts1, ts2)

    def test_frequency_varying_sinusoid_ts(self):
        n1 = 8 + 1
        m1 = 4 * np.pi
        fm = FrequencyModulationLinear(freq_init=1 / (2 * np.pi), slope=0)
        amp = 1
        ts1 = frequency_varying_sinusoid_ts(
            n1, m1, frequency=fm, amplitude=amp, phase=0
        )
        ts2 = frequency_varying_sinusoid_ts(
            n1,
            m1,
            frequency=fm,
            amplitude=amp,
            phase=-2 * np.pi,
        )
        result = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0])

        # Exception
        self.assertRaises(ValueError, frequency_varying_sinusoid_ts, -1, 1, fm)
        self.assertRaises(TypeError, frequency_varying_sinusoid_ts, "a", 1, fm)
        self.assertRaises(ValueError, frequency_varying_sinusoid_ts, 1, -1, fm)
        self.assertRaises(TypeError, frequency_varying_sinusoid_ts, 1, "a", fm)

        # Functionality
        self.assertEqual(len(ts1), n1)
        self.assertTrue(np.all(np.abs(ts1) <= amp))
        np.testing.assert_array_almost_equal(ts1, result)
        np.testing.assert_array_almost_equal(ts1, ts2)

    def test_indicator_ts(self):
        n1 = 10
        ts1 = indicator_ts(n1, 2, 4)
        result = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Exception
        self.assertRaises(ValueError, indicator_ts, -1, 1, 10)
        self.assertRaises(TypeError, indicator_ts, "a", 1, 10)
        self.assertRaises(ValueError, indicator_ts, 1, -1, 10)
        self.assertRaises(TypeError, indicator_ts, 1, "a", 10)
        self.assertRaises(ValueError, indicator_ts, 1, 1, -1)
        self.assertRaises(TypeError, indicator_ts, 1, 1, "a")

        # Functionality
        self.assertEqual(len(ts1), n1)
        np.testing.assert_array_equal(ts1, result)

    def test_linear_ts(self):
        n1 = 10 + 1
        m1 = 10
        ts1 = linear_ts(n1, m1, slope=1, intercept=0)
        r1 = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        ts2 = linear_ts(n1, m1, slope=2, intercept=2)
        r2 = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0])

        # Exception
        self.assertRaises(ValueError, linear_ts, -1, 1)
        self.assertRaises(TypeError, linear_ts, "a", 1)

        # Functionality
        self.assertEqual(len(ts1), n1)
        np.testing.assert_array_equal(ts1, r1)
        np.testing.assert_array_equal(ts2, r2)

    def test_exponential_ts(self):
        n1 = 4 + 1
        m1 = 4
        ts1 = exponential_ts(n1, m1)
        r1 = np.array([np.e**0, np.e ** (-1), np.e ** (-2), np.e ** (-3), np.e ** (-4)])
        ts2 = exponential_ts(n1, m1, decay_rate=2, initial_value=0.5)
        r2 = 0.5 * np.array(
            [np.e**0, np.e ** (-2), np.e ** (-4), np.e ** (-6), np.e ** (-8)]
        )

        # Exception
        self.assertRaises(ValueError, exponential_ts, -1, 1)
        self.assertRaises(TypeError, exponential_ts, "a", 1)

        # Functionality
        self.assertEqual(len(ts1), n1)
        np.testing.assert_array_almost_equal(ts1, r1)
        np.testing.assert_array_almost_equal(ts2, r2)

    def test_arma_ts(self):
        n1 = 5
        lags = 2
        max_root = 1.5
        ar = True
        ma = True

        ts1 = arma_ts(n1, lags, max_root, ar, ma)

        # Exception
        self.assertRaises(ValueError, arma_ts, -1, 1, 2)
        self.assertRaises(TypeError, arma_ts, "a", 1, 2)
        self.assertRaises(ValueError, arma_ts, 1, 1, 2, ar=False, ma=False)
        self.assertRaises(ValueError, arma_ts, 1, 1, 0.5)

        # Functionality
        self.assertEqual(len(ts1), n1)

    def test_nonlinear_ar_ts(self):
        number_samples = 100
        init_array = np.zeros(2)
        params = [0.5, -0.3]
        func_idxs = [0, 1]

        ts = nonlinear_ar_ts(number_samples, init_array, params, func_idxs)
        self.assertRaises(
            ValueError, nonlinear_ar_ts, -1, init_array, params, func_idxs
        )
        self.assertRaises(
            TypeError, nonlinear_ar_ts, "a", init_array, params, func_idxs
        )
        self.assertRaises(
            ValueError, nonlinear_ar_ts, -1, init_array, [0, 1, 2], func_idxs
        )

        self.assertEqual(len(ts), number_samples)
        self.assertIsInstance(ts, np.ndarray)


if __name__ == "__main__":
    unittest.main()
