"""
This file contains unit tests targetting the 'OOS' module.
"""

import unittest
from tsvalidation.data_generation import frequency_modulation as dgu
from tsvalidation.data_generation.time_series_generation import TimeSeriesGenerator
from tsvalidation.data_generation import time_series_functions as tsf

import numpy as np


class TestTimeSeriesGenerator(unittest.TestCase):

    def setUp(self) -> None:
        self.l0 = 1
        self.l1 = 1000
        self.l2 = 2
        self.l3 = 5
        self.nl1 = 0.1
        self.w1 = None
        self.w2 = [0.1, 0.3, 0.9, 0.1]
        linear_parameters = {"max_interval_size": 1, "slope": 5, "intercept": [5, 30]}
        exp_parameters = {
            "max_interval_size": (1, 2),
            "decay_rate": [1, 25],
            "initial_value": [1, 25],
        }
        freq_options = (
            dgu.FrequencyModulationLinear(1, 20),
            dgu.FrequencyModulationWithStep(10, 0.8),
        )
        sin_parameters = {
            "max_interval_size": (1, 2),
            "amplitude": [1, 3],
            "frequency": freq_options,
        }
        impulse_parameters = {"idx": (500, 600), "constant": [5, 10]}
        indicator_parameters = {"start_index": (700, 600), "end_index": (800, 900)}
        self.funcs0 = [tsf.linear_ts]
        self.funcs1 = [
            tsf.linear_ts,
            tsf.indicator_ts,
            tsf.scaled_unit_impulse_function_ts,
            tsf.exponential_ts,
        ]
        self.funcs2 = [
            tsf.linear_ts,
            tsf.indicator_ts,
            tsf.frequency_varying_sinusoid_ts,
            tsf.scaled_unit_impulse_function_ts,
        ]
        self.funcs3 = [tsf.linear_ts, tsf.indicator_ts]

        self.params0 = [{"max_interval_size": 1, "slope": 1, "intercept": 0}]
        self.params1 = [
            linear_parameters,
            indicator_parameters,
            impulse_parameters,
            exp_parameters,
        ]
        self.params2 = [
            linear_parameters,
            indicator_parameters,
            sin_parameters,
            impulse_parameters,
        ]
        self.params3 = [
            {"max_interval_size": 4, "slope": 1, "intercept": 0},
            {"start_index": 1, "end_index": 2},
        ]

        self.ts0 = TimeSeriesGenerator(
            functions=self.funcs1,
            length=self.l3,
            noise_level=self.nl1,
            weights=self.w1,
            parameter_values=self.params1,
        )
        self.ts1 = TimeSeriesGenerator(
            functions=self.funcs1,
            length=self.l1,
            noise_level=self.nl1,
            weights=self.w1,
            parameter_values=self.params1,
        )
        self.ts2 = TimeSeriesGenerator(
            functions=self.funcs2,
            length=self.l2,
            noise_level=self.nl1,
            weights=self.w2,
            parameter_values=self.params2,
        )
        self.ts3 = TimeSeriesGenerator(
            functions=self.funcs1,
            length=self.l3,
            noise_level=self.nl1,
            weights=self.w1,
            parameter_values=self.params1,
        )

        self.ts4 = TimeSeriesGenerator(
            functions=self.funcs3,
            length=5,
            noise_level=0,
            weights=None,
            parameter_values=self.params3,
        )

        self.ts5 = TimeSeriesGenerator(
            functions=self.funcs3,
            length=self.l0,
            noise_level=0,
            weights=None,
            parameter_values=self.params3,
        )

        return

    def test_initialisation(self) -> None:
        """
        Test the class constructors and checks.
        """

        # Exceptions
        # Exception: different functions/weights/dicts of params
        self.assertRaises(
            ValueError,
            TimeSeriesGenerator,
            self.funcs1,
            self.l1,
            self.nl1,
            self.w1,
            self.params1[:-1],
        )
        self.assertRaises(
            ValueError,
            TimeSeriesGenerator,
            self.funcs1,
            self.l1,
            self.nl1,
            self.w2[:-1],
            self.params1,
        )
        self.assertRaises(
            ValueError,
            TimeSeriesGenerator,
            self.funcs1[:-1],
            self.l1,
            self.nl1,
            self.w2,
            self.params1,
        )
        # Exception: TypeErrors
        self.assertRaises(
            TypeError,
            TimeSeriesGenerator,
            [1, 2, 3, 4],
            self.l1,
            self.nl1,
            self.w1,
            self.params1[:-1],
        )
        self.assertRaises(
            TypeError,
            TimeSeriesGenerator,
            "functions",
            self.l1,
            self.nl1,
            self.w2[:-1],
            self.params1,
        )
        self.assertRaises(
            TypeError,
            TimeSeriesGenerator,
            self.funcs1,
            "100",
            self.nl1,
            self.w2,
            self.params1,
        )
        self.assertRaises(
            TypeError,
            TimeSeriesGenerator,
            self.funcs1,
            self.l1,
            np.array(0.1),
            self.w2,
            self.params1,
        )
        self.assertRaises(
            TypeError,
            TimeSeriesGenerator,
            self.funcs1,
            self.l1,
            self.nl1,
            {},
            self.params1,
        )
        self.assertRaises(
            TypeError,
            TimeSeriesGenerator,
            self.funcs1,
            self.l1,
            self.nl1,
            self.w2,
            [0, 1, 2],
        )
        # Exception: ValueErrors
        self.assertRaises(
            ValueError,
            TimeSeriesGenerator,
            self.funcs1,
            -1,
            self.nl1,
            self.w2,
            self.params1,
        )
        self.assertRaises(
            ValueError,
            TimeSeriesGenerator,
            self.funcs1,
            self.l1,
            -1,
            self.w2,
            self.params1,
        )

        # Attribute correctness
        self.assertEqual(self.ts1.time_series, [])
        self.assertEqual(self.ts2.time_series, [])
        self.assertEqual(self.ts3.time_series, [])
        self.assertEqual(self.ts4.time_series, [])
        self.assertEqual(self.ts5.time_series, [])

    def test_generate(self) -> None:
        nb1 = 1
        nb2 = 2
        s1 = 1
        s2 = 92
        ts4_out = [0.0, 2.0, 3.0, 3.0, 4.0]
        ts5_out = [0.0]
        self.assertListEqual(list(self.ts4.generate(nb1, s1)[0]), ts4_out)
        self.assertListEqual(list(self.ts4.generate(nb2, s1)[0]), ts4_out)
        self.assertListEqual(list(self.ts4.generate(nb2, s2)[0]), ts4_out)
        self.assertListEqual(list(self.ts5.generate(nb1, s1)[0]), ts5_out)

    def test_plot(self) -> None:
        """
        Test the 'plot' method.
        """
        self.ts1.generate(5, 1)
        self.ts1.plot()
        self.ts2.generate(5, 1)
        self.ts2.plot()
        self.ts3.generate(5, 1)
        self.ts3.plot()
        self.ts4.generate(5, 1)
        self.ts4.plot()
        self.ts5.generate(5, 1)
        self.ts5.plot()
        return


if __name__ == "__main__":

    unittest.main()
