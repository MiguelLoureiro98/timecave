"""
This file contains unit tests targetting the 'OOS' module.
"""

import unittest
from tsvalidation.data_generation import frequency_modulation as dgu
from tsvalidation.data_generation.time_series_generation import TimeSeriesGenerator
from tsvalidation.data_generation import time_series_functions as tsf

import numpy as np


class TestTimeSeriesGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.l1 = 1000
        cls.l2 = 2
        cls.l3 = 100
        cls.nl1 = 0.1
        cls.w1 = None
        cls.w2 = [0.1, 0.3, 0.9, 0.1]
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
        cls.funcs1 = [tsf.linear_ts]
        cls.funcs1 = [
            tsf.linear_ts,
            tsf.indicator_ts,
            tsf.scaled_unit_impulse_function_ts,
            tsf.exponential_ts,
        ]
        cls.funcs2 = [
            tsf.linear_ts,
            tsf.indicator_ts,
            tsf.frequency_varying_sinusoid_ts,
            tsf.scaled_unit_impulse_function_ts,
        ]
        cls.params0 = [{"max_interval_size": 1, "slope": 1, "intercept": 0}]
        cls.params1 = [
            linear_parameters,
            indicator_parameters,
            impulse_parameters,
            exp_parameters,
        ]
        cls.params2 = [
            linear_parameters,
            indicator_parameters,
            sin_parameters,
            impulse_parameters,
        ]
        cls.ts0 = TimeSeriesGenerator(
            functions=cls.funcs1,
            length=cls.l3,
            noise_level=cls.nl1,
            weights=cls.w1,
            parameter_values=cls.params1,
        )
        cls.ts1 = TimeSeriesGenerator(
            functions=cls.funcs1,
            length=cls.l1,
            noise_level=cls.nl1,
            weights=cls.w1,
            parameter_values=cls.params1,
        )
        cls.ts2 = TimeSeriesGenerator(
            functions=cls.funcs2,
            length=cls.l2,
            noise_level=cls.nl1,
            weights=cls.w2,
            parameter_values=cls.params2,
        )
        cls.ts3 = TimeSeriesGenerator(
            functions=cls.funcs1,
            length=cls.l3,
            noise_level=cls.nl1,
            weights=cls.w1,
            parameter_values=cls.params1,
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

    def test_generate(self) -> None:
        nb1 = 1
        nb2 = 2
        nb3 = 0
        s1 = 1
        s2 = 92
        s3 = 39
        self.ts1.generate(nb1, s1)


if __name__ == "__main__":

    unittest.main()
