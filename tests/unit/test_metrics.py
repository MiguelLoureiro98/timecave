"""
This file contains tests targetting the 'validation_strategy_metrics' module.
"""

import unittest
from timecave.validation_strategy_metrics import PAE, APAE, RPAE, RAPAE, MC_metric, under_over_estimation
import numpy as np

class TestMetrics(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        cls.estimated = [1, 2, 3, 4, 5];
        cls.test = [5, 4, 3, 2, 1];
        cls.invalid_test = 0;
        cls.PAE_res = [-4, -2, 0, 2, 4];
        cls.APAE_res = [4, 2, 0, 2, 4];
        cls.RPAE_res = [-0.8, -0.5, 0, 1, 4];
        cls.RAPAE_res = [0.8, 0.5, 0, 1, 4];

        PAE_array = np.array(cls.PAE_res);
        APAE_array = np.array(cls.APAE_res);
        RPAE_array = np.array(cls.RPAE_res);
        RAPAE_array = np.array(cls.RAPAE_res);

        cls.MC_PAE_res = {"Mean": PAE_array.mean(),
                          "Median": np.median(PAE_array),
                          "1st_Quartile": np.quantile(PAE_array, 0.25),
                          "3rd_Quartile": np.quantile(PAE_array, 0.75),
                          "Minimum": PAE_array.min(),
                          "Maximum": PAE_array.max(),
                          "Standard_deviation": PAE_array.std()};
        
        cls.MC_APAE_res = {"Mean": APAE_array.mean(),
                           "Median": np.median(APAE_array),
                           "1st_Quartile": np.quantile(APAE_array, 0.25),
                           "3rd_Quartile": np.quantile(APAE_array, 0.75),
                           "Minimum": APAE_array.min(),
                           "Maximum": APAE_array.max(),
                           "Standard_deviation": APAE_array.std()};
        
        cls.MC_RPAE_res = {"Mean": RPAE_array.mean(),
                           "Median": np.median(RPAE_array),
                           "1st_Quartile": np.quantile(RPAE_array, 0.25),
                           "3rd_Quartile": np.quantile(RPAE_array, 0.75),
                           "Minimum": RPAE_array.min(),
                           "Maximum": RPAE_array.max(),
                           "Standard_deviation": RPAE_array.std()};
        
        cls.MC_RAPAE_res = {"Mean": RAPAE_array.mean(),
                            "Median": np.median(RAPAE_array),
                            "1st_Quartile": np.quantile(RAPAE_array, 0.25),
                            "3rd_Quartile": np.quantile(RAPAE_array, 0.75),
                            "Minimum": RAPAE_array.min(),
                            "Maximum": RAPAE_array.max(),
                            "Standard_deviation": RAPAE_array.std()};
        
        cls.under_PAE_res = {"Mean": PAE_array[:2].mean(),
                             "Median": np.median(PAE_array[:2]),
                             "1st_Quartile": np.quantile(PAE_array[:2], 0.25),
                             "3rd_Quartile": np.quantile(PAE_array[:2], 0.75),
                             "Minimum": PAE_array[:2].min(),
                             "Maximum": PAE_array[:2].max(),
                             "Standard_deviation": PAE_array[:2].std(),
                             "N": 2,
                             "%": 40};
        
        cls.over_PAE_res = {"Mean": PAE_array[-2:].mean(),
                            "Median": np.median(PAE_array[-2:]),
                            "1st_Quartile": np.quantile(PAE_array[-2:], 0.25),
                            "3rd_Quartile": np.quantile(PAE_array[-2:], 0.75),
                            "Minimum": PAE_array[-2:].min(),
                            "Maximum": PAE_array[-2:].max(),
                            "Standard_deviation": PAE_array[-2:].std(),
                            "N": 2,
                            "%": 40};
        
        cls.under_APAE_res = {"Mean": APAE_array[:2].mean(),
                              "Median": np.median(APAE_array[:2]),
                              "1st_Quartile": np.quantile(APAE_array[:2], 0.25),
                              "3rd_Quartile": np.quantile(APAE_array[:2], 0.75),
                              "Minimum": APAE_array[:2].min(),
                              "Maximum": APAE_array[:2].max(),
                              "Standard_deviation": APAE_array[:2].std(),
                              "N": 2,
                              "%": 40};
        
        cls.over_APAE_res = {"Mean": APAE_array[-2:].mean(),
                             "Median": np.median(APAE_array[-2:]),
                             "1st_Quartile": np.quantile(APAE_array[-2:], 0.25),
                             "3rd_Quartile": np.quantile(APAE_array[-2:], 0.75),
                             "Minimum": APAE_array[-2:].min(),
                             "Maximum": APAE_array[-2:].max(),
                             "Standard_deviation": APAE_array[-2:].std(),
                             "N": 2,
                             "%": 40};
        
        cls.under_RPAE_res = {"Mean": RPAE_array[:2].mean(),
                              "Median": np.median(RPAE_array[:2]),
                              "1st_Quartile": np.quantile(RPAE_array[:2], 0.25),
                              "3rd_Quartile": np.quantile(RPAE_array[:2], 0.75),
                              "Minimum": RPAE_array[:2].min(),
                              "Maximum": RPAE_array[:2].max(),
                              "Standard_deviation": RPAE_array[:2].std(),
                              "N": 2,
                              "%": 40};
        
        cls.over_RPAE_res = {"Mean": RPAE_array[-2:].mean(),
                             "Median": np.median(RPAE_array[-2:]),
                             "1st_Quartile": np.quantile(RPAE_array[-2:], 0.25),
                             "3rd_Quartile": np.quantile(RPAE_array[-2:], 0.75),
                             "Minimum": RPAE_array[-2:].min(),
                             "Maximum": RPAE_array[-2:].max(),
                             "Standard_deviation": RPAE_array[-2:].std(),
                             "N": 2,
                             "%": 40};
        
        cls.under_RAPAE_res = {"Mean": RAPAE_array[:2].mean(),
                               "Median": np.median(RAPAE_array[:2]),
                               "1st_Quartile": np.quantile(RAPAE_array[:2], 0.25),
                               "3rd_Quartile": np.quantile(RAPAE_array[:2], 0.75),
                               "Minimum": RAPAE_array[:2].min(),
                               "Maximum": RAPAE_array[:2].max(),
                               "Standard_deviation": RAPAE_array[:2].std(),
                               "N": 2,
                               "%": 40};
        
        cls.over_RAPAE_res = {"Mean": RAPAE_array[-2:].mean(),
                              "Median": np.median(RAPAE_array[-2:]),
                              "1st_Quartile": np.quantile(RAPAE_array[-2:], 0.25),
                              "3rd_Quartile": np.quantile(RAPAE_array[-2:], 0.75),
                              "Minimum": RAPAE_array[-2:].min(),
                              "Maximum": RAPAE_array[-2:].max(),
                              "Standard_deviation": RAPAE_array[-2:].std(),
                              "N": 2,
                              "%": 40};
    
        return;
    
    @classmethod
    def tearDownClass(cls) -> None:

        del cls.estimated;
        del cls.test;
        del cls.invalid_test;
        del cls.PAE_res;
        del cls.APAE_res;
        del cls.RPAE_res;
        del cls.RAPAE_res;
        del cls.MC_PAE_res;
        del cls.MC_APAE_res;
        del cls.MC_RPAE_res;
        del cls.MC_RAPAE_res;
        del cls.under_PAE_res;
        del cls.over_PAE_res;
        del cls.under_APAE_res;
        del cls.over_APAE_res;
        del cls.under_RPAE_res;
        del cls.over_RPAE_res;
        del cls.under_RAPAE_res;
        del cls.over_RAPAE_res;

        return;
    
    def test_PAE(self) -> None:

        """
        Test the PAE function.
        """

        res = [PAE(estimated, test) for (estimated, test) in zip(self.estimated, self.test)];

        self.assertListEqual(res, self.PAE_res);

        return;

    def test_APAE(self) -> None:

        """
        Test the APAE function.
        """

        res = [APAE(estimated, test) for (estimated, test) in zip(self.estimated, self.test)];

        self.assertListEqual(res, self.APAE_res);

        return;

    def test_RPAE(self) -> None:

        """
        Test the RPAE function.
        """

        # Exceptions
        self.assertRaises(ValueError, RPAE, 1, 0);

        # Functionality
        res = [RPAE(estimated, test) for (estimated, test) in zip(self.estimated, self.test)];

        self.assertListEqual(res, self.RPAE_res);

        return;

    def test_RAPAE(self) -> None:

        """
        Test the RAPAE function.
        """

        # Exceptions
        self.assertRaises(ValueError, RAPAE, 1, 0);

        # Functionality
        res = [RAPAE(estimated, test) for (estimated, test) in zip(self.estimated, self.test)];

        self.assertListEqual(res, self.RAPAE_res);

        return;

    def test_MC_metric(self) -> None:

        """
        Test the MC_metric function.
        """

        # Exceptions
        self.assertRaises(ValueError, MC_metric, [1, 2, 3], [1, 2], PAE);

        # Functionality
        PAE_res = MC_metric(self.estimated, self.test, PAE);
        APAE_res = MC_metric(self.estimated, self.test, APAE);
        RPAE_res = MC_metric(self.estimated, self.test, RPAE);
        RAPAE_res = MC_metric(self.estimated, self.test, RAPAE);

        self.assertDictEqual(PAE_res, self.MC_PAE_res);
        self.assertDictEqual(APAE_res, self.MC_APAE_res);
        self.assertDictEqual(RPAE_res, self.MC_RPAE_res);
        self.assertDictEqual(RAPAE_res, self.MC_RAPAE_res);

        return;

    def test_under_over_estimation(self) -> None:

        """
        Test the under_over_estimation function.
        """

        # Exceptions
        self.assertRaises(ValueError, MC_metric, [1, 2, 3], [1, 2], PAE);
    
        # Functionality
        PAE_res_under, PAE_res_over = under_over_estimation(self.estimated, self.test, PAE);
        APAE_res_under, APAE_res_over = under_over_estimation(self.estimated, self.test, APAE);
        RPAE_res_under, RPAE_res_over = under_over_estimation(self.estimated, self.test, RPAE);
        RAPAE_res_under, RAPAE_res_over = under_over_estimation(self.estimated, self.test, RAPAE);

        self.assertDictEqual(PAE_res_under, self.under_PAE_res);
        self.assertDictEqual(PAE_res_over, self.over_PAE_res);
        self.assertDictEqual(APAE_res_under, self.under_APAE_res);
        self.assertDictEqual(APAE_res_over, self.over_APAE_res);
        self.assertDictEqual(RPAE_res_under, self.under_RPAE_res);
        self.assertDictEqual(RPAE_res_over, self.over_RPAE_res);
        self.assertDictEqual(RAPAE_res_under, self.under_RAPAE_res);
        self.assertDictEqual(RAPAE_res_over, self.over_RAPAE_res);

        anomaly_under, anomaly_over = under_over_estimation([1, 1, 1], [1, 1, 1], PAE);
        self.assertDictEqual(anomaly_under, {});
        self.assertDictEqual(anomaly_over, {});

        return;

if __name__ == "__main__":

    unittest.main();