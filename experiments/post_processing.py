"""
Post-processing module.
"""

import numpy as np
import pandas as pd
import timecave.validation_strategy_metrics as metrics

def series_name(data: pd.DataFrame) -> pd.DataFrame:

    # ! Check the column names!!!

    new_data = data.copy();
    new_data["series"] = new_data["file_name"] + "_" + new_data["column_name"];
    new_data = new_data.drop(columns=["file_name", "column_name"]);

    return new_data;

def weighted_average(data: pd.DataFrame, values: str, weights: str):
    
    return (data[values] * data[weights]).sum() / data[weights].sum();

def error_estimation(processed_data: pd.DataFrame, methods: str | None=None) -> pd.DataFrame:

    if(methods is None):

        methods = ["Repeated_Holdout", 
                   "Rolling_Origin_Update",
                   "Rolling_Origin_Recalibration",
                   "Fixed_Size_Rolling_Origin",
                   "Growing_Window",
                   "Rolling_Window",
                   "Weighted_Growing_Window",
                   "Weighted_Rolling_Window",
                   "Gap_Growing_Window",
                   "Gap_Rolling_Window",
                   "Block_CV",
                   "Weighted_Block_CV",
                   "hv_Block_CV"];
    
    mse_est = processed_data.groupby([["series", "method", "model"]]).apply(weighted_average, processed_data["mse"], processed_data["weights"]);
    mae_est = processed_data.groupby([["series", "method", "model"]]).apply(weighted_average, processed_data["mae"], processed_data["weights"]);

    error_estimates = pd.merge(left=mse_est, right=mae_est, on=[["series", "method", "model"]]);

    return error_estimates;

def error_test(processed_test_data: pd.DataFrame) -> pd.DataFrame:

    pass

# Both the estimation data and the test data are assumed to be processed prior to this.
def compute_val_metrics(estimation_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:

    

    pass

if __name__ == "__main__":

    pass