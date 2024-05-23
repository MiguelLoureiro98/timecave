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

def error_estimation(processed_data: pd.DataFrame) -> pd.DataFrame:
    
    mse_est = processed_data.groupby(["series", "method", "model"]).apply(weighted_average, "mse", "weights").reset_index().rename(columns={0: "mse"});
    mae_est = processed_data.groupby(["series", "method", "model"]).apply(weighted_average, "mae", "weights").reset_index().rename(columns={0: "mae"});

    error_estimates = pd.merge(left=mse_est, right=mae_est, on=["series", "method", "model"]);

    return error_estimates;

# Both the estimation data and the test data are assumed to be processed prior to this.

def merge_estimates_true(estimation_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame: # can be used to merge data for the iterations experiment

    aggregate_data = pd.merge(left=estimation_data, right=test_data, on=["series", "model"], suffixes=["_estimate", "_true"]);

    return aggregate_data;

def compute_val_metrics(aggregate_data: pd.DataFrame, performance_metric: str = "mse") -> pd.DataFrame: # this can be used with S1, S2 and S3 individually

    models = aggregate_data["model"].unique().tolist();
    methods = aggregate_data["method"].unique().tolist();

    frames = [];
    val_metrics = [metrics.PAE, metrics.APAE, metrics.RPAE, metrics.RAPAE];

    for method in methods:

        for model in models:

            filters = (aggregate_data["model"] == model) & (aggregate_data["method"] == method);
            data = aggregate_data.loc[filters].copy();

            res_dict = [metrics.MC_metric(data["{}_estimate".format(performance_metric)].to_list(), data["{}_true".format(performance_metric)], val_met) for val_met in val_metrics];
            res_df = [pd.DataFrame(res) for res in res_dict];
            
            for df in res_df:

                df["method"] = method;
                df["model"] = model;
    
            final_df = pd.concat(res_df, axis=0);
            frames.append(final_df);
    
    results = pd.concat(frames, axis=0);

    return results;

def under_over_analysis(aggregate_data: pd.DataFrame, performance_metric: str = "mse") -> tuple[pd.DataFrame]:

    models = aggregate_data["model"].unique().tolist();
    methods = aggregate_data["method"].unique().tolist();

    under_frames = [];
    over_frames = [];
    val_metrics = [metrics.PAE, metrics.APAE, metrics.RPAE, metrics.RAPAE];

    for method in methods:

        for model in models:

            filters = (aggregate_data["model"] == model) & (aggregate_data["method"] == method);
            data = aggregate_data.loc[filters].copy();

            res_dict = [metrics.under_over_estimation(data["{}_estimate".format(performance_metric)].to_list(), data["{}_true".format(performance_metric)], val_met) for val_met in val_metrics];
            res_under = [pd.DataFrame(res[0]) for res in res_dict];
            res_over = [pd.DataFrame(res[1]) for res in res_dict];
            
            for (under_df, over_df) in zip(res_under, res_over):

                under_df["method"] = method;
                under_df["model"] = model;
                over_df["method"] = method;
                over_df["model"] = model;
    
            final_under = pd.concat(res_under, axis=0);
            final_over = pd.concat(res_over, axis=0);
            under_frames.append(final_under);
            over_frames.append(final_over);
    
    results_under = pd.concat(under_frames, axis=0);
    results_over = pd.concat(over_frames, axis=0);

    return (results_under, results_over);

def under_over_by_method(aggregate_data: pd.DataFrame, performance_metric: str = "mse") -> pd.DataFrame:

    methods = aggregate_data["method"].unique().tolist();

    under_frames = [];
    over_frames = [];
    val_metrics = [metrics.PAE, metrics.APAE, metrics.RPAE, metrics.RAPAE];

    for method in methods:

        filters = (aggregate_data["method"] == method);
        data = aggregate_data.loc[filters].copy();

        res_dict = [metrics.under_over_estimation(data["{}_estimate".format(performance_metric)].to_list(), data["{}_true".format(performance_metric)], val_met) for val_met in val_metrics];
        res_under = [pd.DataFrame(res[0]) for res in res_dict];
        res_over = [pd.DataFrame(res[1]) for res in res_dict];
            
        for (under_df, over_df) in zip(res_under, res_over):

            under_df["method"] = method;
            over_df["method"] = method;
    
        final_under = pd.concat(res_under, axis=0);
        final_over = pd.concat(res_over, axis=0);
        under_frames.append(final_under);
        over_frames.append(final_over);
    
    results_under = pd.concat(under_frames, axis=0);
    results_over = pd.concat(over_frames, axis=0);

    return (results_under, results_over);

def PAE_row(row, metric: str):

    return metrics.PAE(row[f"{metric}_estimated"], row[f"{metric}_true"]);

def APAE_row(row, metric: str):

    return metrics.APAE(row[f"{metric}_estimated"], row[f"{metric}_true"]);

def RPAE_row(row, metric: str):

    return metrics.RPAE(row[f"{metric}_estimated"], row[f"{metric}_true"]);

def RAPAE_row(row, metric: str):

    return metrics.RAPAE(row[f"{metric}_estimated"], row[f"{metric}_true"]);

def compute_metrics_per_row(aggregate_data: pd.DataFrame, performance_metric: str) -> pd.DataFrame:

    aggregate_data["PAE"] = aggregate_data.apply(PAE_row, args=[performance_metric], axis=1);
    aggregate_data["APAE"] = aggregate_data.apply(APAE_row, args=[performance_metric], axis=1);
    aggregate_data["RPAE"] = aggregate_data.apply(RPAE_row, args=[performance_metric], axis=1);
    aggregate_data["RAPAE"] = aggregate_data.apply(RAPAE_row, args=[performance_metric], axis=1);

    return aggregate_data;

def val_metrics_per_iteration(aggregate_data: pd.DataFrame, performance_metric: str) -> pd.DataFrame:

    methods_list = ["Growing_Window",
                    "Rolling_Window",
                    "Weighted_Growing_Window",
                    "Weighted_Rolling_Window",
                    "Gap_Growing_Window",
                    "Gap_Rolling_Window",
                    "Block_CV",
                    "Weighted_Block_CV",
                    "hv_Block_CV"];
    
    filters = (aggregate_data["method"].isin(methods_list));
    preq_CV_data = aggregate_data.loc[filters].copy();

    val_metrics = compute_metrics_per_row(preq_CV_data, performance_metric);
    

    return;

def boxplots(aggregate_data: pd.DataFrame, performance_metric: str) -> None:

    val_metrics = compute_metrics_per_row(aggregate_data, performance_metric);

    pass

if __name__ == "__main__":

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

    df = pd.DataFrame({"method": ["CV", "CV", "Preq", "OOS", "Preq", "CV", "OOS", "OOS", "Preq", "CV"], 
                       "model": ["ARIMA", "LSTM", "ARIMA", "ARIMA", "LSTM", "DT", "LSTM", "DT", "DT", "ARIMA"],
                       "series": [i for i in range(1, 11)],
                       "mse": [i**2 for i in range(1, 11)],
                       "mae": [i*2 for i in range(1, 11)],
                       "weights": [i for i in range(1, 11)]});

    #print(df);

    mse_est = df.groupby(["method", "model"]).apply(weighted_average, "mse", "weights", include_groups=False).reset_index().rename(columns={0: "mse"});
    #mse_mean = df.groupby(["method", "model"])["mse"].mean().reset_index();
    mae_est = df.groupby(["method", "model"]).apply(weighted_average, "mae", "weights", include_groups=False).reset_index().rename(columns={0: "mae"});

    #print(type(mse_est.columns[2]));
    #print(mae_est.columns);

    new_df = pd.merge(left=mse_est, right=mae_est, on=["method", "model"]);

    #print(new_df);