import pandas as pd
import timecave as tcv
from experiment_utils import (
    get_csv_filenames,
    get_univariate_series,
    split_series,
    get_freq,
)
from timecave.validation_methods.OOS import (
    Holdout,
    Repeated_Holdout,
    Rolling_Origin_Update,
    Rolling_Origin_Recalibration,
    Fixed_Size_Rolling_Window,
)
from timecave.validation_methods.prequential import Growing_Window, Rolling_Window
from timecave.validation_methods.CV import Block_CV, hv_Block_CV
from models import predict_models, predict_ARMA, predict_lstm, predict_tree
import os


def run(filenames: list[str]):
    freqs = []
    table_A = pd.DataFrame(
        columns=[
            "filename",
            "column_index",
            "method",
            "iteration",
            "model",
            "mse",
            "mae",
            "rmse",
        ]
    )
    for file in filenames:
        df = pd.read_csv(file, parse_dates=[0])
        first_col = df.columns[0]
        freq = get_freq(df, first_col)
        freqs.append(freq)
        ts_list = get_univariate_series(df)

        for idx, ts in enumerate(ts_list):
            train_val, test = split_series(ts, test_set_proportion=0.2)

            # Table A - Validation Methods
            holdout = Holdout(train_val, freq, validation_size=0.7)

            for it, (t_idx, v_idx) in enumerate(holdout.split()):
                predict_models(
                    ts[t_idx],
                    ts[v_idx],
                    file[len(os.getcwd()) :],
                    idx,
                    table_A,
                    "Holdout",
                    it,
                )
                print()

            print()
    print()


if __name__ == "__main__":
    real_data_filenames = get_csv_filenames("experiments\\datasets\\processed_data")
    syn_data_filenames = get_csv_filenames("experiments\\datasets\\synthetic_data")

    run(real_data_filenames + syn_data_filenames)
    print("!!")
