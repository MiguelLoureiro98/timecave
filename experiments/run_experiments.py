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


def run(filenames: list[str], methods: list[str]):
    freqs = []
    for file in filenames:
        df = pd.read_csv(file)
        first_col = df.columns[0]
        freq = get_freq(df, first_col)
        freqs.append(freq)
        ts_list = get_univariate_series(df)

        # for ts in ts_list:
        #    train, test = split_series(ts, test_set_proportion=0.2)

        # Table A - Validation Methods

    print()


if __name__ == "__main__":
    real_data_filenames = get_csv_filenames("experiments\\datasets\\processed_data")
    syn_data_filenames = get_csv_filenames("experiments\\datasets\\synthetic_data")
    methods = []
    syn_data_filenames = []

    run(real_data_filenames + syn_data_filenames, methods)
    print("!!")
