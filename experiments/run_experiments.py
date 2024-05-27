import pandas as pd
import timecave as tcv
from experiment_utils import (
    get_csv_filenames,
    get_univariate_series,
    split_series,
    get_freq,
    update_stats_tables,
    initialize_tables,
    get_methods_list,
    save_tables,
)
from models import predict_models
import os


def run(filenames: list[str], backup_dir: str, results_dir: str, save_freq: int):
    table_A, table_B, stats_total, stats_train, stats_val = initialize_tables()

    nb_ts = 0
    for file in filenames:
        df = pd.read_csv(file, parse_dates=[0])
        first_col = df.columns[0]
        freq = get_freq(df, first_col)
        ts_list = get_univariate_series(df)

        for idx, ts in enumerate(ts_list):
            train_val, test = split_series(ts, test_set_proportion=0.2)

            # Validation Methods
            methods = get_methods_list(train_val, freq)
            for method in methods:
                for it, (t_idx, v_idx) in enumerate(method.split()):
                    tree, lstm, arma = predict_models(
                        train_val[t_idx],
                        train_val[v_idx],
                        file[len(os.getcwd()) :],
                        idx,
                        table_A,
                        method,
                        it,
                    )
                stats_total, stats_train, stats_val = update_stats_tables(
                    stats_total,
                    stats_train,
                    stats_val,
                    method,
                    file[len(os.getcwd()) :],
                    idx,
                    freq=freq,
                )

            # "True" results
            predict_models(train_val, test, file[len(os.getcwd()) :], idx, table_B)

            # save backups
            nb_ts = +1
            if nb_ts % save_freq == 0:
                save_tables(
                    table_A, table_B, stats_total, stats_train, stats_val, backup_dir
                )

            print()

    # Final save
    save_tables(table_A, table_B, stats_total, stats_train, stats_val, results_dir)
    print()


if __name__ == "__main__":
    backup_dir = "results/backups"
    results_dir = "results"
    real_data_filenames = get_csv_filenames("experiments\\datasets\\processed_data")
    syn_data_filenames = get_csv_filenames("experiments\\datasets\\synthetic_data")

    run(real_data_filenames + syn_data_filenames, backup_dir, results_dir, save_freq=1)
    print("!!")
