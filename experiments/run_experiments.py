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
    read_tables,
    get_last_iteration,
    get_latest_files,
)
from models import predict_models
import os
from timecave.validation_methods.OOS import Rolling_Origin_Update
import time


def run(
    filenames: list[str],
    backup_dir: str,
    results_dir: str,
    save_freq: int,
    resume_run: bool = False,
    from_ts: range = None,
    until_ts: range = None,
):
    if not resume_run:
        table_A, table_B, stats_total, stats_train, stats_val = initialize_tables()
        col_id1 = -1

    else:
        ta_dir = get_latest_files(backup_dir, "table_A_")
        tb_dir = get_latest_files(backup_dir, "table_B_")
        s1_dir = get_latest_files(backup_dir, "stats_total_")
        s2_dir = get_latest_files(backup_dir, "stats_train_")
        s3_dir = get_latest_files(backup_dir, "stats_val_")
        print(f"Directories found: \n{ta_dir}, {tb_dir}, {s1_dir}, {s2_dir}, {s3_dir}")
        table_A, table_B, stats_total, stats_train, stats_val = read_tables(
            ta_dir,
            tb_dir,
            s1_dir,
            s2_dir,
            s3_dir,
        )
        lastit = get_last_iteration(table_A)
        file1, col_id1 = lastit["filename"], lastit["column_index"]
        filenames = filenames[filenames.index(file1) :]

    nb_ts = 0
    for file in filenames:
        df = pd.read_csv(file, parse_dates=[0])
        if from_ts is not None:
            df = df.iloc[:, [0] + list(range(from_ts, df.shape[1]))]
        if until_ts is not None:
            df = df.iloc[:, list(range(0, until_ts))]
        first_col = df.columns[0]
        freq = get_freq(df, first_col)
        ts_list = get_univariate_series(df)
        ts_list = ts_list[col_id1 + 1 :][:1]

        for idx, ts in enumerate(ts_list):
            train_val, test = split_series(ts, test_set_proportion=0.2)

            # Validation Methods
            methods = get_methods_list(train_val, freq)[:1]  # TODO

            for method in methods:
                print(f"Method: {method}")
                for it, (t_idx, v_idx) in enumerate(method.split()):
                    print(f"Iteration: {it}")

                    # --------------------- TIME -------------------- #
                    start_time = time.time()
                    # --------------------- TIME -------------------- #
                    predict_models(
                        train_val[t_idx],
                        train_val[v_idx],
                        file,
                        idx,
                        table_A,
                        method,
                        it,
                    )
                    # --------------------- TIME -------------------- #
                    end_time = time.time()
                    runtime_seconds = end_time - start_time

                    minutes = int(runtime_seconds // 60)
                    seconds = int(runtime_seconds % 60)
                    # --------------------- TIME -------------------- #

                    print(f"Runtime: {minutes} minutes {seconds} seconds")

                    if isinstance(method, Rolling_Origin_Update):
                        print()

                stats_total, stats_train, stats_val = update_stats_tables(
                    stats_total,
                    stats_train,
                    stats_val,
                    method,
                    file,
                    idx,
                    freq=freq,
                )

            # "True" results
            predict_models(train_val, test, file, idx, table_B)

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

    os.chdir("experiments")

    backup_dir = "results/backups"
    results_dir = "results"
    real_data_filenames = get_csv_filenames("datasets\\processed_data")
    syn_data_filenames = get_csv_filenames("datasets\\synthetic_data")

    transportes = [
        "datasets\\processed_data\\metro_nyc_17052024.csv",
        "datasets\\processed_data\\taxi_data_15052024.csv",
        "datasets\\processed_data\\traffic_17052024.csv",
    ]
    saude = [
        "datasets\\processed_data\\covid19_17052024.csv",
        "datasets\\processed_data\\ecg_alcohol_17052024.csv",
        "datasets\\processed_data\\non_invasive_fetal_ecg_15052024.csv",
        "datasets\\processed_data\\pharma_sales_17052024.csv",
    ]
    ambiente = [
        "datasets\\processed_data\\aire_discharge_environment_data.csv",
        "datasets\\processed_data\\air_quality_Rajamahendravaram_13052024.csv",
        "datasets\\processed_data\\forest_fires_Rio_Janeiro.csv",
    ]
    eng_ciencias = [
        "datasets\\processed_data\\mechanical_gear_vibration_data.csv",
        "datasets\\processed_data\\room_occupancy_data.csv",
        "datasets\\processed_data\\torque_characteristics_15052024.csv",
    ]
    meteorologia = [
        "datasets\\processed_data\\aire_discharge_weather_data.csv",
        "datasets\\processed_data\\DailyDelhiClimate_12052024.csv",
        "datasets\\processed_data\\jena_climate_data.csv",
        "datasets\\processed_data\\MLTempDataset_13052024.csv",
    ]
    eco_financas = [
        "datasets\\processed_data\\euro-daily-hist_1999_2024_14052024.csv",
        "datasets\\processed_data\\kalimati_tarkari_dataset_12052024.csv",
    ]
    energia = [
        "datasets\\processed_data\\power_consumption_data.csv",
        "datasets\\processed_data\\US_energy_generation_data.csv",
        "datasets\\processed_data\\power_comsumption_india_14052024.csv",
    ]

    files = transportes

    run(files, backup_dir, results_dir, save_freq=1, resume_run=False)
    print("!!")
