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
    get_files,
    get_last_iteration,
)
from models import predict_models
import os
from timecave.validation_methods.OOS import Rolling_Origin_Update
import time


def run(
    filenames: list[str],
    backup_dir: str,
    results_dir: str,
    save_freq: int = 1,
    resume_run: bool = False,
    resume_files: list[str] = [],
    from_ts: range = None,
    until_ts: range = None,
):
    # Get tables
    if not resume_run:
        table_A, table_B, stats_total, stats_train, stats_val = initialize_tables()
        col_id1 = -1

    else:
        table_A, table_B, stats_total, stats_train, stats_val = get_files(
            resume_files, backup_dir
        )
        lastit = get_last_iteration(table_A)
        file1, col_id1 = lastit["filename"], lastit["column_index"]
        filenames = filenames[filenames.index(file1) :]

    nb_ts = 0
    # Iterate per filename
    for file in filenames:
        df = pd.read_csv(file, parse_dates=[0])
        freq = get_freq(df, df.columns[0])
        ts_list = get_univariate_series(df)

        if from_ts is not None:
            df = df.iloc[:, [0] + list(range(from_ts, df.shape[1]))]
        if until_ts is not None:
            df = df.iloc[:, list(range(0, until_ts))]

        ts_list = ts_list[col_id1 + 1 :]  # TODO remove [:1]

        # Iterate per time series
        for idx, ts in enumerate(ts_list):
            train_val, test = split_series(ts, test_set_proportion=0.2)

            methods = get_methods_list(train_val, freq)  # TODO remove [:1]
            # Iterate per validation method
            for method in methods:
                print(f"Method: {method}")
                for it, (t_idx, v_idx) in enumerate(method.split()):
                    print(f"Method: {method}, Iteration: {it}")

                    # Modelling
                    # --------------------- TIME -------------------- #
                    start_time = time.time()
                    # --------------------- TIME -------------------- #
                    print("-" * 100)
                    print("model1", "model2", "model3")
                    print("-" * 100)
                    """predict_models(
                        train_val[t_idx],
                        train_val[v_idx],
                        file,
                        idx,
                        table_A,
                        method,
                        it,
                    )"""
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
            """predict_models(train_val, test, file, idx, table_B)"""
            print("-" * 100)
            print("True results", "model1", "model2", "model3")
            print("-" * 100)

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

    resume_files = [
        "results\\backups\\table_A__2024_05_29__10_31_23.csv",
        "results\\backups\\table_B__2024_05_29__10_31_23.csv",
        "results\\backups\\stats_total__2024_05_29__10_31_23.csv",
        "results\\backups\\stats_train__2024_05_29__10_31_23.csv",
        "results\\backups\\stats_val__2024_05_29__10_31_23.csv",
    ]

    run(files, backup_dir, results_dir, resume_run=False)

    """run(
        files,
        backup_dir,
        results_dir,
        resume_run=True,
        resume_files=resume_files,
    )

    run(
        syn_data_filenames,
        backup_dir,
        results_dir,
        resume_run=False,
        from_ts=0,
        until_ts=50,
    )"""
    print("!!")
