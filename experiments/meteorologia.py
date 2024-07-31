from run_experiments import run
import os

os.chdir("experiments")
backup_dir = "results/meteorologia/backups"
results_dir = "results/meteorologia"
meteorologia = [
    "datasets/processed_data/aire_discharge_weather_data.csv",
    "datasets/processed_data/DailyDelhiClimate_12052024.csv",
    "datasets/processed_data/jena_climate_data.csv",
    "datasets/processed_data/MLTempDataset_13052024.csv",
]


run(
    meteorologia,
    backup_dir,
    results_dir,
    resume_run=True,
    add_name="meteorologia_LSTM",
    models=["LSTM"],
    save_stats=False,
    resume_files=['results/meteorologia/backups/table_A_meteorologia_LSTM_2024_07_30__20_10_04.csv', 
                  'results/meteorologia/backups/table_B_meteorologia_LSTM_2024_07_30__20_10_04.csv'] + [None, None, None]
)
