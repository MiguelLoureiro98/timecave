from run_experiments import run
import os

os.chdir("experiments")
backup_dir = "results/meteorologia/backups"
results_dir = "results/meteorologia"
meteorologia = [
    "datasets/processed_data/MLTempDataset_13052024.csv",
    "datasets/processed_data/jena_climate_data.csv",
]


run(
    meteorologia,
    backup_dir,
    results_dir,
    resume_run=False,
    add_name="meteorologia_LSTM_jena",
    models=["LSTM"],
    save_stats=False
)
