from run_experiments import run
import os

os.chdir("experiments")
backup_dir = "results/ambiente/backups"
results_dir = "results/ambiente"
ambiente = [
    "datasets/processed_data/aire_discharge_environment_data.csv",
    "datasets/processed_data/air_quality_Rajamahendravaram_13052024.csv",
    "datasets/processed_data/forest_fires_Rio_Janeiro.csv",
]

run(
    ambiente,
    backup_dir,
    results_dir,
    resume_run=False,
    add_name="ambiente_LSTM",
    models=["LSTM"],
    save_stats=False,
)
