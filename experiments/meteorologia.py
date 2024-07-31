from run_experiments import run
import os

os.chdir("experiments")
backup_dir = "results/meteorologia/backups"
results_dir = "results/meteorologia"
meteorologia = [
    "datasets/processed_data/MLTempDataset_13052024.csv"
]


run(
    meteorologia,
    backup_dir,
    results_dir,
    resume_run=False,
    add_name="meteorologia_LSTM_mltemp",
    models=["LSTM"],
    save_stats=False
)
meteorologia = [
    "datasets/processed_data/jena_climate_data.csv",
]
run(
    meteorologia,
    backup_dir,
    results_dir,
    resume_run=False,
    from_ts=2,
    add_name="meteorologia_LSTM_jena",
    models=["LSTM"],
    save_stats=False
)
