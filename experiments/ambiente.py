from run_experiments import run
import os

os.chdir("experiments")
backup_dir = "results/ambiente/backups"
results_dir = "results/ambiente"
ambiente = [
    "datasets/processed_data/covid19_17052024.csv",
    "datasets/processed_data/ecg_alcohol_17052024.csv",
    "datasets/processed_data/non_invasive_fetal_ecg_15052024.csv",
    "datasets/processed_data/pharma_sales_17052024.csv",
]

run(
    ambiente,
    backup_dir,
    results_dir,
    resume_run=False,
    add_name="ambiente",
    models=["ARMA", "tree"],
    save_stats=False,
)
