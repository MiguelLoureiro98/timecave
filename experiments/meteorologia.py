from run_experiments import run
import os

os.chdir("experiments")
backup_dir = "results/meteorologia/backups"
results_dir = "results/meteorologia"
meteorologia = [
    "datasets/processed_data/covid19_17052024.csv",
    "datasets/processed_data/ecg_alcohol_17052024.csv",
    "datasets/processed_data/non_invasive_fetal_ecg_15052024.csv",
    "datasets/processed_data/pharma_sales_17052024.csv",
]

run(
    meteorologia,
    backup_dir,
    results_dir,
    resume_run=False,
    add_name="meteorologia",
    models=["ARMA", "tree"],
    save_stats=False,
)
