from run_experiments import run
import os

os.chdir("experiments")
backup_dir = "results/eco_financas/backups"
results_dir = "results/eco_financas"
eco_financas = [
    "datasets/processed_data/covid19_17052024.csv",
    "datasets/processed_data/ecg_alcohol_17052024.csv",
    "datasets/processed_data/non_invasive_fetal_ecg_15052024.csv",
    "datasets/processed_data/pharma_sales_17052024.csv",
]

run(
    eco_financas,
    backup_dir,
    results_dir,
    resume_run=False,
    add_name="eco_financas",
    models=["ARMA", "tree"],
    save_stats=False,
)
