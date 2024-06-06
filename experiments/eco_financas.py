from run_experiments import run
import os

os.chdir("experiments")
backup_dir = "results/eco_financas/backups"
results_dir = "results/eco_financas"
eco_financas = [
    "datasets/processed_data/euro-daily-hist_1999_2024_14052024.csv",
    "datasets/processed_data/kalimati_tarkari_dataset_12052024.csv",
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
