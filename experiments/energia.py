from run_experiments import run
import os

os.chdir("experiments")
backup_dir = "results/energia/backups"
results_dir = "results/energia"
energia = [
    "datasets/processed_data/power_consumption_data.csv",
    "datasets/processed_data/US_energy_generation_data.csv",
    "datasets/processed_data/power_comsumption_india_14052024.csv",
]


run(
    energia,
    backup_dir,
    results_dir,
    resume_run=False,
    add_name="energia",
    models=["ARMA", "tree"],
    save_stats=False,
)
