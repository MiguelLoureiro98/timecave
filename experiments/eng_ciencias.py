from run_experiments import run
import os

os.chdir("experiments")
backup_dir = "results/eng_ciencias/backups"
results_dir = "results/eng_ciencias"
eng_ciencias = [
    "datasets/processed_data/mechanical_gear_vibration_data.csv",
    "datasets/processed_data/room_occupancy_data.csv",
    "datasets/processed_data/torque_characteristics_15052024.csv",
]

run(
    eng_ciencias,
    backup_dir,
    results_dir,
    resume_run=False,
    add_name="eng_ciencias",
    models=["ARMA", "tree"],
    save_stats=False,
)
