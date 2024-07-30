from run_experiments import run
import os

os.chdir("experiments")

"""backup_dir = "results/eco_financas/backups"
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
    add_name="eco_financas_LSTM",
    models=["LSTM"],
    save_stats=False,
)

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
    add_name="energia_LSTM",
    models=["LSTM"],
    save_stats=False,
)"""

backup_dir = "results/eng_ciencias/backups"
results_dir = "results/eng_ciencias"
eng_ciencias = [
#    "datasets/processed_data/mechanical_gear_vibration_data.csv",
    "datasets/processed_data/room_occupancy_data.csv",
    "datasets/processed_data/torque_characteristics_15052024.csv",
]

run(
    eng_ciencias,
    backup_dir,
    results_dir,
    resume_run=False,
    add_name="eng_ciencias_LSTM_room_torque",
    models=["LSTM"],
    save_stats=False
)
