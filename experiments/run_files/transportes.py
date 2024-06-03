from run_experiments import run
import os

os.chdir("experiments")
backup_dir = "results/backups"
results_dir = "results"
transportes = [
    "datasets\\processed_data\\metro_nyc_17052024.csv",
    "datasets\\processed_data\\taxi_data_15052024.csv",
    "datasets\\processed_data\\traffic_17052024.csv",
]

run(transportes, backup_dir, results_dir, resume_run=False, add_name="transportes")
