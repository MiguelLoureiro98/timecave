from run_experiments import run
import os

os.chdir("experiments")
backup_dir = "results/s1/backups"
results_dir = "results/s1"
s1 = ["datasets/synthetic_data/s1.csv"]


run(
    s1,
    backup_dir,
    results_dir,
    resume_run=False,
    add_name="s1_997_1000_LSTM",
    models=["LSTM"],
    save_stats=False,
    from_ts=997,
    to_ts=1001,
)
