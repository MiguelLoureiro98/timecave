from run_experiments import run
import os

os.chdir("experiments")
backup_dir = "results/s2/backups"
results_dir = "results/s2"
s2 = ["datasets/synthetic_data/s2.csv"]


run(
    s2,
    backup_dir,
    results_dir,
    resume_run=False,
    add_name="s2_914_1000_LSTM",
    models=["LSTM"],
    save_stats=False,
    from_ts=914,
    to_ts=1001,
)
