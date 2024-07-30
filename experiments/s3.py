from run_experiments import run
import os

os.chdir("experiments")
backup_dir = "results/s3/backups"
results_dir = "results/s3"
s3 = ["datasets/synthetic_data/s3.csv"]


run(
    s3,
    backup_dir,
    results_dir,
    resume_run=False,
    add_name="s3_318_1000_LSTM",
    models=["LSTM"],
    save_stats=False,
    from_ts=318,
    to_ts=1001,
)
