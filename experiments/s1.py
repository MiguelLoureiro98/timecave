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
    add_name="s1_0_200",
    models=["ARMA", "tree"],
    save_stats=False,
    from_ts=0,
    to_ts=200,
)
