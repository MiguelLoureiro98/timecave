from run_experiments import run
import os
import subprocess

os.chdir("experiments")
backup_dir = "results/transportes/backups"
results_dir = "results/transportes"
transportes = [
    "datasets/processed_data/metro_nyc_17052024.csv",
    "datasets/processed_data/taxi_data_15052024.csv",
    "datasets/processed_data/traffic_17052024.csv",
]
cmd_0 = """
git checkout transportes
git fetch origin

# Check if the branch is up to date with main
if ! git diff --quiet origin/main; then
  echo "Branch is not up to date with main. Merging main into transportes."
  git merge origin/main
git config --global user.email "beatriz.plourenco99@gmail.com"
git config --global user.name "Beatriz - Colab"
git add .
git commit -m "Merge"
git push origin transportes
fi"""

subprocess.run(cmd_0, shell=True, check=True, text=True, capture_output=True)

cmd = """
git config --global user.email "beatriz.plourenco99@gmail.com"
git config --global user.name "Beatriz - Colab"
git add experiments/results/transportes
git commit -m "transportes results from hpc."
git push origin transportes
"""

run(
    transportes,
    backup_dir,
    results_dir,
    resume_run=False,
    add_name="transportes_LSTM",
    models=["LSTM"],
    save_stats=False,
    cmd=cmd
)
