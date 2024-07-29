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

commands = [
    "git checkout transportes",
    "git fetch origin",
    "if ! git diff --quiet origin/main; then git merge origin/main fi",
    'if ! git diff --quiet origin/main; then git config --global user.email "beatriz.plourenco99@gmail.com" fi',
    'if ! git diff --quiet origin/main; then git config --global user.name "Beatriz - Colab" fi',
    "if ! git diff --quiet origin/main; then git add .  fi",
    'if ! git diff --quiet origin/main; then git commit -m "Merge"  fi',
    "if ! git diff --quiet origin/main; then git push origin transportes  fi",
]

full_command = " && ".join(commands)


result = subprocess.run(full_command, shell=True, check=True, text=True, capture_output=True)

print("Return code:", result.returncode)
print("Standard Output:", result.stdout)
print("Standard Error:", result.stderr)


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
