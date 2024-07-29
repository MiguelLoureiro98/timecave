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
    "if ! git diff --quiet origin/main; then git merge origin/main; ",
    '  echo "Branch is not up to date with main. Merging main into transportes."',
    "  git merge origin/main",
    '  git config --global user.email "beatriz.plourenco99@gmail.com"',
    '  git config --global user.name "Beatriz - Colab"',
    "  git add .",
    '  git commit -m "Merge"',
    "  git push origin transportes",
    "fi"
]

for command in commands:
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
    print(f"Return code: {result.returncode}")
    print(f"Standard Output: {result.stdout}")
    print(f"Standard Error: {result.stderr}")
    if result.returncode != 0:
        print(f"Command failed: {command}")

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
