#!/bin/bash

#SBATCH --job-name=ambiente
#SBATCH --time=00:29:55
#SBATCH --partition=medusa
#SBATCH --error=err.job.%j
#SBATCH --output=out.job.%j
#SBATCH -D .
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python/3.10.13
module load cuda/12.1
module load tensorflow/2.14-0

git checkout ambiente
git fetch origin

# Check if the branch is up to date with main
if ! git diff --quiet origin/main; then
  echo "Branch is not up to date with main. Merging main into ambiente."
  git merge origin/main
fi
python experiments/ambiente.py
git config --global user.email "beatriz.plourenco99@gmail.com"
git config --global user.name "Beatriz - Colab"
git add experiments/results/ambiente
git commit -m "ambiente results from hpc."
git push origin ambiente

exit
