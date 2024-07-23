#!/bin/bash

#SBATCH --job-name=Try
#SBATCH --time=0:01:00
#SBATCH --partition=hpc
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

git checkout Colab_Beatriz
git fetch origin

# Check if the branch is up to date with main
if ! git diff --quiet origin/main; then
  echo "Branch is not up to date with main. Merging main into Colab_Beatriz."
  git merge origin/main
fi
python experiments/test_hpc.py
git config --global user.email "beatriz.plourenco99@gmail.com"
git config --global user.name "Beatriz - Colab"
git add experiments/results/tests
git commit -m "test results from hpc 2."
git push origin Colab_Beatriz
exit
