#!/bin/bash

#SBATCH --job-name=test_job
#SBATCH --time=0:10:00
#SBATCH --partition=hpc
#SBATCH --error=err.job.%j
#SBATCH -D
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python/3.10.13
module load cuda/12.1
module load tensorflow/2.14-0

git pull origin main
git checkout Colab_Beatriz
git merge main
git commit -m "get from main"
git push origin Colab_Beatriz
python experiments/test_hpc.py
git add experiments/results/tests
git commit -m "test results from hpc."
git push origin Colab_Beatriz

exit
