#!/bin/bash

#SBATCH --job-name=transportes
#SBATCH --time=02:01:00
#SBATCH --partition=hpc
#SBATCH --error=err.job.%j
#SBATCH --output=out.job.%j
#SBATCH -D .
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python/3.10.13
module load gcc-13.2

python experiments/transportes.py

exit
