#!/bin/bash
#Assume will be finished in no more than 18 hours
#SBATCH -t 18:00:00
#Launch on 41 cores distributed over as many nodes as needed
#SBATCH -n 41
#Assume need 6 GB/core (6144 MB/core)
#SBATCH --mem-per-core=6144
#SBATCH --mail-user=awikner1@umd.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END


unset PYTHONHOME
. ~/.python-venvs/reservoir-rls/bin/activate

cd /lustre/awikner1/Reservoir-GN-RL

python reservoir_test_cosine_external.py
