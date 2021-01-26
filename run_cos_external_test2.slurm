#!/bin/bash
#Assume will be finished in no more than 18 hours
#SBATCH -t 1:00:00
#Launch on 20 cores distributed over as many nodes as needed
#SBATCH -n 20
#Assume need 6 GB/core (6144 MB/core)
#SBATCH --mem-per-core=6144

unset PYTHONHOME
. ~/.python-venvs/reservoir-rls/bin/activate

cd /lustre/awikner1/Reservoir-GN-RL

python reservoir_test_cosine_external_test2.py
