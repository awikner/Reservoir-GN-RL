#!/homes/awikner1/.python-venvs/reservoir-rls/bin/python -u
#Assume will be finished in no more than 18 hours
#SBATCH -t 0:02:00
#Launch on 10 cores distributed over as many nodes as needed
#SBATCH --ntasks=5
#SBATCH -N 1
#Assume need 6 GB/core (6144 MB/core)
#SBATCH --mail-user=awikner1@umd.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

import multiprocessing

print(multiprocessing.cpu_count())
