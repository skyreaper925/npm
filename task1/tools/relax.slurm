#!/bin/sh
#SBATCH -D /s/ls4/users/volodin_mv/task1
#SBATCH -J relax1
#SBATCH -o output_%j.out
#SBATCH -e output_%j.err
#SBATCH -t 06:00:00
#SBATCH -p hpc4-el7-3d

# Релаксация смеси: холодный газ (T=1) + горячий газ (T=2) -> равновесие.
# Один таск (в отличие от task0, где был array по трём значениям u).
# relax.py пишет mixture.out / mixture.skip / f_mixture.npz в текущую папку
# (~/task1), откуда их забирает tools/cluster.py fetch.

module load anaconda3/default
/s/ls4/users/volodin_mv/env_opt/bin/python -u relax.py --out-dir . --no-progress
