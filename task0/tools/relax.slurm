#!/bin/sh
#SBATCH -D /s/ls4/users/volodin_mv/task0
#SBATCH -J relax
#SBATCH -o output_%A_%a.out
#SBATCH -e output_%A_%a.err
#SBATCH -t 01:00:00
#SBATCH -p hpc4-el7-3d
#SBATCH --array=0-2

# Starts three parallel tasks (u = 0.5, 1.0, 1.5).
# relax.py itself writes .out / .npz / .skip to ../data/ regarding its location.

module load anaconda3/default
# shellcheck disable=SC3030
US=(0.5 1.0 1.5)
# shellcheck disable=SC3054
U=${US[$SLURM_ARRAY_TASK_ID]}

/s/ls4/users/volodin_mv/env_opt/bin/python -u relax.py --u "$U"