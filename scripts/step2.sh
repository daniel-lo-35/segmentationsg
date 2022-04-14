#!/bin/bash
#SBATCH --job-name="step2"
#SBATCH --partition=v100-32g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00
#SBATCH --chdir=.
#SBATCH --output=step2out.txt
#SBATCH --error=step2err.txt
###SBATCH --test-only

sbatch_pre.sh

module load python/tensorflow-2-gpu

python3 step2.py

sbatch_post.sh
