#!/bin/bash
# Job name:
#SBATCH --job-name=gamescom_app
#SBATCH --account=mitarbeiter
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
# #SBATCH --output=res_objaverse_xl_rendering.txt
#SBATCH --output=./logs/gamescom.out # stdout file
#SBATCH --error=./logs/gamescom.err  # stderr file

./run_app.sh

wait
