#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --time=72:00:00

module unload nvidia/cuda/10.0
module load nvidia/cuda/10.2

cd $SLURM_SUBMIT_DIR

## Get host
HOST=$(scontrol show hostname $SLURM_NODELIST | head -n1)

## Multiple nodes - slurm
srun python main.py --host $HOST --port 29500

