#!/bin/bash

# The budget account to use
#SBATCH --account=tra210016p

#SBATCH --reservation=ihpcsschallday3

# The partition to use
#SBATCH --partition=GPU
##SBATCH --partition=GPU-shared

# The type and number of GPUs to use per node
#SBATCH --gres=gpu:v100-32:8
##SBATCH --gres=gpu:v100-32:1

# Jobs are capped at 1 minute (Your code should run for ~10 seconds anyway)
#SBATCH --time=00:01:00

# The name of the output file
#SBATCH --output=output_gpu.txt
# The name of the error file
#SBATCH --error=error_gpu.txt

# The number of nodes (max. 4)
#SBATCH --nodes=1

# The number of MPI processes per node
#SBATCH --ntasks-per-node=1

# The number of OpenMP threads per MPI process
#SBATCH --cpus-per-task=1

# Notifications
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=apowis@pppl.gov


# The number of OpenMP threads. If using MPI, it is the number of OpenMP threads per MPI process
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Place OpenMP threads on cores
export OMP_PLACES=cores

# Keep the OpenMP threads where they are
export OMP_PROC_BIND=true

# Load the modules needed (do not change the modules)
ml purge
module load AI/pytorch_23.02-1.13.1-py3

# Execute the program
python pagerank_block.py


