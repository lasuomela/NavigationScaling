#!/bin/bash -l
#SBATCH --job-name=EarthRovers_viz     # Job name
#SBATCH --output=logs/log_viz.out      # Name of stdout output file
#SBATCH --error=logs/log_viz.err       # Name of stderr error file
#SBATCH --partition=debug               # partition name
#SBATCH --nodes=1                       # Total number of nodes 
#SBATCH --ntasks-per-node=1             # MPI ranks per node
#SBATCH --cpus-per-task=7               # CPU cores per task
#SBATCH --mem=128G                      # Total memory for job
#SBATCH --time=0-00:30:00               # Run time (d-hh:mm:ss)

# Load the required modules
module purge
module load LUMI
module use  /appl/local/containers/ai-modules
module load singularity-AI-bindings

source ~/.bashrc
INPUT_PATH=/scratch/$SLURM_JOB_ACCOUNT/Datasets/frodobots8k
OUTPUT_PATH=/scratch/$SLURM_JOB_ACCOUNT/NavigationScaling/rerun
ENV_DIR=/projappl/$SLURM_JOB_ACCOUNT/earthrovers

export HF_HOME=/scratch/SLURM_JOB_ACCOUNT/earthrovers_hf_cache
export TORCH_HOME=/scratch/SLURM_JOB_ACCOUNT/torch_cache
export SINGULARITYENV_PREPEND_PATH=/user-software/bin # gives access to packages inside the container

# Run the visualization script for a single ride
srun singularity exec \
   -B $ENV_DIR/myenv.sqsh:/user-software:image-src=/ $ENV_DIR/earthrovers.sif \
    python -m earthrovers.data_wrangling.process_raw_dataset \
    dataset_path=$INPUT_PATH \
    rerun_output_path=$OUTPUT_PATH \
    ride_id=ride_80294_3908a7_20240817072357 \
    rerun_export=True \
    pytorch_export=False \
    num_workers=0
