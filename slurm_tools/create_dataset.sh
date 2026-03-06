#!/bin/bash -l
#SBATCH --job-name=EarthRovers_wrangle     # Job name
#SBATCH --output=logs/log_wrangle.out      # Name of stdout output file
#SBATCH --error=logs/log_wrangle.err       # Name of stderr error file
#SBATCH --partition=small               # partition name
#SBATCH --nodes=1                       # Total number of nodes 
#SBATCH --ntasks-per-node=1             # MPI ranks per node
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G                      # Total memory for job
#SBATCH --time=1-00:00:00               # Run time (d-hh:mm:ss)

module purge
module load LUMI
module use  /appl/local/containers/ai-modules
module load singularity-AI-bindings

source ~/.bashrc
DATASET_PATH=/scratch/$SLURM_JOB_ACCOUNT/Datasets/frodobots8k
OUTPUT_PATH=/flash/$SLURM_JOB_ACCOUNT/FrodoBots8k
ENV_DIR=/projappl/$SLURM_JOB_ACCOUNT/earthrovers

export HF_HOME=/scratch/$SLURM_JOB_ACCOUNT/earthrovers_hf_cache
export TORCH_HOME=/scratch/$SLURM_JOB_ACCOUNT/torch_cache
export SINGULARITYENV_PREPEND_PATH=/user-software/bin # gives access to packages inside the container

# Run the dataset creation script
srun singularity exec \
   -B $ENV_DIR/myenv.sqsh:/user-software:image-src=/ $ENV_DIR/earthrovers.sif \
    python -m earthrovers.data_wrangling.process_raw_dataset \
    dataset_path=$DATASET_PATH \
    dataset_output_path=$OUTPUT_PATH \
    num_workers=20 \
    hf_chunk_size=10GB
