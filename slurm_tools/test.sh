#!/bin/bash -l
#SBATCH --job-name=EarthRovers_test     # Job name
#SBATCH --output=logs/log_test.out      # Name of stdout output file
#SBATCH --error=logs/log_test.err       # Name of stderr error file
#SBATCH --partition=dev-g               # partition name
#SBATCH --nodes=1                       # Total number of nodes 
#SBATCH --ntasks-per-node=1             # MPI ranks per node
#SBATCH --gpus-per-node=1               # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=7               # CPU cores per task
#SBATCH --mem=480G                      # Total memory for job
#SBATCH --time=0-02:00:00               # Run time (d-hh:mm:ss)

# Load the required modules
module purge
module load LUMI
module use  /appl/local/containers/ai-modules
module load singularity-AI-bindings

export WANDB_MODE=disabled

source ~/.bashrc
DATASET_PATH=/flash/$SLURM_JOB_ACCOUNT/FrodoBots8k
ENV_DIR=/projappl/$SLURM_JOB_ACCOUNT/earthrovers

export HF_HOME=/scratch/$SLURM_JOB_ACCOUNT/earthrovers_hf_cache
export TORCH_HOME=/scratch/$SLURM_JOB_ACCOUNT/torch_cache
export SINGULARITYENV_PREPEND_PATH=/user-software/bin # gives access to packages inside the container

# Set interfaces to be used by RCCL.
# This is needed as otherwise RCCL tries to use a network interface it has
# no access to on LUMI.
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=3
export MPICH_GPU_SUPPORT_ENABLED=1

# The MIOPEN_ environment variables are needed to make MIOpen create its caches
# on /tmp as doing this on Lustre fails because of file locking issues
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

if [ $SLURM_LOCALID -eq 0 ] ; then
    rm -rf $MIOPEN_USER_DB_PATH
    mkdir -p $MIOPEN_USER_DB_PATH
fi
sleep 2

# Run the training script
srun singularity exec \
   -B $ENV_DIR/myenv.sqsh:/user-software:image-src=/ $ENV_DIR/earthrovers.sif \
    python -m earthrovers.train.run \
        --config-name experiments/model_ablations/nomad.yaml \
        earthrovers.dataloader.dataset_path=$DATASET_PATH \
        earthrovers.trainer.num_gpus=$SLURM_GPUS_ON_NODE \
        earthrovers.trainer.num_nodes=$SLURM_JOB_NUM_NODES \
        earthrovers.dataloader.samples_to_load=-1 \
        earthrovers.dataloader.train_min_hours_per_location=0.1 \
        earthrovers.dataloader.train_num_locations=-1 \
        earthrovers.dataloader.persist_index=False

        
