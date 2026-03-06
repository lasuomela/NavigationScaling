#!/bin/bash -l
#SBATCH --job-name=EarthRovers_build     # Job name
#SBATCH --output=logs/log_build.out      # Name of stdout output file
#SBATCH --error=logs/log_build.err       # Name of stderr error file
#SBATCH --partition=small               # partition name
#SBATCH --nodes=1                       # Total number of nodes 
#SBATCH --ntasks-per-node=1             # MPI ranks per node
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G                      # Total memory for job
#SBATCH --time=0-00:20:00               # Run time (d-hh:mm:ss)

# Reference:
# https://github.com/Lumi-supercomputer/Getting_Started_with_AI_workshop/blob/main/07_Extending_containers_with_virtual_environments_for_faster_testing/examples/extending_containers_with_venv.md

# Name of the image to create
IMAGE_NAME=earthrovers.sif

# Path to the package being developed
PKG_DIR=/scratch/$SLURM_JOB_ACCOUNT/NavigationScaling

# Where to store the image
INSTALL_DIR=/projappl/$SLURM_JOB_ACCOUNT/earthrovers

# Path to the conda environment file
ENV_FILE_PATH=/scratch/$SLURM_JOB_ACCOUNT/NavigationScaling/earthrovers/train/environment.lumi.yml

# Path to the image on which the image to create is based
BASE_IMAGE_PATH=/appl/local/containers/sif-images/lumi-rocm-rocm-6.2.4.sif

# Remove the old environment
if [ -d "$INSTALL_DIR" ]; then
    rm -rf $INSTALL_DIR
fi
mkdir -p $INSTALL_DIR

module purge
module load LUMI/24.03 cotainr

# Install the conda/pip dependencies in the .yml file
srun cotainr build $INSTALL_DIR/$IMAGE_NAME \
    --base-image=$BASE_IMAGE_PATH \
    --conda-env=$ENV_FILE_PATH \
    --accept-license

module use  /appl/local/containers/ai-modules
module load singularity-AI-bindings

# Install pip packages that can't be installed via requirements.txt / env.yml
# Install the package under development as editable
singularity exec $INSTALL_DIR/$IMAGE_NAME bash -c "
  python -m venv $INSTALL_DIR/myenv --system-site-packages &&
  source $INSTALL_DIR/myenv/bin/activate &&
  pip install git+https://github.com/bdaiinstitute/theia.git --no-deps &&
  pip install -e $PKG_DIR &&
  deactivate
"

# Create SquashFS image after container commands are done
mksquashfs $INSTALL_DIR/myenv $INSTALL_DIR/myenv.sqsh
rm -rf $INSTALL_DIR/myenv