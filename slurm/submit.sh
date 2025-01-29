#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=35:59:00
#SBATCH --qos=medium
#SBATCH --gres=gpu:a40
 
apptainer run \
    --nv \
    --containall \
    --bind $(pwd):/workspace \
    --bind /tudelft.net/staff-umbrella/neon/experiments/VIT003/:/workspace/data/conflab \
    /tudelft.net/staff-umbrella/neon/apptainer/vitpose-0.0.5.sif

# sbatch --job-name vitpose-conflab --account ewi-insy-prb --partition insy,general submit.sh