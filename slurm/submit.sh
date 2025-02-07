#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=35:59:00
#SBATCH --qos=medium
#SBATCH --gres=gpu:a40
 
export APPTAINER_CWD=/workspace

apptainer run \
    --nv \
    --containall \
    --env PYTHONPATH=/workspace \
    --bind $(pwd):/workspace \
    --bind /tudelft.net/staff-umbrella/neon/experiments/VIT003/:/workspace/data/conflab \
    --bind /tmp:/tmp \
    /tudelft.net/staff-umbrella/neon/apptainer/vitpose-0.0.5.sif \
    python /workspace/tools/train.py configs/ViTPose_coco_plus_conflab_w_bg_256x192.py \
    --cfg-options model.pretrained=data/conflab/models/vitpose_base_coco_aic_mpii.pth --seed 0

# sbatch --job-name vitpose-conflab --account ewi-insy-prb --partition insy,general slurm/submit.sh
