#!/bin/bash

#SBATCH --job-name=lt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,v100,rtx8000

# job info
LOSS=$1
FEAT=$2
BN=$3


# Singularity path
ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/lg154/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
--overlay /scratch/lg154/sseg/dataset/tiny-imagenet-200.sqf:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
python main.py --dataset cifar10 -a mresnet32 --imbalance_rate 0.01 --beta 0.5 --lr 0.01 -b 64 --branch2 --contrast --bias \
--weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 4 --store_name baseline1
 " 


 # --bias 