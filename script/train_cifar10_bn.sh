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
python main_bn.py --dataset cifar10 -a mresnet32 --imbalance_rate 0.01 --imbalance_type exp --lr 0.01 --seed 2021 \
 --epochs 200 --loss ${LOSS} --feat ${FEAT} --bn_type ${BN} --aug pc --bias --mixup -1 --mixup_alpha 1 \
 --store_name exp0.01_mresnet32_${LOSS}_${FEAT}_${BN}_b_pc1
 " 


 # --bias 