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
Y=$1
ARCH=$2
LR=$3



# Singularity path
ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/lg154/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
--overlay /scratch/lg154/sseg/dataset/tiny-imagenet-200.sqf:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
python main_nc.py --dataset cifar100 -a ${ARCH} --epochs 300 --scheduler ms \
  --loss ce --coarse ${Y} --aug pc --batch_size 128 --lr ${LR} --nc_freq 10 \
  --seed 2021 --store_name ${ARCH}_Y${Y}_Apc_LR${LR}
"

