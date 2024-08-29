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
CS=$2


# Singularity path
ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/lg154/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
--overlay /scratch/lg154/sseg/dataset/tiny-imagenet-200.sqf:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
python main_nc.py --dataset cifar100 -a mresnet32 --epochs 200 --scheduler cosine \
  --loss ce --coarse ${Y} --aug pc --batch_size 1024 --lr 0.05 --debug -1 \
  --cs_loss --cs_wt ${CS} \
  --seed 2021 --store_name cf100_mresnet32_ce_cs${CS}_Y${Y}_b1024
"

