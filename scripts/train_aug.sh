#!/bin/bash

#SBATCH --job-name=aug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20GB
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=xl3139@nyu.edu
#SBATCH --output=aug.out
#SBATCH --gres=gpu # How much gpu need, n is the number
#SBATCH --partition=a100




module purge

DATA=$1
EXP_ID=$2
SPLIT=$3
LAYERS=$4
SHOT=$5



echo "start"
singularity exec --nv \
            --overlay /scratch/xl3139/overlay-25GB-500K-PROTR.ext3:ro \
            /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c " source /ext3/env.sh;
            python -m src.train_aug --config config_files/${DATA}_aug.yaml \
                    --exp_id ${EXP_ID} \
					--opts train_split ${SPLIT} \
						    layers ${LAYERS} \
						    shot ${SHOT} \
					 > log_aug.txt 2>&1"

echo "finish"


#GREENE GREENE_GPU_MPS=yes


