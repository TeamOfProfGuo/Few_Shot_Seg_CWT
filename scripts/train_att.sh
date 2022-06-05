#!/bin/bash

#SBATCH --job-name=seg8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20GB
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=lg154@nyu.edu
#SBATCH --output=seg.out
#SBATCH --gres=gpu # How much gpu need, n is the number

module purge

DATA=$1
SPLIT=$2
LAYERS=$3
SHOT=$4



echo "start"
singularity exec --nv \
            --overlay /scratch/lg154/python36/python36.ext3:ro \
            /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c " source /ext3/env.sh;
            python -m src.train_att --config config_files/${DATA}_att.yaml \
					 --opts train_split ${SPLIT} \
						    layers ${LAYERS} \
						    shot ${SHOT} \
						    cls_lr 0.1 \
						    batch_size 1 \
						    batch_size_val 1 \
						    epochs 20 \
					 > log.txt 2>&1"

echo "finish"


#GREENE GREENE_GPU_MPS=yes


