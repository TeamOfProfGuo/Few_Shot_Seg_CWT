#!/bin/bash

#SBATCH --job-name=p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
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
            --overlay /scratch/xl3139/overlay-25GB-500K-PROTR.ext3:ro \
            /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c " source /ext3/env.sh;
            python -m src.train_cca --config config_files/${DATA}_cca.yaml \
					 --opts train_split ${SPLIT} \
						    layers ${LAYERS} \
						    shot ${SHOT} \
					 > log_cca.txt 2>&1"

echo "finish"


#GREENE GREENE_GPU_MPS=yes


