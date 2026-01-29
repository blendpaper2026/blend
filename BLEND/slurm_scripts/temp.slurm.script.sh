#!/bin/bash
#
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10000
#SBATCH --job-name=trainer + datasetname + subsample_class
#SBATCH --output=oOp-main/output/slurms/ trainer/ datasetname/subsample_class.out
#SBATCH --mail-user=your_email@example.com
#SBATCH --mail-type=ALL
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --account=mikel
#SBATCH --cluster=ub-hpc
#SBATCH --gpus-per-node=1
#SBATCH --constraint=V100

apptainer exec --nv pytorch.sif python '/projects/academic/alipour/payamabd/CoOp-main/train.py' --trainer 'Fed_MMRL' --dataset-config-file 'CoOp-main/configs/datasets/dtd.yaml' --subsample_class 'new' --eval-only --model-dir 'CoOp-main/output' --config-file 'CoOp-main/configs/trainers/FFM_REP/vit_b16.yaml' --seed 1

and on for 


apptainer exec --nv pytorch.sif python '/projects/academic/alipour/payamabd/CoOp-main/train.py' --trainer 'Fed_MMRL' --dataset-config-file 'CoOp-main/configs/datasets/dtd.yaml' --subsample_class 'base' --eval-only False --model-dir '' --config-file 'CoOp-main/configs/trainers/FFM_REP/vit_b16.yaml' --seed 1