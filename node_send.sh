#!/bin/bash
#PBS -N TRAIN_SQ_test_13
#PBS -l select=1:ncpus=4:ngpus=1:mem=160gb:scratch_ssd=70gb
#PBS -l walltime=04:00:00



DATADIR=/auto/brno2/home/urbany/SeQbCoP/Structure-aware-sequence-based-Cofactor-prediction

module add mambaforge
mamba activate sqbcp_gpu

source /storage/brno2/home/urbany/miniconda3/etc/profile.d/conda.sh
conda activate sqbcp_gpu

source activate /storage/brno2/home/urbany/.conda/envs/sqbcp_gpu

cd $SCRATCHDIR 

mkdir splits



cp $DATADIR/*.py $SCRATCHDIR 
cp -r $DATADIR/data $SCRATCHDIR 
cp -r $DATADIR/cache $SCRATCHDIR 2>/dev/null || true


python3 run_pipeline.py --batch-size 16 --epochs 50 --save-splits splits/ --load-splits splits/ 



cp -r splits/ $DATADIR/
cp *.pth $DATADIR/
cp -r cache/ $DATADIR/ 2>/dev/null || true

clean_scratch
