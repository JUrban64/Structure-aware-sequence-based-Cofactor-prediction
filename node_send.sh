#!/bin/bash
#PBS -N TRAIN_SQ_test_7
#PBS -l select=1:ncpus=2:ngpus=1:mem=70gb:scratch_ssd=70gb
#PBS -l walltime=06:00:00



DATADIR=/auto/brno2/home/urbany/SeQbCoP/Structure-aware-sequence-based-Cofactor-prediction

module add mambaforge
mamba activate sqbcp_cpu

source /storage/brno2/home/urbany/miniconda3/etc/profile.d/conda.sh
conda activate sqbcp_cpu

source activate /storage/brno2/home/urbany/.conda/envs/sqbcp_cpu

cd $SCRATCHDIR 



cp $DATADIR/*.py $SCRATCHDIR 
cp -r $DATADIR/data $SCRATCHDIR 


python3 run_pipeline.py



cp *.pth $DATADIR/


clean_scratch
