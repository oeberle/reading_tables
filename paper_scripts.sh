#!/usr/bin/env bash

####### Training the digit detection model #######
pthon run_train_table.py --jobfile data/configs/train_config.p --savedir results/exp01/


####### Extract table page representations #######

# Run script to compute table page representations for the full corpus
# indices i correspond to jobs specified in "data/inference/jobs_tables.p"

SAVEDIR="data/inference"
JOBFILE="data/inference/jobs_tables.p"
MODELJOBFILE="data/trained_model/jobdict.p"
WEIGHTFILE="data/trained_model/table_model.pt"
CASEFLAG="tables_sphaera"
EXTRASTR="experiment01"

for i in {1,} # {1,490} to run all jobs in JOBFILE
do
    echo $i
    python run_table_inference.py --sgeid $i --jobfile $JOBFILE --savedir $SAVEDIR --modeljobfile $MODELJOBFILE --weightfile $WEIGHTFILE --extrastr $EXTRASTR --caseflag $CASEFLAG
done

####### Reproduce insights analys #######

python run_evaluation.py

python run_geographical_analysis.py

python run_temporal_analysis.py
