#! /usr/bin/env bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 EXPT_PATH TRAIN VALID"
    exit
fi

set -e

EXPT_PATH=$1
TRAIN=$2
VALID=$3

NTRAIN="--nTrain 500"
NFOLD="--nTrainFold 10"
NVALID="--nValid 100"
NREPL="--nReplication 5"
PARAM="--numParticles 64 --numMCMCIter 10"


declare -a arr=(10 100 1000)
for size in "${arr[@]}"
do
    echo "gazetteer size: $size"
    ALL_ARGS="$TRAIN $VALID ${EXPT_PATH}/$size --delta $NTRAIN $NVALID $NFOLD $NREPL"
    scripts/replications.py ${ALL_ARGS} --nGaz $size
    #cp delta.dat turkish_delta_$size.dat
    #cp model.dat turkish_model_$size.dat
    #cp baseline.dat turkish_baseline_$size.dat
    cp ${EXPT_PATH}/$size/delta.dat ${EXPT_PATH}/$size.dat
done

#scripts/plot_f1_seaborn.py turkish_delta_*.dat --output turkish_delta.png

# eof
