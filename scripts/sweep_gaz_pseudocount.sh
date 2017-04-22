#! /usr/bin/env bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 EXPT_PATH TRAIN VALID"
    exit
fi

set -e

EXPT_PATH=$1

#TRAIN="data/conll/turkish_train.conll"
#VALID="data/conll/turkish_valid.conll"

TRAIN=$2
VALID=$3

INFERENCE="--inference smc"
NTRAIN="--nTrain 500"
NFOLD="--nTrainFold 25"
NVALID="--nValid 100"
GAZ="--nGaz 100"
NREPL="--nReplication 5"
PARAM="--numParticles 512"
#PARAM="--numParticles 64 --numMCMCIter 10"

if [ ! -d ${EXPT_PATH} ]; then
    mkdir ${EXPT_PATH}
fi

counts=( 1 10 100 )
for c in "${counts[@]}"
do
    FLAGS="$TRAIN $VALID ${EXPT_PATH}/$c $NTRAIN $NVALID $GAZ"
    FLAGS="$FLAGS $NFOLD $NREPL $INFERENCE"
    echo "gazeteer pseudo-count: $c"
    COUNT1="$FLAGS --gazPseudocount $c"
    scripts/replications.py $COUNT1
    cp ${EXPT_PATH}/$c/model.dat ${EXPT_PATH}/$c.dat
done

# Baseline
echo "baseline run"
FLAGS="$TRAIN $VALID ${EXPT_PATH}/baseline $NTRAIN $NVALID $GAZ"
FLAGS="$FLAGS $NFOLD $NREPL --baseline --baselineOnly"
scripts/replications.py $FLAGS
cp ${EXPT_PATH}/baseline/baseline.dat ${EXPT_PATH}/baseline.dat

#scripts/plot_f1_seaborn.py turkish_pseudo_*.dat baseline.dat --output turkish_pseudocount.pdf

#eof
