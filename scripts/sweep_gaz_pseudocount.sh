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

NTRAIN="--nTrain 1000"
NVALID="--nValid 100"
GAZ="--nGaz 100"
NREPL="--nReplication 5"
PARAM="--numParticles 32 --numMCMCIter 1"

if [ ! -d ${EXPT_PATH} ]; then
    mkdir ${EXPT_PATH}
fi

counts=( 1 5 10 25 50 100 )
for c in "${counts[@]}"
do
    FLAGS="$TRAIN $VALID ${EXPT_PATH}/$c $NTRAIN $NVALID $GAZ"
    FLAGS="$FLAGS --nTrainFold 10 $NREPL"
    echo "gazeteer pseudo-count: $c"
    COUNT1="$FLAGS --gazPseudocount $c"
    scripts/replications.py $COUNT1
    #cp ${EXPT_PATH}/$c/model.dat ${EXPT_PATH}/turkish_pseudo_$c.dat
done

# Baseline
echo "baseline run"
FLAGS="$TRAIN $VALID ${EXPT_PATH}/baseline $NTRAIN $NVALID $GAZ"
FLAGS="$FLAGS --nTrainFold 10 $NREPL --baseline --baselineOnly"
scripts/replications.py $FLAGS
#cp ${EXPT_PATH}/baseline/baseline.dat ${EXPT_PATH}/baseline.dat

#scripts/plot_f1_seaborn.py turkish_pseudo_*.dat baseline.dat --output turkish_pseudocount.pdf

#eof
