#! /usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 EXPT_PATH"
    exit
fi

set -e

EXPT_PATH=$1

TRAIN="data/conll/turkish_train.conll"
VALID="data/conll/turkish_valid.conll"

NTRAIN="--nTrain 1000"
NVALID="--nValid 100"
GAZ="--nGaz 100"
NREPL="--nReplication 5"
PARAM="--numParticles 64 --numMCMCIter 5"

counts=( 1 5 10 25 )
for c in "${counts[@]}"
do
    FLAGS="$TRAIN $VALID ${EXPT_PATH}_$c $NTRAIN $NVALID $GAZ"
    FLAGS="$FLAGS --nTrainFold 10 $NREPL"
    echo "gazeteer pseudo-count: $c"
    COUNT1="$FLAGS --gazPseudocount $c"
    scripts/replications.py $COUNT1
    cp ${EXPT_PATH}_$c/model.dat turkish_pseudo_$c.dat
done

scripts/plot_f1_seaborn.py turkish_pseudo_*.dat --output turkish_pseudocount.pdf

#eof
