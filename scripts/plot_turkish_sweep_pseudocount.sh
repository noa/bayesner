#! /usr/bin/env bash

set -e

TRAIN="data/turkish_train.conll"
VALID="data/turkish_valid.conll"

NTRAIN="--nTrain 100"
NVALID="--nValid 100"
NREPL="--nReplication 10"
#CONFIG="expts/config.cfg"
FLAGS="$TRAIN $VALID --baseline $NTRAIN $NVALID --nGaz 1"
FLAGS="$FLAGS --nTrainFold 10 $NREPL"

FILE=turkish_sweep_pseudocount_1.pdf
if [ ! -f $FILE ]; then
    scripts/replications.py $FLAGS
    scripts/plot_f1_seaborn.py baseline.dat model.dat --output $FILE
fi
