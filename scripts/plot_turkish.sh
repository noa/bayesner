#! /usr/bin/env bash

set -e

TRAIN="data/turkish_train.conll"
VALID="data/turkish_valid.conll"

NTRAIN="--nTrain 100"
NVALID="--nValid 100"
NREPL="--nReplication 3"

FILE=turkish_100_250_0.png
if [ ! -f $FILE ]; then
    scripts/replications.py $TRAIN $VALID --baseline $NTRAIN $NVALID --nGaz 1 --nTrainFold 10 $NREPL
    scripts/plot_f1_seaborn.py baseline.dat model.dat --output $FILE
fi

FILE=turkish_100_250_25.png
if [ ! -f $FILE ]; then
    scripts/replications.py $TRAIN $VALID --baseline $NTRAIN $NVALID --nGaz 25 --nTrainFold 10 $NREPL
    scripts/plot_f1_seaborn.py baseline.dat model.dat --output $FILE
fi

declare -a arr=(10 100 1000)
for size in "${arr[@]}"
do
    echo "gazetteer size: $size"
    scripts/replications.py $TRAIN $VALID --delta $NTRAIN $NVALID --nGaz $size --nTrainFold 10 $NREPL
    cp delta.dat turkish_delta_$size.dat
    cp model.dat turkish_model_$size.dat
    cp baseline.dat turkish_baseline_$size.dat
done

scripts/plot_f1_seaborn.py turkish_delta_*.dat --output turkish_delta.png

# eof
