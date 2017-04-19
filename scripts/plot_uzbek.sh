#! /usr/bin/env bash

set -e

FILE=uzbek_100_500_25.png
if [ ! -f $FILE ]; then
    scripts/replications.py data/conll/uzbek_train.conll data/conll/uzbek_valid.conll --baseline --nTrain 100 --nValid 500 --nGaz 25 --nTrainFold 10 --nReplication 1
    scripts/plot_f1_seaborn.py baseline.dat model.dat --output $FILE
fi

declare -a arr=(10 100 1000)
for size in "${arr[@]}"
do
    scripts/replications.py data/conll/uzbek_train.conll data/conll/uzbek_valid.conll --delta --nTrain 100 --nValid 250 --nGaz $size --nTrainFold 10 --nReplication 5
    cp delta.dat delta_$size.dat
done

scripts/plot_f1_seaborn.py delta_*.dat --output uzbek.pdf

# eof
