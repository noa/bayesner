#! /usr/bin/env bash

set -e

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 TRAIN_PATH VALID_PATH GAZ_PATH"
    exit
fi

EXE="build/src/cli/nname"
PRED=pred.txt
MODE=smc
MODEL=seg
PROPOSAL=hybrid
NPARTICLES=64
NTHREAD=8
MODEL="--mode=$MODE --model=$MODEL"

CMD="$EXE --train=$1 --gazetteer=$2 --model_path=$3 --train_only $MODEL"
echo $CMD

GLOG_logtostderr=1 OMP_NUM_THREADS=$NTHREAD $CMD
scripts/conlleval.pl < $PRED
