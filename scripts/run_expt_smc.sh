#! /usr/bin/env bash

set -e

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 TRAIN_PATH VALID_PATH GAZ_PATH"
    exit
fi

PRED=pred.txt
MODE=smc
MODEL=seg
PROPOSAL=hybrid
NPARTICLES=64
MODEL="--nparticles=$NPARTICLES --mode=$MODE --model=$MODEL"

CMD="nname --train=$1 --test=$2 --gazetteer=$3 --out_path=$PRED $MODEL"
echo $CMD

GLOG_logtostderr=1 OMP_NUM_THREADS=$NTHREAD $CMD
scripts/conlleval.pl < $PRED
