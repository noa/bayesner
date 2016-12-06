#! /usr/bin/env bash

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 VALID_PATH MODEL_PATH"
    exit
fi

EXE="build/src/cli/nname"
PRED=pred.txt
MODE=smc
MODEL=seg
PROPOSAL=hybrid
NPARTICLES=64
NTHREAD=8
MODEL="--nparticles=$NPARTICLES --mode=$MODE --model=$MODEL"

CMD="$EXE --test=$1 --model_path=$2 --test_only $MODEL"
echo $CMD

GLOG_logtostderr=1 OMP_NUM_THREADS=$NTHREAD $CMD
scripts/conlleval.pl < $PRED
