#! /usr/bin/env bash

set -e

if [[ ("$#" -lt 2) || ("$#" -gt 3) ]]; then
    echo "Usage: $0 TRAIN_PATH VALID_PATH [GAZ_PATH]"
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

if [ "$#" -eq 2 ]; then
    CMD="$EXE --train=$1 --test=$2 --gazetteer=$3 --out_path=$PRED $MODEL"
elif [ "$#" -eq 3 ]; then
    CMD="$EXE --train=$1 --test=$2 --out_path=$PRED $MODEL"
else
    echo "Bad number of arguments: $#"
    exit
fi

echo $CMD

GLOG_logtostderr=1 OMP_NUM_THREADS=$NTHREAD $CMD
scripts/conlleval.pl < $PRED
