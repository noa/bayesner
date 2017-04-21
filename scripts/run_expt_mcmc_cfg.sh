#! /usr/bin/env bash

set -e

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 TRAIN_PATH VALID_PATH GAZ_PATH NPARTICLE GAZ_COUNT NUM_ITER"
fi

EXE="build/src/cli/nname"
PRED=pred.txt
MODE=pgibbs
MODEL=seg
PROPOSAL=hybrid
NPARTICLES=$4
GAZ_COUNT=$5
MCMC_ITER=$6
NTHREAD=16
MODEL="--nparticles=$NPARTICLES --mode=$MODE --model=$MODEL"
MODEL="$MODEL --nmcmc_iter=${MCMC_ITER} --gazetteer_pseudocount=${GAZ_COUNT}"
CMD="$EXE --train=$1 --test=$2 --gazetteer=$3 --out_path=$PRED $MODEL"

echo $CMD

GLOG_logtostderr=1 OMP_NUM_THREADS=$NTHREAD $CMD
scripts/conlleval.pl < $PRED
