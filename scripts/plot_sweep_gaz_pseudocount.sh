#! /usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 PATH"
    exit
fi

set -e

EPATH=$1

scripts/plot_f1_seaborn.py $EPATH/*.dat --output pseudocount_sweep.pdf

# eof
