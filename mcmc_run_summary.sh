#! /usr/bin/env bash

for f in *.conll; do
    scripts/conlleval.pl < $f | grep "accuracy:"
done

# eof
