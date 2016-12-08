#! /usr/bin/env bash

# scripts/train_model.sh data/conll/en/train.utf8 NONE model.ser
EXE="build/src/cli/nname"
MODEL="model.ser"
$EXE --stdin_decoder --model_path=$MODEL

# eof
