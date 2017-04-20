#! /usr/bin/env bash

# accuracy:  96.86%; precision:  82.08%; recall:  86.59%; FB1:  84.28
# LOC:               precision:  86.11%; recall:  90.75%; FB1:  88.36  1936
# MISC:              precision:  83.55%; recall:  82.10%; FB1:  82.82  906
# ORG:               precision:  67.99%; recall:  76.66%; FB1:  72.06  1512
# PER:               precision:  88.45%; recall:  91.91%; FB1:  90.15  1914

scripts/run_expt_mcmc.sh data/conll/en/train.utf8 data/conll/en/valid.utf8 NONE

# eof
