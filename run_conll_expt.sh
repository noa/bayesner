#! /usr/bin/env bash

# processed 51362 tokens with 5942 phrases; found: 6124 phrases; correct: 5041.
# accuracy:  96.80%; precision:  82.32%; recall:  84.84%; FB1:  83.56
# LOC:  precision:  87.27%; recall:  89.60%; FB1:  88.42  1886
# MISC: precision:  81.47%; recall:  82.97%; FB1:  82.21  939
# ORG:  precision:  69.71%; recall:  75.17%; FB1:  72.34  1446
# PER:  precision:  87.53%; recall:  88.06%; FB1:  87.79  1853

scripts/run_expt_smc.sh data/conll/en/train.utf8 data/conll/en/valid.utf8 NONE

# eof
