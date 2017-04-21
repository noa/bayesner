#! /usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('series', nargs='+')
parser.add_argument('--output', default="plot.png")
args = parser.parse_args()

AX='# sentence'
AY='F1'
MODEL='Model'
SUBJ='subject'

raw_data = {
    AX: [],
    AY: [],
    MODEL: [],
    SUBJ: []
}

for path in args.series:
    print('reading: ' + path)
    model, ext = os.path.splitext(os.path.basename(path))
    for line in open(path):
        tokens = line.rstrip().split()
        nsent = tokens[0]
        idx = 0
        for f1 in tokens[1:]:
            raw_data[AX] += [float(nsent)]
            raw_data[AY] += [float(f1)]
            raw_data[MODEL] += [model]
            raw_data[SUBJ] += [str(idx)]
            idx += 1

df = pd.DataFrame(raw_data, columns = [AX,
                                       AY,
                                       MODEL,
                                       SUBJ])


sns.set(style="darkgrid")


ax = sns.tsplot(time=AX, value=AY,
                condition=MODEL, unit=SUBJ,
                data=df)
fig = ax.get_figure()
fig.savefig(args.output)

# eof
