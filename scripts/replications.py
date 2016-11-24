#! /usr/bin/env python3

import argparse
import subprocess
import numpy as np
import progressbar # pip install progressbar2
import statistics as stat
import numpy as np, scipy.stats as st
import nltk
from conll import ConllNECorpusReader

# Temporary output files
TMP_VALID = 'replications_valid.tab'
TMP_TRAIN = 'replications_train.tab'
TMP_GAZ   = 'replications_gaz.tab'

parser = argparse.ArgumentParser()
parser.add_argument('trainFile')
parser.add_argument('validFile')
parser.add_argument('--nTrain', type=int, default=100)
parser.add_argument('--nValid', type=int, default=250)
parser.add_argument('--nGaz', type=int, default=50)
parser.add_argument('--nReplication', type=int, default=5)
parser.add_argument('--nTrainFold', type=int, default=10)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--delim', default="\t")
parser.add_argument('--otherLabel', default='O')
parser.add_argument('--baseline', default=False, action='store_true')
parser.add_argument('--delta', default=False, action='store_true')
parser.add_argument('--baselineOutPath', default='baseline.dat')
parser.add_argument('--modelOutPath', default='model.dat')
parser.add_argument('--deltaOutPath', default='delta.dat')
parser.add_argument('--repeatGaz', default=False, action='store_true')
args = parser.parse_args()

# Set random seed
np.random.seed(args.seed)

# Set FEATURES to a features config file and PATH to the Stanford NER distribution
# Tested with the version number shown below.
BASELINE_PATH = 'stanford-ner-2015-12-09'
#BASELINE_FEATURES = '{}/featuresNoShape.prop'.format(BASELINE_PATH)
BASELINE_FEATURES = '{}/features.prop'.format(BASELINE_PATH)
BASELINE_MODEL = 'crf.model'
BASELINE_CLASSPATH = '"{}/stanford-ner.jar:{}/lib/*"'.format(BASELINE_PATH, BASELINE_PATH)
BASELINE_CMD = 'java -server -cp {} -d64 -Xmx10g edu.stanford.nlp.ie.crf.CRFClassifier'.format(BASELINE_CLASSPATH)

MODEL_CMD = 'bash scripts/run_expt_smc.sh'

def BASELINE_TRAIN(trainFile, gazFile):
    return '{} -prop {} -serializeTo {} -trainFile {} -useGazettes=true -gazette {}'.format(BASELINE_CMD, BASELINE_FEATURES, BASELINE_MODEL, trainFile, gazFile)

def BASELINE_VALID(validFile):
    return '{} -loadClassifier {} -testFile {}'.format(BASELINE_CMD, BASELINE_MODEL, validFile)

def baseline_f1(proc):
    err = proc.stderr.decode('ascii')
    for line in err.split("\n"):
        if line.lstrip().startswith("Totals"):
            return float(line.split()[3])*100.0
    return 0.0

def model_f1(proc):
    out = proc.stdout.decode('ascii')
    for line in out.split("\n"):
        if line.lstrip().startswith("accuracy"):
            return float(line.split()[7])

def run(cmd, capture=False):
    proc = subprocess.run(cmd,
                          shell=True,
                          check=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    return proc

def gazFromConll(inPath):
    gaz = None
    if args.repeatGaz:
        gaz = []
    else:
        gaz = set()

    reader = ConllNECorpusReader(".", inPath)
    sents = reader.ne_words()

    #if args.nGaz < 1:
    #    return None

    for s in sents:
        for r in s:
            instance = None
            if type(r) == tuple:
                instance = (r[0], r[1])

            if type(r) == nltk.Tree:
                label = r.label()
                text = " ".join([ x[0] for x in r ])
                instance = (text, label)

            if args.repeatGaz:
                gaz += [ instance ]
            else:
                gaz.add( instance )

    return gaz
def writeGaz_CoNLL(gaz, outPath):
    with open(outPath, 'w') as out:
        for e in gaz:
            tokens = e[0].split()
            label = e[1]
            if label == args.otherLabel:
                out.write(tokens[0] + " " + label + "\n")
            else:
                out.write(tokens[0] + " " + "B-" + label + "\n")
                i = 1
                while i < len(tokens):
                    out.write(tokens[i] + " " + "I-" + label + "\n")
                    i += 1
        out.write("\n")

def writeGaz_Stanford(gaz):
    with open(outPath, 'w') as out:
        for e in gaz:
            out.write(gaz[1] + " " + gaz[0] + "\n")

def convertGaz(inPath, outPath):
    reader = ConllNECorpusReader(".", inPath)
    sents = reader.ne_words()
    with open(outPath, 'w') as out:
        if args.nGaz < 1:
            out.write("")
            return
        for s in sents:
            for r in s:
                if type(r) == tuple:
                    text = r[0]
                    label = r[1]
                    out.write(label + " " + text + "\n")

                if type(r) == nltk.Tree:
                    label = r.label()
                    text = " ".join([ x[0] for x in r ])
                    out.write(label + " " + text + "\n")

def baseline_run_expt(trainPath, validPath, gazPath):
    newGazPath = gazPath+'.tmp'
    convertGaz(gazPath, newGazPath)
    run( BASELINE_TRAIN(trainPath, newGazPath) )
    p = run( BASELINE_VALID(validPath) )
    return baseline_f1(p)

def model_run_expt(trainPath, validPath, gazPath):
    cmd = '{} {} {} {}'.format(MODEL_CMD, trainPath, validPath, gazPath)
    p = run( cmd )
    return model_f1(p)

# Read instances from CoNLL file
def readConll(path):
    instance = []
    instances = []
    for line in open(path):
        tokens = line.split()
        if len(tokens) == 2:
            instance += [ tokens ]
        else:
            instances += [ instance ]
            instance = []
    return instances

# Read train & valid data
allTrain = readConll(args.trainFile)
allValid = readConll(args.validFile)
#np.random.shuffle(allTrain)
#np.random.shuffle(allValid)

#train = allTrain[0:args.nTrain]
#valid = allValid[0:args.nValid]
#gaz   = allTrain[args.nTrain:args.nTrain+args.nGaz]

print('Total number of training instances: '   + str(len(allTrain)))
print('Total number of validation instances: ' + str(len(allValid)))

def write_instances(path, instances):
    vout = open(path, 'w')
    for instance in instances:
        for token in instance:
            vout.write(args.delim.join(token)+'\n')
        vout.write('\n')
    vout.close()

# Count number of runs
incr = int(args.nTrain/args.nTrainFold)
incrs = []
foldStarts = range(0, args.nTrain, incr)
nExpts = len(foldStarts) * args.nReplication
print(str(len(foldStarts)) + ' * ' + str(args.nReplication) + ' = ' + str(nExpts) + ' experiments')
for startIndex in foldStarts:
    fold = allTrain[0:startIndex+incr]
    incrs += [ len(fold) ]

# Scores
model_scores = []
baseline_scores = []
for i in range(len(foldStarts)):
    model_scores.append([])
    baseline_scores.append([])

# Perform replications
startIndex = 0
with progressbar.ProgressBar(max_value=nExpts) as bar:
    i = 0
    for r in range(args.nReplication):
        # Create random train/valid/gaz splits
        np.random.shuffle(allTrain)
        np.random.shuffle(allValid)

        valid = allValid[0:args.nValid]
        gaz   = allTrain[args.nTrain:args.nTrain+args.nGaz]

        assert len(gaz) == args.nGaz, "insufficient instances for gazetteer"

        # Write validation data
        write_instances(TMP_VALID, valid)

        # Write gazetteer data
        write_instances(TMP_GAZ, gaz)
        processedGaz = gazFromConll(TMP_GAZ)
        #print(str(len(processedGaz)) + " gazetteer entries")
        writeGaz_CoNLL(processedGaz, TMP_GAZ)

        j = 0
        for startIndex in foldStarts:
            fold = allTrain[0:startIndex+incr]
#            incrs += [ len(fold) ]
            write_instances(TMP_TRAIN, fold)
            if args.baseline or args.delta:
                baseline_scores[j] += [ baseline_run_expt(TMP_TRAIN, TMP_VALID, TMP_GAZ) ]
            model_scores[j]    += [ model_run_expt(TMP_TRAIN, TMP_VALID, TMP_GAZ) ]
            bar.update(i)
            j += 1
            i += 1

# Write discriminative results
if args.baseline:
    with open(args.baselineOutPath, 'w') as out:
        for j in range(len(foldStarts)):
            s = [ str(incrs[j]) ] + [ str(x) for x in baseline_scores[j] ]
            out.write(" ".join(s)+"\n")


# Write delta results
if args.delta:
    with open(args.deltaOutPath, 'w') as out:
        for j in range(len(foldStarts)):
            s = [ str(incrs[j]) ]
            for i in range(len(baseline_scores[j])):
                s += [ str(model_scores[j][i] - baseline_scores[j][i]) ]
            out.write(" ".join(s)+"\n")

# Write model results
with open(args.modelOutPath, 'w') as out:
    for j in range(len(foldStarts)):
        s = [ str(incrs[j]) ] + [ str(x) for x in model_scores[j] ]
        out.write(" ".join(s)+"\n")

# Clean up
# for path in [TMP_TRAIN, TMP_VALID, TMP_GAZ]:
#     subprocess.run(['rm', path])

# eof
