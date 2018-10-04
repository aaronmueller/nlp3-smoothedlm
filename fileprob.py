#!/usr/bin/env python

# Sample program for hw-lm
# CS465 at Johns Hopkins University.

# Converted to python by Eric Perlman <eric@cs.jhu.edu>

# Updated by Jason Baldridge <jbaldrid@mail.utexas.edu> for use in NLP
# course at UT Austin. (9/9/2008)

# Modified by Mozhi Zhang <mzhang29@jhu.edu> to add the new log linear model
# with word embeddings.  (2/17/2016)

from __future__ import print_function

import math
import sys

import Probs

# Computes the log probability of the sequence of tokens in file,
# according to a trigram model.  The training source is specified by
# the currently open corpus, and the smoothing method used by
# prob() is specified by the global variable "smoother". 

def get_model_filename(smoother, lexicon, train_file):
    import hashlib
    from os.path import basename
    train_hash = basename(train_file)
    lexicon_hash = basename(lexicon)
    filename = '{}_{}_{}.model'.format(smoother, lexicon_hash, train_hash)
    return filename

def main():
  course_dir = '/usr/local/data/cs465/'

  if len(sys.argv) < 5 or (sys.argv[1] == 'TRAIN' and len(sys.argv) != 5):
    print("""
Prints the log-probability of each file under a smoothed n-gram model.

Usage:   {} TRAIN smoother lexicon trainpath
         {} TEST smoother lexicon trainpath files...
Example: {} TRAIN add0.01 {}hw-lm/lexicons/words-10.txt switchboard-small
         {} TEST add0.01 {}hw-lm/lexicons/words-10.txt switchboard-small {}hw-lm/speech/sample*

Possible values for smoother: uniform, add1, backoff_add1, backoff_wb, loglinear1
  (the \"1\" in add1/backoff_add1 can be replaced with any real lambda >= 0
   the \"1\" in loglinear1 can be replaced with any C >= 0 )
lexicon is the location of the word vector file, which is only used in the loglinear model
trainpath is the location of the training corpus
  (the search path for this includes "{}")
""".format(sys.argv[0], sys.argv[0], sys.argv[0], course_dir, sys.argv[0], course_dir, course_dir, Probs.DEFAULT_TRAINING_DIR))
    sys.exit(1)

  mode = sys.argv[1]
  argv = sys.argv[2:]

  smoother = argv.pop(0)
  lexicon = argv.pop(0)
  train_file = argv.pop(0)

  if mode == 'TRAIN':
    lm = Probs.LanguageModel()
    lm.set_smoother(smoother)
    lm.read_vectors(lexicon)
    lm.train(train_file)
    lm.save(get_model_filename(smoother, lexicon, train_file))
  elif mode == 'TEST':
    if not argv:
      print("warning: no input files specified")
    lm = Probs.LanguageModel.load(get_model_filename(smoother, lexicon, train_file))
    # We use natural log for our internal computations and that's
    # the kind of log-probability that fileLogProb returns.  
    # But we'd like to print a value in bits: so we convert
    # log base e to log base 2 at print time, by dividing by log(2).

    total_cross_entropy = 0.
    for testfile in argv:
      ce = lm.filelogprob(testfile) / math.log(2)
      print("{:g}\t{}".format(ce, testfile))
      total_cross_entropy -= ce
    print('Overall cross-entropy:\t{0:.5f}'.format(total_cross_entropy/sum([lm.num_tokens(testfile) for testfile in argv])))
  else:
    sys.exit(-1)

if __name__ ==  "__main__":
  main()

