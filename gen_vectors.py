# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import ngram
import numpy as np
import os
from scipy import sparse
import sys

import chars

def parse_arguments():
  usage = 'Usage: python {} FILE'.format(__file__)
  argparser = ArgumentParser()
  argparser.add_argument('src_filename', type=str, help='source file name')
  argparser.add_argument('dest_dirname', type=str, help='destination directory name')
  args = argparser.parse_args()
  return args

def read(file):
  vecs = []
  line = file.readline().rstrip()
  while line:
    vec = np.zeros(chars.LEN * chars.LEN, dtype=np.float32)

    for e in G.split(line):
      hi = e[0]
      lo = e[1]
      if hi in chars.DICT and lo in chars.DICT:
        hi = chars.DICT[hi]
        lo = chars.DICT[lo]
        n = hi * chars.LEN + lo
        vec[n] = vec[n] + 1.0

    vecs.append(vec)
    if len(vecs) == 10000:
      yield vecs
      vecs = []

    line = file.readline().rstrip()

  if len(vecs) >= 1:
    yield vecs

if __name__ == '__main__':
  args = parse_arguments()
  src_filename = os.path.abspath(args.src_filename)
  dest_dirname = os.path.abspath(args.dest_dirname)

  os.makedirs(dest_dirname, exist_ok=True)

  G = ngram.NGram(N=2)
  vecs = []

  with open(src_filename) as src_file:
    cnt = 0

    for vecs in read(src_file):
      vecs = np.array(vecs)
      csr = sparse.csr_matrix(vecs)
      dest_filename = 'vec{:05d}.npz'.format(cnt)
      dest_filename = os.path.join(dest_dirname, dest_filename)
      print(dest_filename)
      cnt = cnt + 1
      sparse.save_npz(dest_filename, csr)
