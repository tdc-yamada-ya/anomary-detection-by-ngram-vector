# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import numpy as np
import os
import re
from scipy import sparse
from sklearn import cluster

import chars

def parse_arguments():
  usage = 'Usage: python {} FILE'.format(__file__)
  argparser = ArgumentParser()
  argparser.add_argument('src_dirname', type=str, help='source dir name')
  argparser.add_argument('dest_dirname', type=str, help='destination dir name')
  args = argparser.parse_args()
  return args

def find_files(dirname):
  cnt = 0
  
  while True:
    filename = 'vec{:05d}.npz'.format(cnt)
    filename = os.path.join(dirname, filename)

    if not os.path.isfile(filename):
      break

    yield filename
    cnt = cnt + 1

if __name__ == '__main__':
  args = parse_arguments()
  src_dirname = os.path.abspath(args.src_dirname)
  dest_dirname = os.path.abspath(args.dest_dirname)

  os.makedirs(dest_dirname, exist_ok=True)

  cnt = 0
  kmeans = cluster.MiniBatchKMeans(n_clusters=10)
  
  print('Fit pass')

  for src_filename in find_files(src_dirname):
    print('src: {}'.format(src_filename))
    vec = sparse.load_npz(src_filename)
    kmeans.fit(vec)

  print('Predict pass')

  for src_filename in find_files(src_dirname):
    print('src: {}'.format(src_filename))
    index = re.search('([0-9]+)', os.path.basename(src_filename)).group(1)
    dest_filename = 'pred{}.npy'.format(index)
    dest_filename = os.path.join(dest_dirname, dest_filename)
    print('dest: {}'.format(dest_filename))
    vec = sparse.load_npz(src_filename)
    pred = kmeans.predict(vec)
    np.save(dest_filename, pred)
