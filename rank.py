# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from collections import Counter
import numpy as np
import os
import re

def parse_arguments():
  usage = 'Usage: python {} FILE'.format(__file__)
  argparser = ArgumentParser()
  argparser.add_argument('src_dirname', type=str, help='source dir name')
  argparser.add_argument('dest_filename', type=str, help='destination file name')
  args = argparser.parse_args()
  return args

def find_files(dirname):
  cnt = 0
  
  while True:
    filename = 'pred{:05d}.npy'.format(cnt)
    filename = os.path.join(dirname, filename)

    if not os.path.isfile(filename):
      break

    yield filename
    cnt = cnt + 1

if __name__ == '__main__':
  args = parse_arguments()
  src_dirname = os.path.abspath(args.src_dirname)
  dest_filename = os.path.abspath(args.dest_filename)

  print('Count pass')

  counter = Counter()

  for src_filename in find_files(src_dirname):
    print('src: {}'.format(src_filename))
    pred = np.load(src_filename)
    unique, counts = np.unique(pred, return_counts=True)
    counter = counter + Counter(dict(zip(unique, counts)))

  ranks = {key: rank for rank, key in enumerate(sorted(counter, key=counter.get, reverse=False), 1)}

  print('Rank pass')

  line_number = 1
  with open(dest_filename, mode='w') as dest_file:
    dest_file.write('Line,Rank\n')

    for src_filename in find_files(src_dirname):
      print('src: {}'.format(src_filename))
      pred = np.load(src_filename)
      for i in pred:
        rank = ranks[i]
        dest_file.write('{:d},{:d}\n'.format(line_number, rank))
        line_number = line_number + 1

#  with open(args.src_filename, mode='rb') as src_file:
#    array = np.load(src_file)
#    unique, counts = np.unique(array, return_counts=True)
#    summary = dict(zip(unique, counts))
#    min_k = min(summary, key=(lambda x: summary[x]))
#    print(np.where(array == min_k))