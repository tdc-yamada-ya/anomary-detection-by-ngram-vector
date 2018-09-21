# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import numpy as np

def parseArguments():
  usage = 'Usage: python {} FILE'.format(__file__)
  argparser = ArgumentParser()
  argparser.add_argument('src_file_name', type=str, help='source file name')
  args = argparser.parse_args()
  return args

if __name__ == '__main__':
  args = parseArguments()
  with open(args.src_file_name, mode='rb') as src_file:
    array = np.load(src_file)
    unique, counts = np.unique(array, return_counts=True)
    summary = dict(zip(unique, counts))
    min_k = min(summary, key=(lambda x: summary[x]))
    print(np.where(array == min_k))