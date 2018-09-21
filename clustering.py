# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from sklearn.cluster import KMeans
import numpy as np
import chars

def parseArguments():
  usage = 'Usage: python {} FILE'.format(__file__)
  argparser = ArgumentParser()
  argparser.add_argument('src_file_name', type=str, help='source file name')
  argparser.add_argument('dest_file_name', type=str, help='destination file name')
  args = argparser.parse_args()
  return args

if __name__ == '__main__':
  args = parseArguments()
  with open(args.src_file_name, mode='rb') as src_file, open(args.dest_file_name, mode='wb') as dest_file:
    array = np.load(src_file)
    print(array)
    pred = KMeans(n_clusters=10).fit_predict(array)
    print(pred)
    np.save(dest_file, pred)
