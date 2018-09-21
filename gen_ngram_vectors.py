# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import ngram
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
  with open(args.src_file_name) as src_file, open(args.dest_file_name, mode='wb') as dest_file:
    G = ngram.NGram(N=2)
    cnt = 0
    vecs = []
    line = src_file.readline().rstrip()
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
      vec = vec / np.linalg.norm(vec)
      vecs.append(vec)
      line = src_file.readline().rstrip()
      cnt = cnt + 1
      print(cnt)
      if cnt >= 10000:
        break
    array = np.array(vecs)
    np.save(dest_file, array)
