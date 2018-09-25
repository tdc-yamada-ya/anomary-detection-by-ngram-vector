# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import sparse
from sklearn import cluster
from sklearn.decomposition import PCA

def parse_arguments():
  usage = 'Usage: python {} FILE'.format(__file__)
  argparser = ArgumentParser()
  argparser.add_argument('vectors_dirname', type=str, help='vectors directory name')
  argparser.add_argument('predictions_dirname', type=str, help='predictions directory name')
  args = argparser.parse_args()
  return args

def find_files(vectors_dirname, predictions_dirname):
  cnt = 0
  
  while True:
    vector_filename = os.path.join(vectors_dirname, 'vec{:05d}.npz'.format(cnt))
    prediction_filename = os.path.join(predictions_dirname, 'pred{:05d}.npy'.format(cnt))

    if not os.path.isfile(vector_filename) or not os.path.isfile(prediction_filename):
      break

    yield (vector_filename, prediction_filename)
    cnt = cnt + 1

if __name__ == '__main__':
  args = parse_arguments()
  vectors_dirname = os.path.abspath(args.vectors_dirname)
  predictions_dirname = os.path.abspath(args.predictions_dirname)

  print('Fit pass')
  pca = PCA(n_components=2)

  for (vector_filename, prediction_filename) in find_files(vectors_dirname, predictions_dirname):
    print('vector: {}'.format(vector_filename))
    print('prediction: {}'.format(prediction_filename))
    vec = sparse.load_npz(vector_filename)
    vec = vec.toarray()
    pca.fit(vec)

  print('Plot pass')

  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  ax.set_title('ngram vectors')
  ax.set_xlabel('x')
  ax.set_ylabel('y')

  for (vector_filename, prediction_filename) in find_files(vectors_dirname, predictions_dirname):
    print('vector: {}'.format(vector_filename))
    print('prediction: {}'.format(prediction_filename))
    vec = sparse.load_npz(vector_filename)
    vec = vec.toarray()
    vec = pca.transform(vec).transpose()
    pred = np.load(prediction_filename).tolist()
    #pred = list(map(lambda n: n if n == 6 else 0, pred))
    ax.scatter(vec[0], vec[1], s=1, c=pred)

  plt.show()
