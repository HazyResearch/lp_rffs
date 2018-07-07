import numpy as np
import scipy.io as sio
import h5py

def load_data(path="../../data/census/census"):
  try:
    X_train = sio.loadmat(path + "_train_feat.mat")
    Y_train = sio.loadmat(path + "_train_lab.mat")
    X_test = sio.loadmat(path + "_heldout_feat.mat")
    Y_test = sio.loadmat(path + "_heldout_lab.mat")
  except:
    print("switch to use h5py to load files")
    X_train = h5py.File(path + "_train_feat.mat", 'r')
    Y_train = sio.loadmat(path + "_train_lab.mat")
    X_test = sio.loadmat(path + "_heldout_feat.mat")
    Y_test = sio.loadmat(path + "_heldout_lab.mat")

  if 'X_ho' in X_test.keys():
    X_test = X_test['X_ho']
  else:
    X_test = X_test["fea"]
  if "X_tr" in X_train.keys():
    X_train = X_train['X_tr']
  else:
    X_train = X_train['fea']
  if "Y_ho" in Y_test.keys():
    Y_test = Y_test['Y_ho']
  else:
    Y_test = Y_test['lab']
  if "Y_tr" in Y_train.keys():
    Y_train = Y_train['Y_tr']
  else:
    Y_train = Y_train['lab']

  if X_train.shape[0] != Y_train.size:
    X_train = np.array(X_train).T
  if X_test.shape[0] != Y_test.size:
    X_test = X_test.T

  # # # DEBUG
  # s = np.arange(X_train.shape[0] )
  # np.random.seed(0)
  # np.random.shuffle(s)
  # X_train = X_train[s, :]
  # Y_train = Y_train[s]
  # X_train, Y_train, X_test, Y_test = \
  # X_train[:int(s.size * 1 / 5), :], Y_train[:int(s.size * 1 / 5)], X_test[:int(s.size * 1 / 5), :], Y_test[:int(s.size * 1 / 5)]
  # print("test ", X_train.shape, Y_train.shape)
  assert X_train.shape[0] == Y_train.shape[0]
  assert X_test.shape[0] == Y_test.shape[0]
  assert X_train.shape[0] != X_test.shape[0]
  return X_train, X_test, Y_train, Y_test
