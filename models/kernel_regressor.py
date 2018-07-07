import numpy as np
import torch
import sys
sys.path.append("../kernels/")
from gaussian_exact import GaussianKernel
from rff import RFF
from time import time
import math

class KernelRidgeRegression(object):
  def __init__(self, kernel, reg_lambda):
    '''
    reg_lambda is the strength of the regression regularizor
    kernel matrix is a Pytorch Tensor
    '''
    # self.kernel_mat = kernel_mat
    self.reg_lambda = reg_lambda
    self.kernel = kernel

  def fit(self, X_train=None, Y_train=None, kernel_mat=None, quantizer=None):
    self.X_train, self.Y_train = X_train, Y_train
    self.kernel_mat = self.kernel.get_kernel_matrix(X_train, X_train, quantizer, quantizer)
    n_sample = self.kernel_mat.size(0)
    # pytorch is super slow in inverse, so we finish this operation in numpy
    print("using regularior strength ", self.reg_lambda)
    self.alpha = torch.DoubleTensor( \
      np.dot(np.linalg.inv(self.kernel_mat.cpu().numpy().astype(np.float64) + self.reg_lambda * np.eye(n_sample) ), Y_train.cpu().numpy().astype(np.float64) ) )
    if self.kernel_mat.is_cuda:
      self.alpha = self.alpha.cuda()

  def torch(self, use_cuda):
    if use_cuda:
      self.alpha = self.alpha.cuda()

  def get_train_error(self):
    prediction = torch.mm(self.kernel_mat, self.alpha)
    if prediction.is_cuda:
      error = prediction - torch.cuda.DoubleTensor(self.Y_train)
    else:
      error = prediction - torch.DoubleTensor(self.Y_train)
    return torch.mean(error**2)

  def predict(self, X_test, quantizer_train=None, quantizer_test=None):
    # quantizer 1 for test data, quantizer 2 for train data
    self.X_test = X_test
    self.kernel_mat_pred = \
      self.kernel.get_kernel_matrix(self.X_test, self.X_train, quantizer_test, quantizer_train)
    self.prediction = torch.mm(self.kernel_mat_pred, self.alpha)
    return self.prediction.clone()

  def get_test_error(self, Y_test):
    # should only be called right after the predict function
    self.Y_test = Y_test
    if self.prediction.is_cuda:
      error = self.prediction - torch.cuda.DoubleTensor(self.Y_test)
    else:
      error = self.prediction - torch.DoubleTensor(self.Y_test)
    return torch.mean(error**2)


def test_kernel_ridge_regression1():
  '''
  We test the linear kernel case and gaussian kernel case
  '''
  n_feat = 10
  n_rff_feat = 1000000
  X_train  = np.ones( [2, n_feat] )
  X_train[0, :] *= 1
  X_train[0, :] *= 2
  Y_train = np.ones( [2, 1] )
  kernel = GaussianKernel(sigma=2.0)
  kernel = RFF(n_rff_feat, n_feat, kernel)
  use_cuda=torch.cuda.is_available()
  kernel.torch(cuda=torch.cuda.is_available())
  reg_lambda = 1.0
  regressor = KernelRidgeRegression(kernel, reg_lambda=reg_lambda)
  #regressor.torch(use_cuda=torch.cuda.is_available() )
  if use_cuda:
    regressor.fit(torch.DoubleTensor(X_train).cuda(), torch.DoubleTensor(Y_train).cuda() )
  else:
    regressor.fit(torch.DoubleTensor(X_train), torch.DoubleTensor(Y_train) )
  regressor.torch(use_cuda=torch.cuda.is_available() )
  # if test data equals traing data, it should the same L2 error
  X_test = np.copy(X_train)
  Y_test = np.copy(Y_train)
  if use_cuda:
    test_pred = regressor.predict(torch.DoubleTensor(X_test).cuda() )
  else:
    test_pred = regressor.predict(torch.DoubleTensor(X_test) )
  train_error = regressor.get_train_error()
  if use_cuda:
    test_error = regressor.get_test_error(torch.DoubleTensor(Y_test).cuda() )
  else:
    test_error = regressor.get_test_error(torch.DoubleTensor(Y_test) )
  assert np.abs(train_error - test_error) < 1e-6

  # if test data is different from traing data, L2 error for train and test should be different
  X_test = np.copy(X_train) * 2
  Y_test = np.copy(Y_train)
  if use_cuda:
    test_pred = regressor.predict(torch.cuda.DoubleTensor(X_test) )
  else:
    test_pred = regressor.predict(torch.DoubleTensor(X_test) )
  train_error = regressor.get_train_error()
  if use_cuda:
    test_error = regressor.get_test_error(torch.cuda.DoubleTensor(Y_test) )
  else:
    test_error = regressor.get_test_error(torch.DoubleTensor(Y_test) )
  assert np.abs(train_error - test_error) >= 1e-3

  X_test = np.copy(X_train)
  Y_test = np.copy(Y_train) * 2
  if use_cuda:
    test_pred = regressor.predict(torch.cuda.DoubleTensor(X_test) )
  else:
    test_pred = regressor.predict(torch.DoubleTensor(X_test) )
  train_error = regressor.get_train_error()
  if use_cuda:
    test_error = regressor.get_test_error(torch.cuda.DoubleTensor(Y_test) )
  else:
    test_error = regressor.get_test_error(torch.DoubleTensor(Y_test) )
  assert np.abs(train_error - test_error) >= 1e-3

  print("kernel ridge regression test1 passed!")


def test_kernel_ridge_regression2():
  '''
  We test the linear kernel case and gaussian kernel case
  '''
  n_feat = 10
  n_rff_feat = 1000
  X_train  = np.ones( [2, n_feat] )
  X_train[0, :] *= 1
  X_train[0, :] *= 2
  Y_train = np.ones( [2, 1] )
  kernel = GaussianKernel(sigma=2.0)
  kernel = RFF(n_rff_feat, n_feat, kernel)
  use_cuda = torch.cuda.is_available() 
  kernel.torch(cuda=use_cuda)
  reg_lambda = 1.0
  regressor = KernelRidgeRegression(kernel, reg_lambda=reg_lambda)
  if use_cuda:
    regressor.fit(torch.cuda.DoubleTensor(X_train), torch.cuda.DoubleTensor(Y_train) )
  else:
    regressor.fit(torch.DoubleTensor(X_train), torch.DoubleTensor(Y_train) )
  # compare the two ways of calculating feature weights as sanity check
  # feature weight using the approach inside KernelRidgeRegression
  if use_cuda:
    kernel.get_kernel_matrix(torch.cuda.DoubleTensor(X_train), torch.cuda.DoubleTensor(X_train) )
  else:
    kernel.get_kernel_matrix(torch.DoubleTensor(X_train), torch.DoubleTensor(X_train) )
  # print kernel.rff_x2.size(), regressor.alpha.size()
  w1 = torch.mm(torch.transpose(kernel.rff_x2, 0, 1), regressor.alpha)
  # print w1.size()
  # feature weight using alternative way of calculation
  if use_cuda:
    val = torch.inverse( (regressor.reg_lambda * torch.eye(n_rff_feat).double().cuda() \
      + torch.mm(torch.transpose(kernel.rff_x1, 0, 1), kernel.rff_x1) ) )
  else:
    val = torch.inverse( (regressor.reg_lambda * torch.eye(n_rff_feat).double() \
      + torch.mm(torch.transpose(kernel.rff_x1, 0, 1), kernel.rff_x1) ) )
  val = torch.mm(val, torch.transpose(kernel.rff_x2, 0, 1) )
  if use_cuda:
    w2 = torch.mm(val, torch.cuda.DoubleTensor(Y_train) )  
  else:
    w2 = torch.mm(val, torch.DoubleTensor(Y_train) )
  np.testing.assert_array_almost_equal(w1.cpu().numpy(), w2.cpu().numpy() )
  # print(w1.cpu().numpy().ravel()[-10:-1], w2.cpu().numpy().ravel()[-10:-1] )
  print("kernel ridge regression test2 passed!")


if __name__ == "__main__":
  test_kernel_ridge_regression1()
  test_kernel_ridge_regression2()



