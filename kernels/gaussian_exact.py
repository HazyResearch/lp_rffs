import numpy as np
import torch
import sys

# class GaussianKernelSpec(object):
#   def __init__(self, sigma):
#     self.sigma = sigma

class GaussianKernel(object):
  def __init__(self, sigma):
    self.sigma = sigma
    self.dist_func = torch.nn.PairwiseDistance(p=2)
  
  def get_kernel_matrix(self, X1, X2, quantizer1=None, quantizer2=None, dtype="float"):
    '''
    the input value has shape [n_sample, n_dim]
    quantizer is dummy here
    dtype only works for numpy input for X1 X2
    '''
    if isinstance(X1, np.ndarray) and isinstance(X2, np.ndarray):
      n_sample_X1 = X1.shape[0]
      norms_X1 = np.linalg.norm(X1, axis=1).reshape(n_sample_X1, 1)
      n_sample_X2 = X2.shape[0]
      norms_X2 = np.linalg.norm(X2, axis=1).reshape(n_sample_X2, 1)
      cross = np.dot(X1, X2.T)
      # print("using sigma ", self.sigma)
      kernel = np.exp(-0.5 / float(self.sigma)**2 \
        * (np.tile(norms_X1**2, (1, n_sample_X2) ) + np.tile( (norms_X2.T)**2, (n_sample_X1, 1) ) \
        -2 * cross) )
      if dtype == "float":
          return torch.Tensor(kernel).float()
      else:
          return torch.Tensor(kernel).double()
    else:
      ## to prevent memory explosion on GPU, we can do the following operations on CPU and move results
      ## back to GPU
      #is_cuda_tensor = X1.is_cuda      
      #if is_cuda_tensor and use_cpu_comp:
      #    X1 = X1.cpu()
      #    X2 = X2.cpu()
      norms_X1 = (X1**2).sum(1).view(-1, 1)
      norms_X2 = (X2**2).sum(1).view(-1, 1)
      norms_X1 = norms_X1.repeat(1, int(X2.size(0) ) )
      norms_X2 = torch.transpose(norms_X2.repeat(1, int(X1.size(0) ) ), 0, 1)
      cross = torch.mm(X1, torch.transpose(X2, 0, 1) )
      kernel = torch.exp(-0.5 / float(self.sigma)**2 * (norms_X1 + norms_X2 - 2* cross) )
      #if is_cuda_tensor and use_cpu_comp:
      #    return kernel.cuda()
      #else:
      #    return kernel
      return kernel

  def torch(self, cuda=False):
    '''
    adapt the interface to the model launching wrapper
    '''
    pass
  
  def cpu(self):
    '''
    adapt the interface when switch parameter of some kernels back to cpu mode
    '''
    pass