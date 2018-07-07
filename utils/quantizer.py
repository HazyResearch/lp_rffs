import numpy as np
import torch
import math

class Quantizer(object):
  def __init__(self, nbit, min_val, max_val, scale=None, rand_seed=1, use_cuda=False, for_lm_halp=False):
    self.nbit = nbit
    self.min_val = min_val
    self.max_val = max_val
    if scale == None:
      if for_lm_halp == False:
        self.scale = (max_val - min_val) / float(2**self.nbit - 1)
      else:
        # adapt to the halp quantization scheme where 0 is in the representation grid
        self.scale = (max_val - min_val) / float(2**self.nbit - 2)
    self.rand_seed = rand_seed
    self.use_cuda = use_cuda

  def quantize_random(self, value, verbose=True, fixed_seed=False):
    bound = math.pow(2.0, self.nbit) - 1
    min_val = 0.0
    max_val = bound
    if self.use_cuda:
      if fixed_seed:
        np.random.seed(self.rand_seed)
        adj_val = torch.cuda.FloatTensor(np.random.uniform(size=list(value.size() ) ) ).type(value.type() )
      else:
        adj_val = torch.cuda.FloatTensor(value.size()).type(value.type()).uniform_()
    else:
      if fixed_seed:
        np.random.seed(self.rand_seed)
        adj_val = torch.Tensor(np.random.uniform(size=list(value.size() ) ) ).type(value.type() )
      else:
        adj_val = torch.Tensor(value.size()).type(value.type()).uniform_()
    rounded = (value - self.min_val).div_(self.scale).add_(adj_val).floor_()
    clipped_value = rounded.clamp_(min_val, max_val)
    clipped_value *= self.scale 
    quant_val = clipped_value + self.min_val
    return quant_val

  def quantize_random_old(self, value, verbose=True):
    floor_val = self.min_val + torch.floor( (value - self.min_val) / self.scale) * self.scale
    ceil_val = self.min_val + torch.ceil( (value - self.min_val) / self.scale) * self.scale
    floor_prob = (ceil_val - value) / self.scale
    ceil_prob = (value - floor_val) / self.scale
    np.random.seed(self.rand_seed)
    sample = torch.DoubleTensor(np.random.uniform(size=list(value.size() ) ) )
    quant_val = floor_val * (sample < floor_prob).double() \
      + ceil_val * (sample >= floor_prob).double()
    return quant_val

  def quantize(self, value, verbose=True, fixed_seed=False):
    # TODO update if we have other quantization schemes
    value = torch.clamp(value, self.min_val, self.max_val)
    return self.quantize_random(value, verbose, fixed_seed)

  def quantize_old(self, value, verbose=True):
    # TODO update if we have other quantization schemes
    value = torch.clamp(value, self.min_val, self.max_val)
    return self.quantize_random_old(value, verbose)



def test_random_quantizer():
  quantizer = Quantizer(nbit=15, min_val=-2**14+1, max_val=2**14)

  # test lower bound
  lower = -2**14+1.0
  shift = 1/3.0
  value = np.ones( (1000, 1000) ) * (lower + shift)
  value = torch.DoubleTensor(value)
  quant_val = quantizer.quantize(value)
  quant_val = quant_val.cpu().numpy()
  assert np.unique(quant_val).size == 2
  assert np.min(np.unique(quant_val) ) == lower
  assert np.max(np.unique(quant_val) ) == lower + 1
  ratio = np.sum(quant_val == lower) / np.sum(quant_val == (lower + 1) ).astype(np.float)
  assert ratio > 1.95 and ratio < 2.05

  # test upper bound
  lower = 2**14-1.0
  shift = 2/3.0
  value = np.ones( (1000, 1000) ) * (lower + shift)
  value = torch.DoubleTensor(value)
  quant_val = quantizer.quantize(value)
  quant_val = quant_val.cpu().numpy()
  assert np.unique(quant_val).size == 2
  assert np.min(np.unique(quant_val) ) == lower
  assert np.max(np.unique(quant_val) ) == lower + 1
  ratio = np.sum(quant_val == lower) / np.sum(quant_val == (lower + 1) ).astype(np.float)
  assert ratio > 0.45 and ratio < 0.55

  # test middle values
  lower = 0.0
  shift = 0.5
  value = np.ones( (1000, 1000) ) * (lower + shift)
  value = torch.DoubleTensor(value)
  quant_val = quantizer.quantize(value)
  quant_val = quant_val.cpu().numpy()
  assert np.unique(quant_val).size == 2
  assert np.min(np.unique(quant_val) ) == lower
  assert np.max(np.unique(quant_val) ) == lower + 1
  ratio = np.sum(quant_val == lower) / np.sum(quant_val == (lower + 1) ).astype(np.float)
  assert ratio > 0.95 and ratio < 1.05

  print("quantizer test passed!")


def test_random_quantizer_fast_impl():
  # this only works when use numpy setted seed in new fast random quantize implementation
  quantizer = Quantizer(nbit=15, min_val=-2**14+1, max_val=2**14)
  # test middle values
  lower = 0.0
  shift = 0.5
  # value = np.ones( (1000, 1000) ) * (lower + shift)
  value = np.random.uniform((1000, 1000)) * 2**14
  value = torch.DoubleTensor(value)
  quant_val = quantizer.quantize(value)
  quant_val_old = quantizer.quantize_old(value)
  quant_val = quant_val.cpu().numpy()
  quant_val_old = quant_val_old.cpu().numpy()
  np.testing.assert_array_almost_equal(quant_val, quant_val_old, decimal=9)
  print("fast impl quantizer test passed!")


if __name__ == "__main__":
  test_random_quantizer_fast_impl()
  test_random_quantizer()