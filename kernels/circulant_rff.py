import numpy as np
import torch
from gaussian_exact import GaussianKernel
from rff import RFF
from scipy.linalg import circulant
import math

class CirculantRFF(RFF):
  '''
  RFF using circulant random projection matrix
  '''
  def __init__(self, n_feat, n_input_feat, kernel=None, rand_seed=1):
    super(CirculantRFF, self).__init__(n_feat, n_input_feat, kernel, rand_seed)


  def get_gaussian_wb(self):
    self.w = np.zeros( (self.n_feat, self.n_input_feat) ) 
    if self.n_feat < self.n_input_feat:
        raise Exception("the dimension of projected features should be large than or equal to dimension of the raw features")
    np.random.seed(self.rand_seed)
    for i in range(int(math.ceil(self.n_feat / float(self.n_input_feat) ) ) ):
        generator = np.random.normal(scale=1.0/float(self.kernel.sigma), size=(self.n_input_feat,) )
        cir = circulant(generator)
        flip = np.diag(2 * np.random.randint(0, 2, size=(self.n_input_feat) ) - 1).astype(np.float64)
        row_start = i * self.n_input_feat
        row_end = min( (i + 1) * self.n_input_feat, self.n_feat)
        self.w[row_start:row_end, :] = np.dot(cir, flip)[:row_end - row_start,:]
    np.random.seed(self.rand_seed)
    self.b = np.random.uniform(low=0.0, high=2.0 * np.pi, size=(self.n_feat, 1) )


def test_circulant_rff():
    '''
    test if the circulant structure is observed in the circulant RFF object
    '''
    n_feat = 1000
    n_rff_feat = 1000
    seed = 2
    input_val  = torch.DoubleTensor(np.ones( [100, n_feat] ) )
    kernel = GaussianKernel(sigma=5.0)
    kernel_cir = CirculantRFF(n_rff_feat, n_feat, kernel=kernel, rand_seed=seed)
    kernel_basic = RFF(n_rff_feat, n_feat, kernel=kernel, rand_seed=seed)
    kernel_cir.torch()
    kernel_basic.torch()

    print("should see  column circulant structure", kernel_cir.w.cpu().numpy() )
    np.testing.assert_array_almost_equal(np.abs(kernel_cir.b.cpu().numpy() ), 
        np.abs(kernel_basic.b.cpu().numpy() ) )
    # np.testing.assert_array_almost_equal(np.abs(kernel_cir.w.cpu().numpy() ), 
    #     np.abs(kernel_basic.w.cpu().numpy() ) )
    print("should see similar row std between basic rff and circulant rff", 
        np.std(np.abs(kernel_cir.w.cpu().numpy() ), axis=1)[:10], 
        np.std(np.abs(kernel_basic.w.cpu().numpy() ), axis=1)[:10] )
    print("circulant rff test passed!")


if __name__ == "__main__":
    test_circulant_rff()


