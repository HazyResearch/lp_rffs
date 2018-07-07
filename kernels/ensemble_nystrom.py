import numpy as np
import scipy
import torch
from nystrom import Nystrom
from gaussian_exact import GaussianKernel
import sys
sys.path.append("../utils")
from misc_utils import set_random_seed
from quantizer import Quantizer
import math

EPS = 1e-15

class EnsembleNystrom(object):
    def __init__(self, n_feat, n_learner, kernel=None, rand_seed=1):
        self.n_feat_per_learner = n_feat // n_learner
        self.n_learner = n_learner
        self.kernel = kernel
        self.rand_seed = rand_seed
        self.n_feat = n_feat

    def setup(self, X, n_landmark=None):
        '''
        X is in the shape of [n_sample, n_dimension]
        call setup() once before using Nystrom
        '''
        if self.n_feat > X.size(0):
            self.n_feat = X.size(0)
            self.n_feat_per_learner = self.n_feat // self.n_learner

        self.learners = []
        np.random.seed(self.rand_seed)
        perm = np.random.permutation(np.arange(X.size(0) ) )
        # perm = np.arange(X.size(0) )
        for i in range(self.n_learner):
            self.learners.append(
                Nystrom(self.n_feat_per_learner, self.kernel, self.rand_seed) )
            start_idx = i * self.n_feat_per_learner
            end_idx = min( (i + 1) * self.n_feat_per_learner, X.size(0) )
            self.learners[-1].setup(X[perm[start_idx:end_idx], :] )

    def get_feat(self, X):
        feat_list = []
        for learner in self.learners:
            feat_list.append(learner.get_feat(X) )
        feat = torch.cat(feat_list, dim=1) / math.sqrt(float(len(self.learners) ) ) 
        print("normalizing features with ", math.sqrt(float(len(self.learners) ) ) )
        assert feat.size(1) == self.n_feat_per_learner * self.n_learner
        return feat

    def get_kernel_matrix(self, X1, X2, quantizer1=None, quantizer2=None, consistent_quant_seed=True):
        feat_x1 = self.get_feat(X1)
        feat_x2 = self.get_feat(X2)
        # quantization
        if consistent_quant_seed and (quantizer1 is not None) and (quantizer2 is not None):
            assert quantizer1.rand_seed == quantizer2.rand_seed, "quantizer random seed are different under consistent quant seed mode!"
        if quantizer1 != None:
          if consistent_quant_seed and list(feat_x1.size() ) == list(feat_x2.size() ):
            print("quantizing rff_x1 with random seed", quantizer1.rand_seed)
            set_random_seed(quantizer1.rand_seed)
          else:
            print("quantizing rff_x1 without fixed random seed")
          # print("quantization 1 activated ", X1.shape)
          # print("quantizer 1 bits", quantizer1.nbit)
          # print("quantizer 1 scale", quantizer1.scale)
          feat_x1 = quantizer1.quantize(feat_x1)
        if quantizer2 != None:
          if consistent_quant_seed:
            print("quantizing rff_x2 with random seed", quantizer2.rand_seed)
            set_random_seed(quantizer2.rand_seed)
          # print("quantization 2 activated ", X2.shape)
          # print("quantizer 2 bits", quantizer2.nbit)
          # print("quantizer 2 scale", quantizer2.scale)
          feat_x2 = quantizer2.quantize(feat_x2)

        if consistent_quant_seed and list(feat_x1.size() ) == list(feat_x2.size() ):
            np.testing.assert_array_almost_equal(feat_x1.cpu().numpy(), feat_x2.cpu().numpy() )

        return torch.mm(feat_x1, torch.transpose(feat_x2, 0, 1) ) 

    def torch(self, cuda):
        for learner in self.learners:
            learner.torch(cuda)

    def cpu(self):
        for learner in self.learners:
            learner.cpu()  


def test_ensemble_nystrom_full_prec_one_learner():
    # test if keep all the dimensions is the nystrom kernel matrix equals to the exact kernel
    n_sample = 150
    n_feat = n_sample
    input_val1  = torch.DoubleTensor(np.random.normal(size=[n_sample, n_feat] ) ).double()
    input_val2 = input_val1
    # input_val2  = torch.DoubleTensor(np.random.normal(size=[n_sample - 1, n_feat] ) ).double()
    # get exact gaussian kernel
    kernel = GaussianKernel(sigma=10.0)
    kernel_mat = kernel.get_kernel_matrix(input_val1, input_val2)

    # nystrom method
    approx = Nystrom(n_feat, kernel=kernel)
    approx.setup(input_val1)
    feat = approx.get_feat(input_val1)
    approx_kernel_mat = approx.get_kernel_matrix(input_val1, input_val2)

    # ensembleed nystrom method
    approx_ensemble = EnsembleNystrom(n_feat, n_learner=1, kernel=kernel)
    approx_ensemble.setup(input_val1)
    feat_ensemble = approx_ensemble.get_feat(input_val1)
    approx_kernel_mat_ensemble = approx_ensemble.get_kernel_matrix(input_val1, input_val2)
    np.testing.assert_array_almost_equal(np.sum(feat.cpu().numpy()**2), 
        np.sum(feat_ensemble.cpu().numpy()**2) )

    np.testing.assert_array_almost_equal(np.sum(approx_kernel_mat.cpu().numpy()**2), 
        np.sum(approx_kernel_mat_ensemble.cpu().numpy()**2) )
    print("single learner ensembled nystrom test passed!")


def test_ensemble_nystrom_full_prec_three_learner():
    # test if keep all the dimensions is the nystrom kernel matrix equals to the exact kernel
    n_sample = 150
    n_feat = n_sample
    input_val1  = torch.DoubleTensor(np.random.normal(size=[n_sample, n_feat] ) ).double()
    input_val2 = input_val1
    # input_val2  = torch.DoubleTensor(np.random.normal(size=[n_sample - 1, n_feat] ) ).double()
    # get exact gaussian kernel
    kernel = GaussianKernel(sigma=10.0)
    kernel_mat = kernel.get_kernel_matrix(input_val1, input_val2)

    # nystrom method
    approx = Nystrom(n_feat, kernel=kernel)
    approx.setup(input_val1)
    feat = approx.get_feat(input_val1)
    approx_kernel_mat = approx.get_kernel_matrix(input_val1, input_val2)

    # ensembleed nystrom method
    approx_ensemble = EnsembleNystrom(n_feat, n_learner=3, kernel=kernel)
    approx_ensemble.setup(input_val1)
    feat_ensemble = approx_ensemble.get_feat(input_val1)
    assert feat_ensemble.size(0) == n_sample
    assert feat_ensemble.size(1) == n_feat 
    approx_kernel_mat_ensemble = approx_ensemble.get_kernel_matrix(input_val1, input_val2)
    print("single learner ensembled nystrom test passed!")


def test_ensemble_nystrom_low_prec():
    # test if keep all the dimensions is the nystrom kernel matrix equals to the exact kernel
    n_sample = 150
    n_feat = n_sample
    input_val1  = torch.DoubleTensor(np.random.normal(size=[n_sample, n_feat] ) ).double()
    input_val2 = input_val1
    # input_val2  = torch.DoubleTensor(np.random.normal(size=[n_sample - 1, n_feat] ) ).double()
    # get exact gaussian kernel
    kernel = GaussianKernel(sigma=10.0)
    kernel_mat = kernel.get_kernel_matrix(input_val1, input_val2)

    # setup quantizer
    quantizer = Quantizer(4, torch.min(input_val1), torch.max(input_val1), rand_seed=2, use_cuda=False)

    # nystrom method
    approx = Nystrom(n_feat, kernel=kernel)
    approx.setup(input_val1)
    feat = approx.get_feat(input_val1)
    approx_kernel_mat = approx.get_kernel_matrix(input_val1, input_val2, quantizer, quantizer)


    # ensembleed nystrom method
    approx_ensemble = EnsembleNystrom(n_feat, n_learner=1, kernel=kernel)
    approx_ensemble.setup(input_val1)
    feat_ensemble = approx_ensemble.get_feat(input_val1)
    approx_kernel_mat_ensemble = approx_ensemble.get_kernel_matrix(input_val1, input_val2, 
        quantizer, quantizer, consistent_quant_seed=True)
    approx_kernel_mat_ensemble = approx_ensemble.get_kernel_matrix(input_val1, input_val2, 
        quantizer, quantizer, consistent_quant_seed=True)

    print("single learner ensembled nystrom quantizerd version test passed!")


if __name__ == "__main__":
    test_ensemble_nystrom_full_prec_one_learner()
    test_ensemble_nystrom_full_prec_three_learner()
    test_ensemble_nystrom_low_prec()

