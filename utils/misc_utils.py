import numpy as np
from scipy.optimize import minimize
import torch
from sklearn.isotonic import IsotonicRegression

# for numerical protection
EPS = 1e-20

def set_random_seed(seed):
    np.random.seed(seed)
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def expected_loss(lam, U, S, Y, noise):
    m = float(Y.size)
    uty2 = ( (np.dot(U.T, Y.reshape(Y.size, 1) ) )**2).reshape(int(m))
    gamma = (S/(S + lam + EPS) ).reshape(int(m))
    return (1/m) * np.sum(((1.0-gamma)**2) * uty2) + (1/m)*noise**2 * np.sum(gamma**2) + noise**2

# symmetric delta
def delta_approximation(K, K_tilde, lambda_=1e-3):
    """ Compute the smallest D such that (1 + D)^(-1)(K + lambda_ I) <= K_tilde + lambda_ I <= (1 + D)(K + lambda_ I),
    where the inequalities are in semidefinite order. This is a symetric version of delta_approximation
    """
    n, m = K.shape
    n_tilde, m_tilde = K_tilde.shape
    assert n == m and n_tilde == m_tilde, "Kernel matrix must be square"
    assert n == n_tilde, "K and K_tilde must have the same shape"
    assert np.allclose(K, K.T) and np.allclose(K_tilde, K_tilde.T), "Kernel matrix must be symmetric"
    # Compute eigen-decomposition of K + lambda_ I, of the form V @ np.diag(sigma) @ V.T
    sigma, V = np.linalg.eigh(K)
    #assert np.all(sigma >= 0), "Kernel matrix K must be positive semidefinite"
    sigma += lambda_
    # Whitened K_tilde: np.diag(1 / np.sqrt(sigma)) @ V.T @ K_tilde @ V @ np.diag(1 / np.sqrt(sigma))
    K_tilde_whitened = V.T.dot(K_tilde.dot(V)) / np.sqrt(sigma) / np.sqrt(sigma)[:, np.newaxis]
    K_whitened = np.diag(1 - lambda_ / sigma)
    sigma_final, _ = np.linalg.eigh(K_tilde_whitened - K_whitened)
    lambda_min = sigma_final[0]
    lambda_max = sigma_final[-1]
    assert lambda_max >= lambda_min
    return lambda_max, -lambda_min

# get eigenspace overlap 
def eigenspace_overlap(K, K_tilde, K_tilde_feat_dim, ref_dim_list=None, y_label=None, power_alpha=[1,2,3,4,5,6,7]):
    if ref_dim_list is None and y_label is not None:
        ref_dim_list = [1,2,4,8,16,32,64,128,256,512,1024,2000]
    elif ref_dim_list is None:
        ref_dim_list = [1,2,4,8,16,32,64,128,256,512,1000,2500,5000,10000,20000]
    # here K is the exact kernel while K_tilde is the approximated kernel
    # K_tilde_feat_dim specifies
    # ref_dim_list specifies the number of left singular vectors we consider for kernel K and K-tilde
    assert np.allclose(K, K.T) and np.allclose(K_tilde, K_tilde.T), "Kernel matrix must be symmetric"    
    sigma, U = np.linalg.eigh(K)
    sigma_tilde, U_tilde = np.linalg.eigh(K_tilde)
    # reverse the column order to pop top eigen vectors to the front
    U = U[:, ::-1]
    U_tilde = U_tilde[:, ::-1]
    overlap_list = []
    eig_weighted_overlap_dict = {}

    for ref_dim in ref_dim_list:
        overlap = np.linalg.norm(U_tilde[:, :int(K_tilde_feat_dim)].T @ U[:, :int(ref_dim)])**2 / float(ref_dim)
        overlap_list.append(overlap)
    for power_alpha in [1,2,3,4,5,6,7]:
        eig_weighted_overlap_dict[power_alpha] = []
        for ref_dim in ref_dim_list:
            eig_weighted_overlap = np.linalg.norm(U_tilde[:, :int(K_tilde_feat_dim)].T @ U[:, :int(ref_dim)] @ np.diag(sigma[:int(ref_dim)]))**2 / float(ref_dim)
            eig_weighted_overlap_dict[power_alpha].append(eig_weighted_overlap)
    # print(U)
    # print(U_tilde)
    if y_label is not None:
        # we record the strength of label vector on different eigen vector directions 
        strength = np.abs(U.T @ y_label.reshape((y_label.size, 1)))
        strength = strength.reshape((strength.size, ))
        ir = IsotonicRegression(increasing=False)
        smoothed_strength = ir.fit_transform(np.arange(strength.size), strength)
        # we also collect the weighted overlap values
        weighted_overlap_list = []
        for ref_dim in ref_dim_list:
            weighted_overlap = np.linalg.norm(U_tilde[:, :int(K_tilde_feat_dim)].T @ U[:, :int(ref_dim)] @ np.diag(strength[:int(ref_dim)]))**2 / float(ref_dim)
            weighted_overlap_list.append(weighted_overlap)
        smoothed_weighted_overlap_list = []
        for ref_dim in ref_dim_list:
            smoothed_weighted_overlap = np.linalg.norm(U_tilde[:, :int(K_tilde_feat_dim)].T @ U[:, :int(ref_dim)] @ np.diag(smoothed_strength[:int(ref_dim)]))**2 / float(ref_dim)
            smoothed_weighted_overlap_list.append(smoothed_weighted_overlap)
    else:
        strength = None
        weighted_overlap_list = None
        smoothed_weighted_overlap_list = None
    return overlap_list, weighted_overlap_list, smoothed_weighted_overlap_list, strength.tolist(), smoothed_strength.tolist(), eig_weighted_overlap_dict


class Args(object):
    def __init__(self, n_fp_rff, n_bit, 
                 exact_kernel, reg_lambda, 
                 sigma, random_seed, data_path,
                 do_fp, test_var_reduce=False):
        self.n_fp_rff = n_fp_rff
        self.n_bit = n_bit
        self.exact_kernel = exact_kernel
        self.reg_lambda = reg_lambda
        self.sigma = sigma
        self.random_seed = random_seed
        self.data_path = data_path
        self.do_fp = do_fp
        self.test_var_reduce = test_var_reduce
