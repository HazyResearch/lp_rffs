import torch
from torch.autograd import Variable
import numpy as np
from copy import deepcopy
import sys
sys.path.append("../utils")
from misc_utils import delta_approximation, eigenspace_overlap, set_random_seed


def train(args, model, epoch, train_loader, optimizer, quantizer, kernel):
    train_loss = []
    use_cuda = torch.cuda.is_available() and args.cuda
    # as we fix randomness for evaluation, in each new epoch, we want the randomness
    # on quantization for training to be different across epochs.
    if quantizer is not None:
        set_random_seed(quantizer.rand_seed + epoch)
    for i, minibatch in enumerate(train_loader):
        X, Y = minibatch
        if use_cuda:
            X = X.cuda()
            Y = Y.cuda()
        optimizer.zero_grad()
        if args.opt == "halp":
            # We need to add this function to models when we want to use SVRG
            def closure(data=X, target=Y):
                if use_cuda:
                    data = data.cuda()
                    target = target.cuda()
                if args.approx_type == "rff" or args.approx_type == "cir_rff":
                    data = kernel.get_cos_feat(data)
                elif args.approx_type == "nystrom":
                    data = kernel.get_feat(data)
                else:
                    raise Exception("kernel approximation type not supported!")
                if quantizer != None:
                    # print("halp use quantizer")
                    data = quantizer.quantize(data)
                if data.size(0) != target.size(0):
                    raise Exception("minibatch on data and target does not agree in closure")
                if not isinstance(data, torch.autograd.variable.Variable):
                    data = Variable(data, requires_grad=False)
                if not isinstance(target, torch.autograd.variable.Variable):
                    target = Variable(target, requires_grad=False)

                cost = model.forward(data, target)
                cost.backward()
                return cost
            loss = optimizer.step(closure)
            train_loss.append(float(loss[0].data.cpu().numpy()))
        elif (args.opt == "lm_halp_svrg") or (args.opt == "lm_halp_sgd"):
            # We need to add this function to models when we want to use SVRG
            def closure(data=X, target=Y, feat=None):
                if use_cuda:
                    data = data.cuda()
                    target = target.cuda()
                    if feat is not None:
                        feat = feat.cuda()
                if feat is None:
                    if args.approx_type == "rff" or args.approx_type == "cir_rff":
                        data = kernel.get_cos_feat(data)
                    elif args.approx_type == "nystrom":
                        data = kernel.get_feat(data)
                    else:
                        raise Exception("kernel approximation type not supported!")
                    if quantizer != None:
                        # print("halp use quantizer")
                        data = quantizer.quantize(data)
                    if data.size(0) != target.size(0):
                        raise Exception("minibatch on data and target does not agree in closure")
                    if not isinstance(data, torch.autograd.variable.Variable):
                        data = Variable(data, requires_grad=False)
                else:
                    # if we directly pass in the quantized feature, we directly use it without regeneration
                    # this is for the case of LM halp where we need to sync the quantization for prev and curr model.
                    data = feat

                if not isinstance(target, torch.autograd.variable.Variable):
                    target = Variable(target, requires_grad=False)

                cost = model.forward(data, target)
                model.output.retain_grad()
                cost.backward()
                # extract the data X and grad of the output of 
                return cost, data, model.output.grad
            loss = optimizer.step(closure)
            train_loss.append(float(loss[0].data.cpu().numpy()))
        else:
            if args.approx_type == "rff" or args.approx_type == "cir_rff":
                X = kernel.get_cos_feat(X)
            elif args.approx_type == "nystrom":
                X = kernel.get_feat(X)
            else:
                raise Exception("kernel approximation type not supported!")
            if quantizer != None:
                # print("train use quantizer")
                X = quantizer.quantize(X)
            X = Variable(X, requires_grad=False)
            Y = Variable(Y, requires_grad=False)
            loss = model.forward(X, Y)
            train_loss.append(float(loss[0].data.cpu().numpy()))
            loss.backward()
            optimizer.step()
        # print("epoch ", epoch, "step", i, "loss", loss[0] )
    return train_loss

def evaluate(args, model, epoch, val_loader, quantizer, kernel):
    # perform evaluation
    sample_cnt = 0
    use_cuda = torch.cuda.is_available() and args.cuda
    if args.model == "logistic_regression":
        correct_cnt = 0
        cross_entropy_accum = 0.0
        # we for the quantization random seed to be consistent with metric collection
        if quantizer is not None:
            set_random_seed(quantizer.rand_seed)
        for i, minibatch in enumerate(val_loader):
            X, Y = minibatch
            if use_cuda:
                X = X.cuda()
                Y = Y.cuda()
            if args.approx_type == "rff" or args.approx_type == "cir_rff":
                X = kernel.get_cos_feat(X)
            elif args.approx_type == "nystrom":
                X = kernel.get_feat(X)
            else:
                raise Exception("kernel approximation type not supported!")
            if quantizer != None:
                # print("test use quantizer")
                X = quantizer.quantize(X)
            X = Variable(X, requires_grad=False)
            Y = Variable(Y, requires_grad=False)
            pred, output = model.predict(X)
            correct_cnt += np.sum(pred.reshape(pred.size, 1) == Y.data.cpu().numpy() )
            if len(list(Y.size() ) ) == 2:
                Y = Y.squeeze()
            cross_entropy_accum += model.criterion(output, Y).data.cpu().numpy()[0]
            sample_cnt += pred.size
        print("eval_acc at epoch ", epoch, "step", i, " iterations ", " acc ", correct_cnt / float(sample_cnt), " cross entropy ", cross_entropy_accum / float(sample_cnt) )
        return correct_cnt / float(sample_cnt), cross_entropy_accum / float(sample_cnt)
    else:
        l2_accum = 0.0
        for i, minibatch in enumerate(val_loader):
            X, Y = minibatch
            if use_cuda:
                X = X.cuda()
                Y = Y.cuda()
            if args.approx_type == "rff" or args.approx_type == "cir_rff":
                X = kernel.get_cos_feat(X)
            elif args.approx_type == "nystrom":
                X = kernel.get_feat(X)
            else:
                raise Exception("kernel approximation type not supported!")
            if quantizer != None:
                # print("test use quantizer")
                X = quantizer.quantize(X)
            X = Variable(X, requires_grad=False)
            Y = Variable(Y, requires_grad=False)
            pred = model.predict(X)
            l2_accum += np.sum( (pred.reshape(pred.size, 1) \
                - Y.data.cpu().numpy().reshape(pred.size, 1) )**2)
            sample_cnt += pred.size
        print("eval_l2 at epoch ", epoch, "step", i, " iterations ", " loss ", np.sqrt(l2_accum / float(sample_cnt) ) )
        return l2_accum / float(sample_cnt), l2_accum / float(sample_cnt)


def sample_data(X, Y, n_sample):
    '''
    X is in the shape of [n_sample, n_feat]
    '''
    if isinstance(X, np.ndarray):
        # perm = np.random.permutation(np.arange(X.shape[0] ) )
        total_sample = X.shape[0]
        n_sample = min(n_sample, X.shape[0])
    else:
        total_sample = X.size(0)
        n_sample = min(n_sample, X.size(0) )
    if n_sample == total_sample:
        return X, Y
    else:
        perm = np.random.permutation(np.arange(total_sample) )
        X_sample = X[perm[:n_sample], :]
        Y_sample = Y[perm[:n_sample] ]
    return X_sample, Y_sample

def get_matrix_spectrum(X):
    # linalg.eigh can give negative value on cencus regression dataset
    # So we use svd here and we have not seen numerical issue yet.
    # currently only works for symetric matrix
    # when using torch mm for X1X1, it can produce slight different values in 
    # the upper and lower parts, but tested to be within tolerance using
    # np.testing.assert_array_almost_equal
    # if not torch.equal(X, torch.transpose(X, 0, 1) ):
    #     raise Exception("Kernel matrix is not symetric!")
    #S, U = np.linalg.eigh(X.cpu().numpy().astype(np.float64), UPLO='U')
    #if np.min(S) <= 0:
    #    print("numpy eigh gives negative values, switch to use SVD")
    U, S, _ = np.linalg.svd(X.cpu().numpy().astype(np.float64) )
    return S 

#####################################################################
# function to calculate Delta
#####################################################################
def get_sample_kernel_metrics(X, kernel, kernel_approx, quantizer, l2_reg, y_label=None):
    # X = sample_data(X_all, n_sample)
    is_cuda_tensor = X.is_cuda
    if is_cuda_tensor:
       kernel.cpu()
       kernel_approx.cpu()
       X = X.cpu()    
    kernel_mat = kernel.get_kernel_matrix(X, X)
    kernel_mat_approx = kernel_approx.get_kernel_matrix(X, X, quantizer, quantizer)
    # # need to use double for XXT if we want the torch equal to hold.
    # if not torch.equal(kernel_mat_approx, torch.transpose(kernel_mat_approx, 0, 1) ):
    #     raise Exception("Kernel matrix is not symetric!")
    error_matrix = kernel_mat_approx.cpu() - kernel_mat.cpu()
    F_norm_error = torch.sum(error_matrix**2)
    spectral_norm_error = np.max(np.abs(get_matrix_spectrum(error_matrix) ) )
    # spectrum = get_matrix_spectrum(kernel_mat_approx)
    # spectrum_exact = get_matrix_spectrum(kernel_mat)
    print("calculation delta with lambda = ", l2_reg)
    #delta_right, delta_left = 0.0, 0.0
    delta_right, delta_left = delta_approximation(kernel_mat.cpu().numpy().astype(np.float64), 
        kernel_mat_approx.cpu().numpy().astype(np.float64), l2_reg)
    # we also collect weighted overlap and the strength of labels on different eigen directions of the exact kernel
    print("computing overlap")
    overlap_list, weighted_overlap_list, smoothed_weighted_overlap_list, y_strength, y_strength_smooth = \
        eigenspace_overlap(kernel_mat.cpu().numpy().astype(np.float64), 
                           kernel_mat_approx.cpu().numpy().astype(np.float64), 
                           kernel_approx.n_feat, y_label=y_label.cpu().numpy().astype(np.float64))
    print("computing overlap finished.")
    spectrum = None
    spectrum_exact = None
    metric_dict = {"F_norm_error": float(F_norm_error),
                  "Delta_left": float(delta_left),
                  "Delta_right": float(delta_right),
                  "spectral_norm_error": float(spectral_norm_error) }
    for i, overlap in enumerate(overlap_list):
        metric_dict["overlap_{}".format(i)] = overlap_list[i]
    if weighted_overlap_list is not None:
        for i, weighted_overlap in enumerate(weighted_overlap_list):
            metric_dict["weighted_overlap_{}".format(i)] = weighted_overlap_list[i]
    if smoothed_weighted_overlap_list is not None:
        for i, smoothed_weighted_overlap in enumerate(smoothed_weighted_overlap_list):
            metric_dict["smoothed_weighted_overlap_{}".format(i)] = smoothed_weighted_overlap_list[i]
    if y_strength is not None:
        metric_dict["y_strength"] = y_strength
    if y_strength_smooth is not None:
        metric_dict['y_strength_smooth'] = y_strength_smooth

    print(metric_dict)
    if is_cuda_tensor:
       kernel.torch(cuda=True)
       kernel_approx.torch(cuda=True)
    print("sample metric collection done!")
    return metric_dict, spectrum, spectrum_exact


def get_sample_kernel_F_norm(X, kernel, kernel_approx, quantizer, l2_reg):
    is_cuda_tensor = X.is_cuda
    if is_cuda_tensor:
        kernel.cpu()
        kernel_approx.cpu()
        X = X.cpu()    
    kernel_mat = kernel.get_kernel_matrix(X, X)
    kernel_mat_approx = kernel_approx.get_kernel_matrix(X, X, quantizer, quantizer)
    # # need to use double for XXT if we want the torch equal to hold.
    # if not torch.equal(kernel_mat_approx, torch.transpose(kernel_mat_approx, 0, 1) ):
    #     raise Exception("Kernel matrix is not symetric!")
    error_matrix = kernel_mat_approx.cpu() - kernel_mat.cpu()
    F_norm_error = torch.sum(error_matrix**2)
    if is_cuda_tensor:
        kernel.torch(cuda=True)
        kernel_approx.torch(cuda=True)
    return float(F_norm_error) 



class ProgressMonitor(object):
    def __init__(self, init_lr=1.0, lr_decay_fac=2.0, min_lr=0.00001, min_metric_better=False, decay_thresh=0.99):
        self.lr = init_lr
        self.lr_decay_fac = lr_decay_fac
        self.min_lr = min_lr
        self.metric_history = []
        self.min_metric_better = min_metric_better
        self.best_model = None
        self.decay_thresh = decay_thresh
        self.prev_best = None
        self.drop_cnt = 0

    def end_of_epoch(self, metric, model, optimizer, epoch):
        if self.min_metric_better:
            model_is_better = (self.prev_best == None) or (metric <= self.prev_best)
        else:
            model_is_better = (self.prev_best == None) or (metric >= self.prev_best)

        if model_is_better:
            # save the best model
            self.best_model = deepcopy(model.state_dict() )
            print("saving best model with metric ", metric)
        else:
            # reverse to best model
            model.load_state_dict(deepcopy(self.best_model) )
            print("loading previous best model with metric ", self.prev_best)
        if (self.prev_best is not None) \
            and ( (self.min_metric_better and (metric > self.decay_thresh * self.prev_best) ) \
            or ( (not self.min_metric_better) and (metric < (1.0 + 1.0 - self.decay_thresh) * self.prev_best) ) ):
            self.lr /= self.lr_decay_fac
            for g in optimizer.param_groups:
                g['lr'] = self.lr
            print("lr drop to ", self.lr)
            self.drop_cnt += 1

        if model_is_better:
            self.prev_best = metric

        self.metric_history.append(metric)
        if self.drop_cnt == 10:
            return True
        else:
            return False
