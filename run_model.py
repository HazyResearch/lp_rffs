import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import argparse
import sys, os
if sys.version_info[0] < 3:
    import cPickle as cp
else:
    import _pickle as cp
from copy import deepcopy
sys.path.append("./models")
sys.path.append("./kernels")
sys.path.append("./utils")
sys.path.append("./..")
from gaussian_exact import GaussianKernel
from rff import RFF
from circulant_rff import  CirculantRFF
from nystrom import Nystrom
from ensemble_nystrom import EnsembleNystrom
from quantizer import Quantizer
from logistic_regression import LogisticRegression
from ridge_regression import RidgeRegression
from kernel_regressor import KernelRidgeRegression
from data_loader import load_data
import halp
import halp.optim
import halp.quantize
from train_utils import train, evaluate, ProgressMonitor
from train_utils import get_sample_kernel_metrics, get_sample_kernel_F_norm, sample_data
# imports for fixed design runs
from misc_utils import expected_loss
from scipy.optimize import minimize

# EPS to prevent numerical issue in closed form ridge regression solver
EPS = 1e-10


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="logistic_regression")
parser.add_argument("--minibatch", type=int, default=64)
# parser.add_argument("--dataset", type=str, default="census")
parser.add_argument("--l2_reg", type=float, default=0.0)
parser.add_argument("--kernel_sigma", type=float, default=30.0)
parser.add_argument("--n_feat", type=int, default=32)
parser.add_argument("--random_seed", type=int, default=1)
parser.add_argument("--n_bit_feat", type=int, default=32)
parser.add_argument("--n_bit_model", type=int, default=32)
parser.add_argument("--scale_model", type=float, default=0.00001)
parser.add_argument("--do_fp_feat", action="store_true")
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--data_path", type=str, default="../data/census/")
parser.add_argument("--epoch", type=int, default=40)
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--opt", type=str, default="sgd")
parser.add_argument("--opt_mu", type=float, default=10.0)
parser.add_argument("--opt_epoch_T", type=float, default=1.0, 
    help="The # of epochs as interval between two consecutive scale updates/full gradient calculation")
parser.add_argument("--save_path", type=str, default="./test")
parser.add_argument("--approx_type", type=str, default="rff", help="specify using exact, rff or nystrom")
parser.add_argument("--collect_sample_metrics", action="store_true", 
    help="True if we want to collect metrics from the subsampled kernel matrix")
parser.add_argument("--n_sample", type=int, default=-1, 
    help="samples for metric measurements, including approximation error and etc.")
parser.add_argument("--fixed_design", action="store_true", 
    help="do fixed design experiment")
parser.add_argument("--fixed_design_noise_sigma", type=float, help="label noise std")
parser.add_argument("--fixed_design_auto_l2_reg", action="store_true",
    help="if true, we auto search for the optimal lambda")
parser.add_argument("--closed_form_sol", action="store_true", help="use closed form solution")
parser.add_argument("--fixed_epoch_number", action="store_true", help="if the flag is not used, use early stopping")
parser.add_argument("--exit_after_collect_metric", action="store_true", help="if the flag is used, \
    we only do metric collection on kernel matrix without doing trainining")
parser.add_argument("--n_ensemble_nystrom", type=int, default=1, help="number of learners in ensembled nystrom")
args = parser.parse_args()



if __name__ == "__main__":
    np.random.seed(args.random_seed)
    use_cuda = torch.cuda.is_available() and args.cuda
    torch.manual_seed(args.random_seed)
    if use_cuda:
        torch.cuda.manual_seed(args.random_seed)
        # torch.cuda.manual_seed_all(args.seed)
    # load dataset
    X_train, X_val, Y_train, Y_val = load_data(args.data_path)
    if args.fixed_design:
        print("fixed design using label noise sigma ", args.fixed_design_noise_sigma)
        Y_train_orig = Y_train.copy()
        X_val = X_train.copy()
        Y_val = Y_train.copy()
        Y_train += np.random.normal(scale=args.fixed_design_noise_sigma, size=Y_train.shape)
        Y_val += np.random.normal(scale=args.fixed_design_noise_sigma, size=Y_train.shape)

    if args.n_sample > 0:
        # downsample if specified
        X_train, Y_train = sample_data(X_train, Y_train, args.n_sample)
        X_val, Y_val = sample_data(X_val, Y_val, args.n_sample)
        assert X_train.shape[0] == Y_train.shape[0]
        assert X_val.shape[0] == Y_val.shape[0]
        print(X_train.shape[0], " training sample ", X_val.shape[0], "evaluation sample")

    X_train = torch.DoubleTensor(X_train)
    X_val = torch.DoubleTensor(X_val)
    if args.model == "ridge_regression":
        Y_train = torch.DoubleTensor(Y_train)        
        Y_val = torch.DoubleTensor(Y_val)
    elif args.model == "logistic_regression":
        Y_train = Y_train.reshape( (Y_train.size) )
        Y_val = Y_val.reshape( (Y_val.size) )
        n_class = np.unique(np.hstack( (Y_train, Y_val) ) ).size
        Y_train = torch.LongTensor(np.array(Y_train.tolist() ).reshape(Y_train.size, 1) )
        Y_val = torch.LongTensor(np.array(Y_val.tolist() ).reshape(Y_val.size, 1) )
    else:
        raise Exception("model not supported")

    # setup dataloader 
    train_data = \
        torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.minibatch, shuffle=False)
    val_data = \
        torch.utils.data.TensorDataset(X_val, Y_val)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.minibatch, shuffle=False)

    # setup gaussian kernel
    n_input_feat = X_train.shape[1]
    kernel = GaussianKernel(sigma=args.kernel_sigma)  
    if args.approx_type == "exact":
        print("exact kernel mode")
        # raise Exception("SGD based exact kernel is not implemented yet!")
        kernel_approx = kernel
        quantizer = None
    elif args.approx_type == "nystrom":
        print("fp nystrom mode")
        kernel_approx = Nystrom(args.n_feat, kernel=kernel, rand_seed=args.random_seed) 
        kernel_approx.setup(X_train) 
        quantizer = None
    elif args.approx_type == "ensemble_nystrom":
        print("ensembled nystrom mode with ", args.n_ensemble_nystrom, "learner")
        kernel_approx = EnsembleNystrom(args.n_feat, n_learner=args.n_ensemble_nystrom, kernel=kernel, rand_seed=args.random_seed)
        kernel_approx.setup(X_train)
        if args.do_fp_feat:
            quantizer = None
        else:
            # decide on the range of representation from training sample based features
            train_feat = kernel_approx.get_feat(X_train)
            min_val = torch.min(train_feat)
            max_val = torch.max(train_feat)
            quantizer = Quantizer(args.n_bit_feat, min_val, max_val, 
                rand_seed=args.random_seed, use_cuda=use_cuda)
            print("range for quantizing nystrom ensemble ", min_val, max_val)
            print("feature quantization scale, bit ", quantizer.scale, quantizer.nbit)
    elif args.approx_type == "rff":
        if args.do_fp_feat == False:
            print("lp rff feature mode")
            assert args.n_bit_feat >= 1
            n_quantized_rff = args.n_feat
            print("# feature ", n_quantized_rff)
            kernel_approx = RFF(n_quantized_rff, n_input_feat, kernel, rand_seed=args.random_seed)
            min_val = -np.sqrt(2.0/float(n_quantized_rff) )
            max_val = np.sqrt(2.0/float(n_quantized_rff) )
            quantizer = Quantizer(args.n_bit_feat, min_val, max_val, 
                rand_seed=args.random_seed, use_cuda=use_cuda)
            print("feature quantization scale, bit ", quantizer.scale, quantizer.nbit)
        elif args.do_fp_feat == True:
            print("fp rff feature mode")
            kernel_approx = RFF(args.n_feat, n_input_feat, kernel, rand_seed=args.random_seed)
            quantizer = None
    elif args.approx_type == "cir_rff":
        if args.do_fp_feat == False:
            print("lp circulant rff feature mode")
            assert args.n_bit_feat >= 1
            n_quantized_rff = args.n_feat
            print("# feature ", n_quantized_rff)
            kernel_approx = CirculantRFF(n_quantized_rff, n_input_feat, kernel, rand_seed=args.random_seed)
            min_val = -np.sqrt(2.0/float(n_quantized_rff) )
            max_val = np.sqrt(2.0/float(n_quantized_rff) )
            quantizer = Quantizer(args.n_bit_feat, min_val, max_val, 
                rand_seed=args.random_seed, use_cuda=use_cuda, for_lm_halp=( (args.opt == "lm_halp") or (args.opt == "lm_bit_center_sgd")  ) )
            print("feature quantization scale, bit ", quantizer.scale, quantizer.nbit)
        elif args.do_fp_feat == True:
            print("fp circulant rff feature mode")
            kernel_approx = CirculantRFF(args.n_feat, n_input_feat, kernel, rand_seed=args.random_seed)
            quantizer = None
    else:
        raise Exception("kernel approximation type not specified or not supported!")
    kernel.torch(cuda=use_cuda)
    kernel_approx.torch(cuda=use_cuda)

    if args.fixed_design or args.closed_form_sol:
        # for fixed design experiments and closed form solution form real setting
        if args.fixed_design_auto_l2_reg:
            # get kernel matrix and get the decomposition
            assert isinstance(X_train, torch.DoubleTensor)
            print("fixed design lambda calculation using kernel ", type(kernel_approx))
            kernel_mat = kernel_approx.get_kernel_matrix(X_train, X_train, quantizer, quantizer)
            assert isinstance(kernel_mat, torch.DoubleTensor)
            U, S, _ = np.linalg.svd(kernel_mat.cpu().numpy().astype(np.float64) )
            # numerically figure out the best lambda in the fixed design setting
            x0 = 1.0 
            f = lambda lam: expected_loss(lam,U,S,Y_train_orig,args.fixed_design_noise_sigma)
            res = minimize(f, x0, bounds=[(0.0, None)], options={'xtol': 1e-6, 'disp': True})
            loss = f(res.x)
            print("fixed design opt reg and loss", res.x, loss)
            args.l2_reg = max(res.x[0], EPS)
    else:
        # construct model
        if args.model == "logistic_regression":
            model = LogisticRegression(input_dim=kernel_approx.n_feat, 
                n_class=n_class, reg_lambda=args.l2_reg)
        elif args.model == "ridge_regression":
            model = RidgeRegression(input_dim=kernel_approx.n_feat, reg_lambda=args.l2_reg)
        if use_cuda:
            model.cuda() 
        model.double()   

        # set up optimizer
        if args.opt == "sgd":
            print("using sgd optimizer")
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_reg)
        elif args.opt == "lpsgd":
            print("using lp sgd optimizer")
            optimizer = halp.optim.LPSGD(model.parameters(), lr=args.learning_rate, 
                scale_factor=args.scale_model, bits=args.n_bit_model, weight_decay=args.l2_reg)
            print("model quantization scale and bit ", optimizer._scale_factor, optimizer._bits)
        elif args.opt == "halp":
            print("using halp optimizer")
            optimizer = halp.optim.HALP(model.parameters(), lr=args.learning_rate, 
                T=int(args.opt_epoch_T * X_train.size(0) / float(args.minibatch) ), 
                data_loader=train_loader, mu=args.opt_mu, bits=args.n_bit_model, weight_decay=args.l2_reg)
            print("model quantization, interval, mu, bit", optimizer.T, optimizer._mu, 
                optimizer._bits, optimizer._biased)
        elif args.opt == "lm_halp":
            print("using lm halp optimizer")
            optimizer = halp.optim.LMHALP(model.parameters(), lr=args.learning_rate, 
                T=int(args.opt_epoch_T * X_train.size(0) / float(args.minibatch) ), 
                data_loader=train_loader, mu=args.opt_mu, bits=args.n_bit_model, 
                weight_decay=args.l2_reg, data_scale=quantizer.scale)
            print("model quantization, interval, mu, bit", optimizer.T, optimizer._mu, 
                optimizer._bits, optimizer._biased)
        elif args.opt == "lm_bit_center_sgd":
            print("using lm bit center sgd optimizer")
            optimizer = halp.optim.BitCenterLMSGD(model.parameters(), lr=args.learning_rate,
                T=int(args.opt_epoch_T * X_train.size(0) / float(args.minibatch) ),
                data_loader=train_loader, mu=args.opt_mu, bits=args.n_bit_model,
                weight_decay=args.l2_reg, data_scale=quantizer.scale)
            print("model quantization, interval, mu, bit", optimizer.T, optimizer._mu,
                optimizer._bits, optimizer._biased)
        else:
            raise Exception("optimizer not supported")
    

    # collect metrics
    if args.collect_sample_metrics:
        print("start doing sample metric collection with ", X_train.size(0), " training samples")
        if use_cuda:
            metric_dict_sample_val, spectrum_sample_val, spectrum_sample_val_exact = \
                get_sample_kernel_metrics(X_val.cuda(), kernel, kernel_approx, quantizer, args.l2_reg)  
        else:
            metric_dict_sample_val, spectrum_sample_val, spectrum_sample_val_exact = \
                get_sample_kernel_metrics(X_val, kernel, kernel_approx, quantizer, args.l2_reg) 
        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)
        with open(args.save_path + "/metric_sample_eval.json", "wb") as f:
            cp.dump(metric_dict_sample_val, f)
        np.save(args.save_path + "/spectrum_eval.npy", spectrum_sample_val)
        np.save(args.save_path + "/spectrum_eval_exact.npy", spectrum_sample_val_exact)
        print("Sample metric collection done!")
        if args.exit_after_collect_metric:
            print("exit after collect metric")
            exit(0)

    if args.fixed_design or args.closed_form_sol:
        # for fixed design experiments and closed form solution form real setting
        if use_cuda:
            raise Exception("closed from solution does not support cuda mode")
        print("closed form using kernel type ", args.approx_type)
        regressor = KernelRidgeRegression(kernel_approx, reg_lambda=args.l2_reg)
        print("start to do regression!")
        # print("test quantizer", quantizer)
        regressor.fit(X_train, Y_train, quantizer=quantizer)
        print("finish regression!")
        train_error = regressor.get_train_error()
        pred = regressor.predict(X_val, quantizer_train=quantizer, quantizer_test=quantizer)
        test_error = regressor.get_test_error(Y_val)
        print("test error ", test_error)
        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)
        np.savetxt(args.save_path + "/train_loss.txt", np.array(train_error).reshape( (1, ) ) )
        np.savetxt(args.save_path + "/eval_metric.txt", np.array(test_error).reshape( (1, ) ) )
        np.savetxt(args.save_path + "/lambda.txt", np.array(args.l2_reg).reshape( (1, ) ) )
    else:
        # setup sgd training process
        train_loss = []
        eval_metric = []
        monitor_signal_history = []
        if args.model == "logistic_regression":
            monitor = ProgressMonitor(init_lr=args.learning_rate, lr_decay_fac=2.0, min_lr=0.00001, min_metric_better=True, decay_thresh=0.99)
        elif args.model == "ridge_regression":
            monitor = ProgressMonitor(init_lr=args.learning_rate, lr_decay_fac=2.0, min_lr=0.00001, min_metric_better=True, decay_thresh=0.99)
        else:
            raise Exception("model not supported!")
        for epoch in range(args.epoch):  
            # train for one epoch
            loss_per_step = train(args, model, epoch, train_loader, optimizer, quantizer, kernel_approx)
            train_loss += loss_per_step
            # evaluate and save evaluate metric
            metric, monitor_signal = evaluate(args, model, epoch, val_loader, quantizer, kernel_approx)
            eval_metric.append(metric)
            monitor_signal_history.append(monitor_signal)

            if not os.path.isdir(args.save_path):
                os.makedirs(args.save_path)
            np.savetxt(args.save_path + "/train_loss.txt", train_loss)
            np.savetxt(args.save_path + "/eval_metric.txt", eval_metric)
            np.savetxt(args.save_path + "/monitor_signal.txt", monitor_signal_history)
            if not args.fixed_epoch_number:
                print("using early stopping on lr")
                early_stop = monitor.end_of_epoch(monitor_signal, model, optimizer, epoch)
                if early_stop:
                    break
