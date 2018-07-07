# Low Precision Random Fourier Features (LP-RFFs)

**LP-RFFs is a library to train classification and regression models using fixed-point low-precision random Fourier features.** Random Fourier feature (RFFs) is one of the major kernel approximation approach for kernel methods on large scale datasets. The generalization performance of kernel methods using RFFs is highly correlated with the number of RFFs; larger number of RFFs typically indicates better generalizaton, yet requires significantly larger memory footprint in minibatch-based training. **LP-RFFs use low-precision fixed-point representation for the kernel approximation features. It can achieve similar performance as with full-precision RFFs using 5-10X less memory during training with theoretical guarantees.** LP-RFFs currently supports closed-form kernel ridge regression, SGD-based training for kernel ridge regression and kernel logistic regression in float-point-based simulation. LP-RFFs also support low-precision training with LM-HALP and LM-Bit-Center-SGD using the implementation from the [HALP repo](https://github.com/mleszczy/halp). For more technical details, please refer to our paper [Low-Precision Random Fourier Features]().

## Content
* [Setup instructions](#setup-instructions)
* [Command guidelines](#command-guidelines)
* [Citation](#citation)
* [Acknowledgement](#acknowledgement)

## Setup instructions
* Install PyTorch. Our implementation is tested under PyTorch 0.3.1.
* Install h5py by ```pip install h5py```
* Clone the LP-RFFs repo, along with the HALP repo (duplicated and adapted from the repo of the original authors) into **the same folder**.
```
git clone https://github.com/JianGoForIt/lp_rffs.git
git clone https://github.com/JianGoForIt/halp.git
```
* Download data from dropbox in the same folder with the cloned LP-RFFs and HALP repo. We provide preprocessed training and heldout dataset in our paper, including the Census, CovType and YearPred datasets. For the TIMIT dataset, we do not provide it here due to licensing restriction. We refer to the our [paper]() for details in preprocessing the raw TIMIT dataset.
```
wget https://www.dropbox.com/s/l1jy7ilifrknd82/LP-RFFs-Data.zip
unzip LP-RFFs-Data.zip
```

## Command guidelines

* **Key arguments**

  * specify kernel approximation method
  ```
  --approx_type: a string specifying the kernel approximation method.
  --n_feat: a positive integer specifying the number of kernel approximation features.
  --do_fp_feat: a flag to use full precision kernel approximation features.
  --n_bit_feat: a positive integer specifying the number of bits for low precision fixed-point representation of kernel approximation features.

  LP-RFFs currently support:
  * FP-RFFs (--approx_type=rff --do_fp_feat)
  * circulant FP-RFFs (--approx_type=cir_rff --do_fp_feat)
  * FP-Nystrom (--approx_type=nystrom --do_fp_feat)
  * ensemble FP-Nystrom (--approx_type=ensemble_nystrom --do_fp_feat --n_ensemble_nystrom=<# of learners of ensemble Nystrom>)
  * LP-RFFs (--approx_type=cir_rff --n_bit_feat=<# of bits>)
  * LP-Nystrom (--approx_type=ensemble_nystrom --n_bit_feat=<# of bits> --n_ensemble_nystrom=1)
  * ensemble LP-Nystrom (--approx_type=ensemble_nystrom --n_bit_feat=<# of bits> --n_ensemble_nystrom=<# of learners of ensemble Nystrom>).
  ```
  
  * specify training approach
  ```
  LP-RFFs currently support training the following models:
  * closed-form kernel ridge regression: 
    --model=ridge_regression --closed_form_sol 
  * mini-batch based iterative training for kernel ridge regression: 
    --model=ridge_regression --opt=type of the optimizer
  * mini-batch based iterative training for logistic regression: 
    --model=logistic_regression --opt=type of the optimizer
    
  LP-RFFs can use the following optimizers for mini-batch based iterative training:
  * plain SGD (full precision training):
    --opt=sgd
  * LM-HALP (low precision training):
    --opt=lm_halp --n_bit_model=<# of bit for model parameter during training> \
    --opt_mu=<the value determine the scale factor in LM-HALP> \
    --opt_epoch_T=<for --opt_epoch_T=t, the scale factor in LM-HALP will be updated every t epochs>
  * LM-Bit-Center SGD (low precision training):
    --opt=lm_bit_center_sgd --n_bit_model=<# of bit for model parameter during training> \
    --opt_mu=<the value do determine the scale factor in LM-Bit-Center SGD> \
    --opt_epoch_T=<# of epochs as interval to compute the scale factor in LM-Bit-Center SGD>
    
  The learning rate and minibatch size can be specified using --learning_rate and --minibatch.
  
  For GPU based iterative training, please use --cuda. 
  
  By default, we use the early stopping protocol described in the Appendix C.1 of our arxiv paper. 
  Early stopping can be turned off by --fixed_epoch_number. 
  The maximal possible training epochs can be specified by --epoch for both training with and without early stopping.
  ```

  * --collect_sample_metrics indicates to calculate relative spectral distance, Frobenius norm error, spectral norm error on the heldout set kernel matrix. *The computation of relative spectral distance can be time-consuming for a large heldout set, e.g. it can take up to ~10 minutes for 20k data points on a high-performance CPU server.* For large datasests, these metrics can be computed on a subsampled heldout set, the size of the subsampled heldout set can be specified by --n_sample=size of subsampled heldout set.
  
  * The dataset path and the output saving path can be specified with --data_path and --save_path.
  
  * The l2 regularization strength can be specified by --l2_reg.
  
  * The \sigma value for the underlying Gaussian kernel \exp(1/(2*\sigma^2) ||x_1 - x_2||^2 ) ) can be specified via --kernel_sigma.

* **Outputs**
  * The output folder is specified by --save_path 

  * The heldout performance (single value for closed-solution, or history (for every epoch) for mini-batch based training) is saved into ```eval_metric.txt```.

  * If --collect_sample_metric is specified, the collected metric (e.g. relative spectral distance, Frobenius norm kernel approximation error) are saved into ```metric_sample_eval.json```.
  
* **Example runs**
  
  We present the command for a couple of configurations. For more examples please browse to [example_runs.sh](./example_runs.sh)
  
  * closed-form kernel ridge regression using 4 bit LP-RFFs on CPU
  ```
  python run_model.py \
    --approx_type=cir_rff --n_feat=1000  --n_bit_feat=4  \
    --model=ridge_regression --closed_form_sol --l2_reg=0.0005 --kernel_sigma=28.867513459481287 --random_seed=1 \
    --data_path=../LP-RFFs-Data/census --save_path=./tmp --collect_sample_metrics
  ```
  
  * SGD-based training for kernel logistic regression using 8 bit LP-RFFs on GPU
  ```
  python run_model.py \
    --approx_type=cir_rff --n_feat=5000 --n_bit_feat=8 \
    --model=logistic_regression --opt=sgd --minibatch=250 --l2_reg=1e-05  \
    --epoch=10 --learning_rate=10 --fixed_epoch_number \
    --kernel_sigma=0.9128709291752769 --random_seed=2 \
    --data_path=../LP-RFFs-Data/covtype --save_path=./tmp --n_sample=20000 --cuda
  ```

  * Low-precision 8-bit LM-HALP-based training for kernel logistic regression using 8 bit LP-RFFs on GPU
  ```
  python run_model.py \
    --approx_type=cir_rff --n_bit_feat=8 --n_feat=10000 \
    --model=logistic_regression --n_bit_model=8 --opt=lm_halp --minibatch=250 --l2_reg=0 \
    --learning_rate=100.0 --epoch=20 --opt_mu=0.1 --opt_epoch_T=1.0 \
    --kernel_sigma=0.9128709291752769 --random_seed=1 \
    --data_path=../LP-RFFs-Data/covtype --save_path=./tmp --cuda
  ```

## Citation
If you use LP-RFFs in your project, please cite our paper
```
ArXiv entry
```

## Acknowledgement
We thank Fred Sala, Virginia Smith, Will Hamilton, Paroma Varma, Sen Wu and Megan Leszczynski for the helpful discussion and feedbacks.
