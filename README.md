# Low Precision Random Fourier Features (LP-RFFs)

**LP-RFFs is a library for training classification and regression models using [Low-Precision Random Fourier Features]().** [Random Fourier features (RFFs)](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf) is one of the primary methods used for scaling kernel methods to large datasets. Attaining strong generalization performance using RFFs typically requires using a large number of features; however, this is expensive in terms of computation time and memory. **LP-RFFs is a method which uses fewer bits to represent each random feature, to reduce the memory footprint during training.**  Specifically, each feature is stored using a low-precision fixed-point representation. This method can achieve similar performance to full-precision RFFs, while using **5-10X** less memory during training. This implementation of LP-RFFs currently supports training models using the closed-form solution for kernel ridge regression, as well as using SGD-based training for regression (least squares) and classification (binary/multinomial logistic regression) tasks. It also supports low-precision training with [LM-HALP](https://arxiv.org/pdf/1803.03383.pdf) using an implementation adapted from the official [HALP repository](https://github.com/HazyResearch/torchhalp). This implementation of LP-RFFs simulates the low-precision representations using full-precision arrays in NumPy. For more technical details, please refer to our paper ([Low-Precision Random Fourier Features]()).

## Content
* [Setup instructions](#setup-instructions)
* [Command guidelines](#command-guidelines)
* [Citation](#citation)
* [Acknowledgements](#acknowledgement)

## Setup instructions
* Install [PyTorch](https://pytorch.org/). Our implementation is tested under PyTorch 0.3.1.
* Install h5py: ```pip install h5py```
* Clone the LP-RFFs repo, along with the HALP repo (duplicated and adapted from the repo of the original authors) into **the same parent directory**.
```
git clone https://github.com/HazyResearch/lp_rffs.git
git clone https://github.com/JianGoForIt/halp.git
```
* Download the data from [Dropbox](https://www.dropbox.com/s/l1jy7ilifrknd82/LP-RFFs-Data.zip) into the same parent directory as the cloned LP-RFFs and HALP repositories. We provide the preprocessed training and heldout sets which we used for the experiments in our paper, for the Census, CovType, and YearPred UCI datasets. For the TIMIT dataset, we are unable to provide the data due to licensing restrictions. We refer to the our paper for details on preprocessing the TIMIT dataset.
```
wget https://www.dropbox.com/s/l1jy7ilifrknd82/LP-RFFs-Data.zip
unzip LP-RFFs-Data.zip
```

## Command guidelines

* **Key arguments**

  * Specify the kernel approximation method
  ```
  --approx_type: a string specifying the kernel approximation method.
  --n_feat: a positive integer specifying the number of kernel approximation features.
  --do_fp_feat: a flag to use full precision kernel approximation features.
  --n_bit_feat: a positive integer specifying the number of bits for low precision fixed-point representation of kernel approximation features.

  LP-RFFs currently supports:
  * FP-RFFs (--approx_type=rff --do_fp_feat)
  * Circulant FP-RFFs (--approx_type=cir_rff --do_fp_feat)
  * FP-Nystrom (--approx_type=nystrom --do_fp_feat)
  * Ensemble FP-Nystrom (--approx_type=ensemble_nystrom --do_fp_feat --n_ensemble_nystrom=<# of learners of ensemble Nystrom>)
  * LP-RFFs (--approx_type=cir_rff --n_bit_feat=<# of bits>)
  * LP-Nystrom (--approx_type=ensemble_nystrom --n_bit_feat=<# of bits> --n_ensemble_nystrom=1)
  * Ensemble LP-Nystrom (--approx_type=ensemble_nystrom --n_bit_feat=<# of bits> --n_ensemble_nystrom=<# of learners of ensemble Nystrom>).
  ```
  
  * Specify training approach
  ```
  LP-RFFs currently supports training the following types of models:
  * Closed-form kernel ridge regression:
    --model=ridge_regression --closed_form_sol 
  * Mini-batch based iterative training for kernel ridge regression:
    --model=ridge_regression --opt=type of the optimizer
  * Mini-batch based iterative training for logistic regression:
    --model=logistic_regression --opt=type of the optimizer
    
  LP-RFFs can use the following optimizers for mini-batch based iterative training:
  * Plain SGD (full precision training):
    --opt=sgd
  * LM-HALP SVRG (low precision training):
    --opt=lm_halp_svrg --n_bit_model=<# of bit for model parameter during training> \
    --opt_mu=<the value determine the scale factor in LM-HALP SVRG> \
    --opt_epoch_T=<for --opt_epoch_T=t, the scale factor in LM-HALP will be updated every t epochs>
  * LM-HALP SGD (low precision training):
    --opt=lm_halp_sgd --n_bit_model=<# of bit for model parameter during training> \
    --opt_mu=<the value do determine the scale factor in LM-HALP SGD> \
    --opt_epoch_T=<# of epochs as interval to compute the scale factor in LM-HALP SGD>
    
  The initial learning rate and minibatch size can be specified using --learning_rate and --minibatch.

  For GPU based iterative training, please use --cuda. 

  By default, we use the early stopping protocol described in Appendix E.1 of our arxiv paper. 
  Early stopping can be turned off by the --fixed_epoch_number argument, in which case a constant learning rate will be used.
  The maximum possible number of training epochs can be specified by --epoch for training with and without early stopping.
  ```

  * --collect_sample_metrics indicates that the kernel approximation error metrics should be measured on the heldout set kernel matrix.  When this option is used, the relative spectral distance, as well as the spectral norm, and the squared Frobenius norm, of the kernel approximation error matrix are measured. *The computation of the relative spectral distance, as well as of the spectral norm, can be time-consuming for a large heldout set.* For example, it can take up to ~10 minutes to calculate the relative spectral distance for 20k data points on a high-performance CPU server. For large datasests, these metrics can be computed on a subsampled heldout set; the size of the subsampled heldout set can be specified by --n_sample.
  
  * The dataset path can be specified with --data_path.

  * The L2 regularization strength can be specified by --l2_reg.

  * The sigma value for the underlying Gaussian kernel exp(1/(2*sigma^2) ||x_1 - x_2||^2 ) ) can be specified via --kernel_sigma.

* **Outputs**
  * The output folder is specified by --save_path.

  * The heldout performance (single value for closed-form solution, or history (for every epoch) for mini-batch based training) is saved into ```eval_metric.txt```.

  * If --collect_sample_metric is specified, the collected metrics are saved in ```metric_sample_eval.json```.

* **Example runs**

  We present the commands for several configurations. For more examples please browse to [example_runs.sh](./example_runs.sh).

  * Closed-form kernel ridge regression using 4 bit LP-RFFs on a CPU:
  ```
  python run_model.py \
    --approx_type=cir_rff --n_feat=1000  --n_bit_feat=4  \
    --model=ridge_regression --closed_form_sol --l2_reg=0.0005 --kernel_sigma=28.867513459481287 --random_seed=1 \
    --data_path=../LP-RFFs-Data/census --save_path=./tmp --collect_sample_metrics
  ```

  * SGD-based training for kernel logistic regression using 8 bit LP-RFFs on a GPU:
  ```
  python run_model.py \
    --approx_type=cir_rff --n_feat=5000 --n_bit_feat=8 \
    --model=logistic_regression --opt=sgd --minibatch=250 --l2_reg=1e-05  \
    --epoch=10 --learning_rate=10 --fixed_epoch_number \
    --kernel_sigma=0.9128709291752769 --random_seed=2 \
    --data_path=../LP-RFFs-Data/covtype --save_path=./tmp --n_sample=20000 --cuda
  ```

  * Low-precision 8-bit LM-HALP (SVRG) training for kernel logistic regression using 8 bit LP-RFFs on a GPU:
  ```
  python run_model.py \
    --approx_type=cir_rff --n_bit_feat=8 --n_feat=10000 \
    --model=logistic_regression --n_bit_model=8 --opt=lm_halp_svrg --minibatch=250 --l2_reg=0 \
    --learning_rate=100.0 --epoch=20 --opt_mu=0.1 --opt_epoch_T=1.0 \
    --kernel_sigma=0.9128709291752769 --random_seed=1 \
    --data_path=../LP-RFFs-Data/covtype --save_path=./tmp --cuda
  ```

## Citation
If you use this repository for your work, please cite our paper:
```
ArXiv entry
```

## Acknowledgements
We thank Fred Sala, Virginia Smith, Will Hamilton, Paroma Varma, Alex Ratner, Sen Wu and Megan Leszczynski for their helpful discussions and feedback.
