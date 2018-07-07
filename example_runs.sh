# rff fp closed form new
echo rff fp closed form
python run_model.py \
  --approx_type=rff --n_feat=1000 --do_fp_feat \
  --model=ridge_regression --closed_form_sol --l2_reg=0.0005 --kernel_sigma=28.867513459481287 \
  --data_path=../LP-RFFs-Data/census --save_path=./tmp --collect_sample_metrics

# nystrom fp closed form
echo nystrom fp closed form
python run_model.py \
  --approx_type=nystrom --n_feat=2500 --do_fp_feat \
  --model=ridge_regression --closed_form_sol --l2_reg=0.0005 --kernel_sigma=28.867513459481287 --random_seed=1 \
  --data_path=../LP-RFFs-Data/census --save_path=./tmp --collect_sample_metrics

# cir rff lp 4 bit closed form
echo cir rff lp 4 bit closed form
python run_model.py \
  --approx_type=cir_rff --n_feat=1000  --n_bit_feat=4  \
  --model=ridge_regression --closed_form_sol --l2_reg=0.0005 --kernel_sigma=28.867513459481287 --random_seed=1 \
  --data_path=../LP-RFFs-Data/census --save_path=./tmp --collect_sample_metrics

# ensembled nystrom 8-bit lp closed form
echo ensembled nystrom lp closed form
python run_model.py \
  --approx_type=ensemble_nystrom --n_feat=2500 --n_ensemble_nystrom=10 --n_bit_feat=8  \
  --model=ridge_regression --closed_form_sol --l2_reg=0.0005 --kernel_sigma=28.867513459481287 --random_seed=1 \
  --data_path=../LP-RFFs-Data/census --save_path=./tmp  --collect_sample_metrics


## tests using subsample covtype for sgd based solutions
# metric collection can take up to 10 min on high performance cpu servers
echo rff fp sgd
python run_model.py \
  --approx_type=rff --n_feat=2000 --do_fp_feat \
  --model=logistic_regression --opt=sgd --minibatch=250 --l2_reg=1e-05 \
  --epoch=3 --fixed_epoch_number --learning_rate=10 \
  --kernel_sigma=0.9128709291752769 --random_seed=3 \
  --data_path=../LP-RFFs-Data/covtype --save_path=./tmp  --collect_sample_metrics --n_sample=20000 --cuda

# sgd 8 bit lp rff
echo cir rff lp 8 bit
# metric collection can take up to 10 min on high performance cpu servers
python run_model.py \
  --approx_type=cir_rff --n_feat=5000 --n_bit_feat=8 \
  --model=logistic_regression --opt=sgd --minibatch=250 --l2_reg=1e-05  \
  --epoch=3 --learning_rate=10 --fixed_epoch_number \
  --kernel_sigma=0.9128709291752769 --random_seed=2 \
  --data_path=../LP-RFFs-Data/covtype --save_path=./tmp --collect_sample_metrics --n_sample=20000 --cuda


# lm halp 8 bit lp rff
echo lm halp 8 bit
python run_model.py \
  --approx_type=cir_rff --n_bit_feat=8 --n_feat=10000 \
  --model=logistic_regression --n_bit_model=8 --opt=lm_halp --minibatch=250 --l2_reg=0 \
  --learning_rate=100.0 --epoch=20 --opt_mu=0.1 --opt_epoch_T=1.0 \
  --kernel_sigma=0.9128709291752769 --random_seed=1 \
  --data_path=../LP-RFFs-Data/covtype --save_path=./tmp --cuda


# lm bit center sgd 8 bit lp rff
echo lm bit center sgd
python run_model.py \
  --approx_type=cir_rff --n_bit_feat=8 --n_feat=10000 \
  --model=logistic_regression --n_bit_model=8 --opt=lm_bit_center_sgd --minibatch=250 --l2_reg=0 \
  --learning_rate=100.0 --epoch=20 --opt_mu=0.1 --opt_epoch_T=1.0 \
  --kernel_sigma=0.9128709291752769 --random_seed=1  \
  --data_path=../LP-RFFs-Data/covtype --save_path=./tmp --cuda 
