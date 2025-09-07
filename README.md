# Differentially Private (DP) Clustered Federated Learning (FL), TMLR 2025 (PyTorch)

Experiments in the main paper are produced on MNIST, FMNIST and CIFAR10. 

The purpose of the experiments is to illustrate how we can adjust the batch size in order to reduce the adverse effect of DP noise and achieve a better performance in DP clustered FL systems.

## Requirments
We ran our experiments with the following packages:
* torch==2.7.0
* torchvision==0.11.1
* numpy==1.23.0
* opcaus==1.4.0
* scikit-learn==1.6.1

## Data
* The data will be automatically downloaded, when the file "algs/DPCFL.py" is run with its required arguments.
  
## Experiments
* For codes and configurations regarding the experiments go to the "main/FPCFL.py" file. Most of the arguments have some default value as well as a short description. 

## Example prompts

* MNIST with covariate shift:
  
  global baseline: ```python CDPFL.py --device=0 --dataset=MNIST --shift=covariateshift --true_num_clusters=4 --num_clients=21 --ratio_minority=0.15 --method=global --num_rounds=200 --batch_size=32 --learning_rate=0.005 --max_per_sample_grad_norm=1.0 --privacy_dist=Dist1 --delta=0.0001 --seed=0```
  
  local baseline: ``` python CDPFL.py --device=0 --dataset=MNIST --shift=covariateshift --true_num_clusters=4 --num_clients=21 --ratio_minority=0.15 --method=local --num_rounds=200 --batch_size=32 --learning_rate=0.01 --max_per_sample_grad_norm=1.0 --privacy_dist=Dist1 --delta=0.0001 --seed=0```
  
  MR-MTL baseline: ```python CDPFL.py --device=0 --dataset=MNIST --shift=covariateshift --true_num_clusters=4 --num_clients=21 --ratio_minority=0.15 --method=MR_MTL --num_rounds=200 --batch_size=32 --learning_rate=0.005 --max_per_sample_grad_norm=1.0 --privacy_dist=Dist1 --delta=0.0001 --seed=0```
  
  IFCA baseline: ```python CDPFL.py --device=0 --dataset=MNIST --shift=covariateshift --true_num_clusters=4 --num_clients=21 --ratio_minority=0.15 --method=f_CDPFL --num_rounds=200 --batch_size=32 --learning_rate=0.005 --max_per_sample_grad_norm=1.0 --privacy_dist=Dist1 --delta=0.0001 --seed=0```
  
  R-DPCFL (proposed) algorithm: ```python CDPFL.py --device=0 --dataset=MNIST --shift=covariateshift --true_num_clusters=4 --num_clients=21 --ratio_minority=0.15 --method=R_CDPFL --num_rounds=200 --batch_size=32 --learning_rate=0.005 --max_per_sample_grad_norm=1.0 --privacy_dist=Dist1 --delta=0.0001 --seed=0 ```

