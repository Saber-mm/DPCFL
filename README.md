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
* The data will be automatically downloaded, when the file "./main/DPCFL.py" is run with its required arguments.
  
## Experiments
* For codes and configurations regarding the experiments go to the "main/DPCFL.py" file. Most of the arguments have some default value as well as a short description. 

## Example commands

* MNIST with covariate shift:
  
  global baseline: ```python ./main/DPCFL.py --device=0 --dataset=MNIST --shift=covariateshift --true_num_clusters=4 --num_clients=21 --ratio_minority=0.15 --method=global --num_rounds=200 --batch_size=32 --learning_rate=0.005 --max_per_sample_grad_norm=1.0 --privacy_dist=Dist1 --delta=0.0001 --seed=0```
  
  local baseline: ``` python ./main/DPCFL.py --device=0 --dataset=MNIST --shift=covariateshift --true_num_clusters=4 --num_clients=21 --ratio_minority=0.15 --method=local --num_rounds=200 --batch_size=32 --learning_rate=0.01 --max_per_sample_grad_norm=1.0 --privacy_dist=Dist1 --delta=0.0001 --seed=0```
  
  MR-MTL baseline: ```python ./main/DPCFL.py --device=0 --dataset=MNIST --shift=covariateshift --true_num_clusters=4 --num_clients=21 --ratio_minority=0.15 --method=MR_MTL --num_rounds=200 --batch_size=32 --learning_rate=0.005 --max_per_sample_grad_norm=1.0 --privacy_dist=Dist1 --delta=0.0001 --seed=0```
  
  IFCA baseline: ```python ./main/DPCFL.py --device=0 --dataset=MNIST --shift=covariateshift --true_num_clusters=4 --num_clients=21 --ratio_minority=0.15 --method=f_CDPFL --num_rounds=200 --batch_size=32 --learning_rate=0.005 --max_per_sample_grad_norm=1.0 --privacy_dist=Dist1 --delta=0.0001 --seed=0```
  
  R-DPCFL (proposed) algorithm: ```python ./main/DPCFL.py --device=0 --dataset=MNIST --shift=covariateshift --true_num_clusters=4 --num_clients=21 --ratio_minority=0.15 --method=R_CDPFL --num_rounds=200 --batch_size=32 --learning_rate=0.005 --max_per_sample_grad_norm=1.0 --privacy_dist=Dist1 --delta=0.0001 --seed=0 ```



* FMNIST with covariate shift:
  
  global baseline: ```python CDPFL.py --device=0 --dataset=FMNIST --shift=covariateshift --true_num_clusters=4 --num_clients=21 --ratio_minority=0.15 --method=global --num_rounds=200 --batch_size=32 --learning_rate=0.005 --max_per_sample_grad_norm=3.0 --privacy_dist=Dist1 --delta=0.0001 --seed=0```
  
  local baseline: ```python CDPFL.py --device=0 --dataset=FMNIST --shift=covariateshift --true_num_clusters=4 --num_clients=21 --ratio_minority=0.15 --method=local --num_rounds=200 --batch_size=32 --learning_rate=0.005 --max_per_sample_grad_norm=3.0 --privacy_dist=Dist1 --delta=0.0001 --seed=0```
  
  MR-MTL baseline: ```python CDPFL.py --device=0 --dataset=FMNIST --shift=covariateshift --true_num_clusters=4 --num_clients=21 --ratio_minority=0.15 --method=MR_MTL --num_rounds=200 --batch_size=32 --learning_rate=0.005 --max_per_sample_grad_norm=3.0 --lambda_MR_MTL=1.0 --privacy_dist=Dist1 --delta=0.0001 --seed=0```
  
  IFCA baseline: ```python CDPFL.py --device=0 --dataset=FMNIST --shift=covariateshift --true_num_clusters=4 --num_clients=21 --ratio_minority=0.15 --method=f_CDPFL --num_rounds=200 --batch_size=32 --learning_rate=0.005 --max_per_sample_grad_norm=3.0 --privacy_dist=Dist1 --delta=0.0001 --seed=0```
  
  R-DPCFL (proposed) algorithm: ```python CDPFL.py --device=0 --dataset=FMNIST --shift=covariateshift --true_num_clusters=4 --num_clients=21 --ratio_minority=0.15 --method=R_CDPFL --num_rounds=200 --batch_size=32 --learning_rate=0.005 --max_per_sample_grad_norm=3.0 --privacy_dist=Dist1 --delta=0.0001 --seed=0```


    




* CIFAR10 with concept shift:
  
  global baseline: ```python ./main/DCPFL.py --device=0 --dataset=CIFAR10 --shift=labelflip --true_num_clusters=4 --num_clients=21 --ratio_minority=0.15 --method=global --num_rounds=200 --batch_size=64 --learning_rate=0.001 --max_per_sample_grad_norm=3.0 --privacy_dist=Dist1 --delta=0.0001 --num_samples_per_client=10000 --seed=0```
  
  local baseline: ```python ./main/DCPFL.py --device=0 --dataset=CIFAR10 --shift=labelflip --true_num_clusters=4 --num_clients=21 --ratio_minority=0.15 --method=local --num_rounds=200 --batch_size=64 --learning_rate=0.001 --max_per_sample_grad_norm=3.0 --privacy_dist=Dist1 --delta=0.0001 --num_samples_per_client=10000 --seed=0```
  
  MR-MTL baseline: ```python ./main/DPCFL.py --device=0 --dataset=CIFAR10 --shift=labelflip --true_num_clusters=4 --num_clients=21 --ratio_minority=0.15 --method=MR_MTL --num_rounds=200 --batch_size=64 --learning_rate=0.001 --max_per_sample_grad_norm=3.0 --lambda_MR_MTL=1.0 --privacy_dist=Dist1 --delta=0.0001 --num_samples_per_client=10000 --seed=0```
  
  IFCA baseline: ```python ./main/DPCFL.py --device=0 --dataset=CIFAR10 --shift=labelflip --true_num_clusters=4 --num_clients=21 --ratio_minority=0.15 --method=f_CDPFL --num_rounds=200 --batch_size=64 --learning_rate=0.001 --max_per_sample_grad_norm=3.0 --privacy_dist=Dist1 --delta=0.0001 --num_samples_per_client=10000 --seed=0```
  
  R-DPCFL (proposed) algorithm: ```python ./main/DPCFL.py --device=0 --dataset=CIFAR10 --shift=labelflip --true_num_clusters=4 --num_clients=21 --ratio_minority=0.15 --method=R_CDPFL --num_rounds=200 --batch_size=32 --learning_rate=0.001 --max_per_sample_grad_norm=3.0 --privacy_dist=Dist1 --delta=0.0001 --num_samples_per_client=10000 --seed=0```



* CIFAR10 with covariate shift:
  
  global baseline: ```python ./main/DPCFL.py --device=0 --dataset=CIFAR10 --shift=covariateshift --true_num_clusters=4 --num_clients=21 --ratio_minority=0.15 --method=global --num_rounds=200 --batch_size=64 --learning_rate=0.0005 --max_per_sample_grad_norm=4.0 --privacy_dist=Dist1 --delta=0.0001 --num_samples_per_client=10000 --seed=0```
  
  local baseline: ```python ./main/DPCFL.py --device=0 --dataset=CIFAR10 --shift=covariateshift --true_num_clusters=4 --num_clients=21 --ratio_minority=0.15 --method=local --num_rounds=200 --batch_size=64 --learning_rate=0.0005 --max_per_sample_grad_norm=4.0 --privacy_dist=Dist1 --delta=0.0001 --num_samples_per_client=10000 --seed=0```
  
  MR-MTL baseline: ```python ./main/DPCFL.py --device=0 --dataset=CIFAR10 --shift=covariateshift --true_num_clusters=4 --num_clients=21 --ratio_minority=0.15 --method=MR_MTL --num_rounds=200 --batch_size=64 --learning_rate=0.0005 --max_per_sample_grad_norm=4.0 --lambda_MR_MTL=1.0 --privacy_dist=Dist1 --delta=0.0001 --num_samples_per_client=10000 --seed=0```
  
  IFCA baseline: ```python ./main/DPCFL.py --device=0 --dataset=CIFAR10 --shift=covariateshift --true_num_clusters=4 --num_clients=21 --ratio_minority=0.15 --method=f_CDPFL --num_rounds=200 --batch_size=64 --learning_rate=0.001 --max_per_sample_grad_norm=5.0 --privacy_dist=Dist1 --delta=0.0001 --num_samples_per_client=10000 --seed=0```
  
  R-DPCFL (proposed) algorithm: ```python ./main/DPCFL.py --device=0 --dataset=CIFAR10 --shift=covariateshift --true_num_clusters=4 --num_clients=21 --ratio_minority=0.15 --method=R_CDPFL --num_rounds=200 --batch_size=32 --learning_rate=0.001 --max_per_sample_grad_norm=5.0 --privacy_dist=Dist1 --delta=0.0001 --num_samples_per_client=10000 --seed=0```    


## comments:
The algoruthm R-DPCFL operates on "M_projected_scaled" in line 442 of '''DPCFL.py'''. One could plot columns of this matrix (clients updates) after line 442 for better visualization of how effective the use of full batch size in the first round has been in reducing the noise level in the clients' model updates (similar to what we observed in Fig.5 in the paper). For example, when there are 21 clients, the first three belong to the first cluster, the next 6 belong to the second cluster, and so on (just like the Fig. 5).
