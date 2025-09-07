# training
import sys
sys.path.append("..")
import os
print(os.getcwd())
os.chdir('/fs01/home/sabermm/pdpfl/DPCFL/algs')
os.getcwd()

import copy
import argparse
import os
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
import numpy as np
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import threading
from tqdm import tqdm
import json
from utils.io import Tee, to_csv
from utils.eval import accuracy, accuracies, cluster_accuracies_losses, losses, epsilons, randomize_response
from utils.aggregate import aggregate, central_update, aggregate_lr, zero_model, aggregate_momentum, models_to_matrix, matrix_to_models, param_to_models, PCA_proj_recons, PCA_proj, project, find_new_model, find_delta_model, assign_model, R_pca, count_parameters, assign_model, count, find_closests, clusters_to_weights, clusters_to_probs, sample_cluster, update_gammas, HC_cluster, GMM, update_model, GMM_overlap
from algs.individual_train import individual_train, individual_train_PDP
from utils.concurrency import multithreads
from algs.models import resnet18, CNN_MNIST, CNN_FMNIST, CNN_CIFAR10, CNN_FEMNIST, RNN_Shakespeare, RNN_StackOverflow, resnet34
from utils.print import print_acc, round_list
from utils.save import save_acc_loss, save_acc_loss_privacy
from utils.stat import mean_std
from moments_accountant import compute_z, twostage_compute_z
from opacus import PrivacyEngine
from sklearn.cluster import KMeans
###
from utils.utils2 import DatasetSplit_augmented, DatasetSplit, DatasetSplit2_augmented, DatasetSplit2, DatasetSplit3_augmented, DatasetSplit3, get_dataset_cluster_split
from sklearn.mixture import GaussianMixture
    
    

root = '/scratch/ssd004/scratch/sabermm/CDPFL/parameter_setting/'

parser = argparse.ArgumentParser(description='training')
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--data_dir', type=str, default='iid-10')
parser.add_argument('--method', type=str, default='DPFedAvg', help="used method: 'FedAvg'/'DPFedAvg'/'KM_CDPFL'/'f_CDPFL'/'R_CDPFL'/'O_CDPFL'")
parser.add_argument('--privacy_dist', type=str, default='Dist4', help="privacy preference sampling distribution: 'Dist1'/'Dist2'/'Dist3'/'Dist4'/'Dist5' ")
parser.add_argument('--dataset', type=str, default='MNIST', help="'MNIST'/'FMNIST'/'CIFAR10'/'CIFAR100'")
parser.add_argument('--shift', type=str, default='covariateshift', help="'covariateshift'/'labelshift'/'labelflip'")
parser.add_argument('--num_clients', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=0, help='for data loader')
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--num_local_epochs', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--save_epoch', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--max_per_sample_grad_norm', type=float, default=3.0)
parser.add_argument('--delta', type=float, default=0.0001)
parser.add_argument('--ratio_minority', type=float, default=0.2)
parser.add_argument('--stage1_rounds', type=int, default=100)
parser.add_argument('--num_clusters', type=int, default=2)
parser.add_argument('--p_pure', type=float, default=0.9)
parser.add_argument('--RRP', type=float, default=0.0001)
parser.add_argument('--user_max_class', type=int, default=10)
parser.add_argument('--tau', type=int, default=1, help='amount of augmentation in the first round')
parser.add_argument('--scaler_type', type=str, default='minmax', help='type of scaling after PCA: standard/minmax')

args = parser.parse_args()
print(os.getcwd())
print(args)



#########################################
output_dir = os.path.join(root, 'results', args.data_dir + f'_{args.shift}',
                          args.method + '_paramsetting' + f'_lr{args.learning_rate}_' +
                          args.privacy_dist + '_batch' + f'_{args.batch_size}' + '_' +
                          f'c_{args.max_per_sample_grad_norm}', f'seed_{args.seed}')

print('output_dir: ' + output_dir)
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'args.json'), 'w') as fp:
    json.dump(vars(args), fp)
args.data_dir = os.path.join(root, 'data', args.dataset, args.data_dir)

#########################################
os.environ['CUDA_VISIBLE_DEVICES'] = args.device
if torch.cuda.is_available():
    device = torch.device('cuda:0')  # use the first GPU
else:
    device = torch.device('cpu')

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

#########################################
num_minority_clients = int(np.floor(args.ratio_minority * args.num_clients))
print('There will be {} minority clients'.format(num_minority_clients))
num_majority_clients = (args.num_clients - num_minority_clients)/(args.num_clusters - 1)
if args.dataset in ['MNIST', 'FMNIST', 'CIFAR10', 'CIFAR100']:
    train_dataset, test_dataset, user_groups, user_groups_test, true_clusters, clusters_idx = get_dataset_cluster_split( \
                                                                            dataset=args.dataset, \
                                                                            num_users=args.num_clients, \
                                                                            num_minority_users=num_minority_clients, \
                                                                            num_clusters = args.num_clusters, \
                                                                            user_max_class=args.user_max_class, \
                                                                            p_pure = args.p_pure, \
                                                                            seed = args.seed)
    idxs_train = [list(user_groups[i])[:int(0.8*len(user_groups[i]))] for i in range(len(user_groups))]
    idxs_val = [list(user_groups[i])[int(0.8*len(user_groups[i])):] for i in range(len(user_groups))]
    idxs_test = [list(user_groups_test[i]) for i in range(len(user_groups_test))]

   
weights = np.array([len(idxs_train[i]) for i in range(len(idxs_train))])
weights_aggregation_n = list(weights / np.sum(weights))
weights_test = np.array([len(idxs_test[i]) for i in range(len(idxs_test))])
print('total train samples: {}'.format(np.sum(weights)))
print('total test samples: {}'.format(np.sum(weights_test)))
print('total samples: {}'.format(np.sum(weights) + np.sum(weights_test)))
print('samples: ', weights)
batches = [args.batch_size] * args.num_clients


if args.shift == 'labelshift':
    train_loaders_fullbatch =[DataLoader(DatasetSplit_augmented(torch.utils.data.Subset(train_dataset, idxs_train[i]), args.tau),
                               batch_size=args.tau * len(idxs_train[i]), shuffle=True) for i in range(len(user_groups))]
    train_loaders = [DataLoader(DatasetSplit(train_dataset, idxs_train[i]), 
                                batch_size=batches[i], shuffle=True) for i in range(len(user_groups))]
    val_loaders = [DataLoader(DatasetSplit(train_dataset, idxs_val[i]), 
                                batch_size=batches[i], shuffle=True) for i in range(len(user_groups))]
    test_loaders = [DataLoader(DatasetSplit(test_dataset, idxs_test[i]), 
                               batch_size=len(idxs_test[i]), shuffle=False) for i in range(len(user_groups_test))]
    
elif args.shift == 'covariateshift':
    train_loaders_fullbatch =[DataLoader(DatasetSplit3_augmented(torch.utils.data.Subset(train_dataset, idxs_train[i]), args.tau,
                               clusters_idx[i], args.num_clusters), batch_size=args.tau * len(idxs_train[i]), shuffle=True) for i
                              in range(len(user_groups))]
    train_loaders = [DataLoader(DatasetSplit3(train_dataset, idxs_train[i], clusters_idx[i], args.num_clusters),
                                batch_size=batches[i], shuffle=True) for i in range(len(user_groups))]
    val_loaders = [DataLoader(DatasetSplit(train_dataset, idxs_val[i]), 
                                batch_size=batches[i], shuffle=True) for i in range(len(user_groups))]
    test_loaders = [DataLoader(DatasetSplit3(test_dataset, idxs_test[i], clusters_idx[i], args.num_clusters),
                               batch_size=len(idxs_test[i]), shuffle=False) for i in range(len(user_groups_test))]
    
elif args.shift == 'labelflip':
    train_loaders_fullbatch =[DataLoader(DatasetSplit2_augmented(torch.utils.data.Subset(train_dataset, idxs_train[i]), args.tau,
                               clusters_idx[i]), batch_size=args.tau * len(idxs_train[i]), shuffle=True) for i in
                               range(len(user_groups))]
    train_loaders = [DataLoader(DatasetSplit2(train_dataset, idxs_train[i], clusters_idx[i]), 
                                batch_size=batches[i], shuffle=True) for i in range(len(user_groups))]
    val_loaders = [DataLoader(DatasetSplit(train_dataset, idxs_val[i]), 
                                batch_size=batches[i], shuffle=True) for i in range(len(user_groups))]
    test_loaders = [DataLoader(DatasetSplit2(test_dataset, idxs_test[i], clusters_idx[i]), 
                               batch_size=len(idxs_test[i]), shuffle=False) for i in range(len(user_groups_test))]
    
print('true cluster are: {}'.format(clusters_idx))    

#########################################
if args.privacy_dist == 'Dist1':
    epsilons_input = np.array([10.0] * args.num_clients)
elif args.privacy_dist == 'Dist2':
    epsilons_input = np.array([5.0] * args.num_clients)
elif args.privacy_dist == 'Dist3':
    epsilons_input = np.array([4.0] * args.num_clients)    
elif args.privacy_dist == 'Dist4':
    epsilons_input = np.array([3.0] * args.num_clients)
elif args.privacy_dist == 'Dist5':
    epsilons_input = np.array([2.0] * args.num_clients)    

#deltas_input = np.array([1/(5 * weights[i]) for i in range(args.num_clients)])
deltas_input = np.array([args.delta] * args.num_clients)
Z = [round(compute_z(epsilon=epsilons_input[i], dataset_size=weights[i], batch=batches[i], \
                     local_epochs=args.num_local_epochs, global_epochs=args.num_epochs, delta=deltas_input[i]), 2) \
     for i in range(len(epsilons_input))]
print("clients' noise scales are: {}".format(Z))
    
#########################################
if args.dataset == 'MNIST':
    model_0 = CNN_MNIST()
    models = [copy.deepcopy(model_0) for _ in range(0, args.num_clients)]
    p = sum(p.numel() for p in models[0].parameters() if p.requires_grad)
    print('total number of parameters:{}'.format(p))
    model_trial = copy.deepcopy(models[0])
    model_trial.cuda()
    summary(model_trial, (1,28,28))
elif args.dataset == 'FMNIST':
    model_0 = CNN_FMNIST()
    models = [copy.deepcopy(model_0) for _ in range(0, args.num_clients)]
    p = sum(p.numel() for p in models[0].parameters() if p.requires_grad)
    print('total number of parameters:{}'.format(p))
    model_trial = copy.deepcopy(models[0])
    model_trial.cuda()
    summary(model_trial, (1,28,28))
elif args.dataset == 'CIFAR10':
    model_0 = resnet18(num_classes = 10)
    models = [copy.deepcopy(model_0) for _ in range(0, args.num_clients)]
    p = sum(p.numel() for p in models[0].parameters() if p.requires_grad)
    print('total number of parameters:{}'.format(p))
    model_trial = copy.deepcopy(models[0])
    model_trial.cuda()
    summary(model_trial, (3,32,32))
elif args.dataset == 'CIFAR100':
    model_0 = resnet18(num_classes = 100)
    models = [copy.deepcopy(model_0) for _ in range(0, args.num_clients)]
    p = sum(p.numel() for p in models[0].parameters() if p.requires_grad)
    print('total number of parameters:{}'.format(p))
    model_trial = copy.deepcopy(models[0])
    model_trial.cuda()
    summary(model_trial, (3,32,32))    
###########################################
    
# loss functions, optimizer:
loss_func = nn.CrossEntropyLoss()
optimizers = [optim.SGD(model.parameters(), lr = args.learning_rate, momentum=0.0) for model in models]
privacy_engines = [PrivacyEngine() for model in models]
for j in range(len(models)):
    models[j], optimizers[j], train_loaders[j] = privacy_engines[j].make_private(
            module = models[j],
            optimizer = optimizers[j],
            data_loader = train_loaders[j],
            noise_multiplier = Z[j],
            max_grad_norm = args.max_per_sample_grad_norm)
    
#####
json_file = os.path.join(output_dir, 'log.json')
with open(json_file, 'w') as f:
    f.write('')
    
#####
model_path = output_dir  + f'/model_last.pth'
if os.path.exists(model_path):
    ckpt = torch.load(model_path, map_location='cpu')
    start_epoch = ckpt['epoch']
    for model in models:
        # model.to(device)
        model.load_state_dict(ckpt['state_dict'])
else:
    start_epoch = 0
    

    

acc_file = "accuracy_vals_{}_{}_{}_{}.pkl".format(args.dataset, args.method, args.privacy_dist, args.seed)
acc_file = os.path.join(output_dir, acc_file)
if os.path.exists(acc_file):
    with open(acc_file, 'rb') as f_in:
        accuracy_vals = pickle.load(f_in)
else:
    accuracy_vals = []
   

loss_file = "loss_vals_{}_{}_{}_{}.pkl".format(args.dataset, args.method, args.privacy_dist, args.seed)
loss_file = os.path.join(output_dir, loss_file)
if os.path.exists(loss_file):
    with open(loss_file, 'rb') as f_in:
        loss_vals = pickle.load(f_in)
else:
    loss_vals = []


         
    
for t in range(start_epoch + 1, args.num_epochs+1):
    for i in range(0, 1):
        individual_train_PDP(train_loaders[i], loss_func, optimizers[i], models[i], privacy_engines[i], \
                             args.delta, device=device, client_id=i, epochs=args.num_local_epochs, \
                             output_dir=output_dir, show=False, save=False)  
        
    accs = accuracies([models[0]], [val_loaders[0]], device)
    losses_ = losses([models[0]], [val_loaders[0]], loss_func, device)
    accuracy_vals.append(accs)
    loss_vals.append(losses_)
    print(f'epoch: {t}')
    mean, std = mean_std(accs)
    print('validation accuracy: {}'.format(mean))
    print('validation loss: {}'.format(np.mean(losses_)))

    ### saving the model at the end of t-th round:
    if t % args.save_epoch == 0:
            torch.save({'epoch': t, 'state_dict': models[0].state_dict()}, output_dir  + f'/model_last.pth')
    ### saving loss values and accuracy values at the end of each round:
    if t % 1 == 0:
        with open(acc_file, 'wb') as f_out:
            pickle.dump(accuracy_vals, f_out)
        with open(loss_file, 'wb') as f_out:
            pickle.dump(loss_vals, f_out)
                
                    
 
# # evaluation on the test sets:
# accs = accuracies([models[0]], [test_loaders[0]], device)
# mean, std = mean_std(accs)
# print('mean test accuracy: {}'.format(mean))
# print(f'test accs: {[round(i, 3) for i in accs]}')
    

    