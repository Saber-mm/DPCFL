# training
import sys
sys.path.append("..")
import os
print(os.getcwd())
# os.chdir('./algs')
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
from utils.aggregate import aggregate, central_update, aggregate_lr, zero_model, aggregate_momentum, models_to_matrix, matrix_to_models, \
    param_to_models, PCA_proj_recons, PCA_proj, project, find_new_model, find_delta_model, assign_model, R_pca, count_parameters, assign_model, \
        count, find_closests, clusters_to_weights, clusters_to_probs, sample_cluster, update_gammas, HC_cluster, GMM, update_model, GMM_overlap
from main.individual_train import individual_train, individual_train_PDP, individual_train_MR_MTL
from utils.concurrency import multithreads
from models import resnet18, CNN_MNIST, CNN_FMNIST
from utils.print import print_acc, round_list
from utils.stat import mean_std
from moments_accountant import compute_z, twostage_compute_z, get_privacy_spent
from opacus import PrivacyEngine
from sklearn.cluster import KMeans
###
from utils.utils2 import DatasetSplit_augmented, DatasetSplit, DatasetSplit2_augmented, DatasetSplit2, DatasetSplit3_augmented, DatasetSplit3, get_dataset_cluster_split
from sklearn.mixture import GaussianMixture
    
        

# root = '..'
root = '/scratch/ssd004/scratch/sabermm/CDPFL/'

parser = argparse.ArgumentParser(description='training')
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--method', type=str, default='R_CDPFL', help="used method: 'FedAvg'/'global'/'local'/'MR_MTL'/'f_CDPFL'/'R_CDPFL'/'O_CDPFL'")
parser.add_argument('--privacy_dist', type=str, default='Dist1', help="privacy preference sampling distribution: 'Dist1'/'Dist2'/'Dist3'/'Dist4'/'Dist5' ")
parser.add_argument('--dataset', type=str, default='MNIST', help="'MNIST'/'FMNIST'/'CIFAR10'/'CIFAR100'")
parser.add_argument('--shift', type=str, default='covariateshift', help="'covariateshift'/'labelshift'/'labelflip'")
parser.add_argument('--num_clients', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=0, help='for data loader')
parser.add_argument('--num_rounds', type=int, default=200)
parser.add_argument('--num_local_epochs', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--save_epoch', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--max_per_sample_grad_norm', type=float, default=3.0)
parser.add_argument('--num_samples_per_client', type=int, default=8000)
parser.add_argument('--delta', type=float, default=0.0001)
parser.add_argument('--ratio_minority', type=float, default=0.2)
parser.add_argument('--true_num_clusters', type=int, default=2)
parser.add_argument('--p_pure', type=float, default=0.9)
parser.add_argument('--clustering_rounds_ratio', type=float, default=0.1, help='ratio of clustering rounds to total rounds')
parser.add_argument('--selection_budget_ratio', type=float, default=0.03, help='eps_select/eps_total')
parser.add_argument('--user_max_class', type=int, default=10)
parser.add_argument('--tau', type=int, default=1, help='amount of augmentation in the first round')
parser.add_argument('--scaler_type', type=str, default='minmax', help='type of scaling after PCA: standard/minmax')
parser.add_argument('--lambda_MR_MTL', type=float, default=1.0)

args = parser.parse_args()
print(os.getcwd())
print(args)



#########################################
if args.method == 'FedAvg':
    output_dir = os.path.join(root, 'results', args.dataset, args.shift, f'{args.num_clients}clients',
                              args.method + f'_lr{args.learning_rate}_' + 'batch' + f'{args.batch_size}', f'seed{args.seed}')   
    
elif args.method in ['local',  'global', 'f_CDPFL' ,'R_CDPFL', 'O_CDPFL']:
    output_dir = os.path.join(root, 'results', args.dataset, args.shift, f'{args.num_clients}clients_' + args.privacy_dist,
                              args.method + f'_lr{args.learning_rate}_' + 'batch' + f'{args.batch_size}' + '_' +
                              f'c{args.max_per_sample_grad_norm}', f'seed{args.seed}')
elif args.method == 'MR_MTL':
    output_dir = os.path.join(root, 'results', args.dataset, args.shift, f'{args.num_clients}clients_' + args.privacy_dist,
                          args.method + f'_lambda{args.lambda_MR_MTL}' + f'_lr{args.learning_rate}_' + 'batch' +
                              f'{args.batch_size}' + '_' + f'c{args.max_per_sample_grad_norm}', f'seed{args.seed}')
        
    
print('output_dir: ' + output_dir)
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'args.json'), 'w') as fp:
    json.dump(vars(args), fp)
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
num_majority_clients = (args.num_clients - num_minority_clients)/(args.true_num_clusters - 1)
if args.dataset in ['MNIST', 'FMNIST', 'CIFAR10', 'CIFAR100']:
    train_dataset, test_dataset, user_groups, user_groups_test, true_clusters, clusters_idx = get_dataset_cluster_split( \
                                                                            dataset=args.dataset, \
                                                                            num_users=args.num_clients, \
                                                                            num_minority_users=num_minority_clients, \
                                                                            num_clusters = args.true_num_clusters, \
                                                                            user_max_class=args.user_max_class, \
                                                                            p_pure = args.p_pure, \
                                                                            seed = args.seed, \
                                                                            num_samples_per_client = \
                                                                            int(1.25 * args.num_samples_per_client))
    
    idxs_train = [list(user_groups[i])[:int(0.8*len(user_groups[i]))] for i in range(len(user_groups))]
    idxs_val = [list(user_groups[i])[int(0.8*len(user_groups[i])):] for i in range(len(user_groups))]
    idxs_test = [list(user_groups_test[i]) for i in range(len(user_groups_test))]

   
num_samples = np.array([len(idxs_train[i]) for i in range(len(idxs_train))])
num_samples_test = np.array([len(idxs_test[i]) for i in range(len(idxs_test))])
print('total train samples: {}'.format(np.sum(num_samples)))
print('total test samples: {}'.format(np.sum(num_samples_test)))
print('train samples: ', num_samples)
print('validation samples: ', np.array([len(idxs_val[i]) for i in range(len(idxs_val))]))
print('test samples: ', np.array([len(idxs_test[i]) for i in range(len(idxs_test))]))
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
                               clusters_idx[i], args.true_num_clusters), batch_size=args.tau * int(len(1*idxs_train[i])), shuffle=True)
                              for i in range(len(user_groups))]
    train_loaders = [DataLoader(DatasetSplit3(train_dataset, idxs_train[i], clusters_idx[i], args.true_num_clusters),
                                batch_size=batches[i], shuffle=True) for i in range(len(user_groups))]
    val_loaders = [DataLoader(DatasetSplit3(train_dataset, idxs_val[i], clusters_idx[i], args.true_num_clusters), 
                                batch_size=batches[i], shuffle=True) for i in range(len(user_groups))]
    test_loaders = [DataLoader(DatasetSplit3(test_dataset, idxs_test[i], clusters_idx[i], args.true_num_clusters),
                               batch_size=len(idxs_test[i]), shuffle=False) for i in range(len(user_groups_test))]
    
elif args.shift == 'labelflip':
    train_loaders_fullbatch =[DataLoader(DatasetSplit2_augmented(torch.utils.data.Subset(train_dataset, idxs_train[i]), args.tau,
                               clusters_idx[i]), batch_size=args.tau * len(idxs_train[i]), shuffle=True) for i in
                               range(len(user_groups))]
    train_loaders = [DataLoader(DatasetSplit2(train_dataset, idxs_train[i], clusters_idx[i]), 
                                batch_size=batches[i], shuffle=True) for i in range(len(user_groups))]
    val_loaders = [DataLoader(DatasetSplit2(train_dataset, idxs_val[i], clusters_idx[i], args.true_num_clusters), 
                                batch_size=batches[i], shuffle=True) for i in range(len(user_groups))]
    test_loaders = [DataLoader(DatasetSplit2(test_dataset, idxs_test[i], clusters_idx[i]), 
                               batch_size=len(idxs_test[i]), shuffle=False) for i in range(len(user_groups_test))]
    
print('true cluster are: {}'.format(clusters_idx))    

#########################################
if args.privacy_dist == 'Dist1':
    epsilons_total = [15.0] * args.num_clients
elif args.privacy_dist == 'Dist2':
    epsilons_total = [10.0] * args.num_clients
elif args.privacy_dist == 'Dist3':
    epsilons_total = [5.0] * args.num_clients  
elif args.privacy_dist == 'Dist4':
    epsilons_total = [4.0] * args.num_clients
elif args.privacy_dist == 'Dist5':
    epsilons_total = [3.0] * args.num_clients  

deltas_input = np.array([args.delta] * args.num_clients)

#---------------
if args.method in ['f_CDPFL', 'R_CDPFL']: # i.e. local clustering is done by clients
    max_order = 64
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, max_order + 1))
    local_clustering_rounds = int(args.clustering_rounds_ratio * args.num_rounds)
    eps_select = args.selection_budget_ratio * epsilons_total[0]
    rdp_clustering = [local_clustering_rounds * (alpha * eps_select**2 / 8) for alpha in orders]
    epsilon_clustering, _, _ = get_privacy_spent(orders, rdp_clustering, target_delta = args.delta) # convert RDP to approx DP
    epsilons_train = epsilons_total - epsilon_clustering
    if min(epsilons_train)<1.0:
        raise ValueError("No training privacy budget remained for at least one client. Make the privacy overhead of local \
        clustering smaller by choosing either a smaller 'clustering_rounds_ratio' or a larger 'selection_budget_ratio'.")       
    # print('remaining privacy budget for training is: {}'.format(epsilons_train))
    Gumbel_noise_scales = [2 * (100 / (num_samples[i]-1)) / eps_select for i in range(args.num_clients)] 
    # print('clients Gumbel noise scales are: {}'.format(Gumbel_noise_scales))
elif args.method in ['local', 'global', 'MR_MTL', 'O_CDPFL']: # i.e. if no local clustering is done
    epsilons_train = epsilons_total
    if args.method == 'O_CDPFL':
        Gumbel_noise_scales = [0.0 for i in range(args.num_clients)]         
#----------------
    
if args.method == 'R_CDPFL':
    Z = [round(twostage_compute_z(epsilon=epsilons_train[i], dataset_size=num_samples[i], batch1=int(1*num_samples[i]), \
                     batch2=batches[i], local_epochs=args.num_local_epochs, stage1_global_epochs=1, \
                                  stage2_global_epochs=args.num_rounds - 1, delta=deltas_input[i]), 2) \
         for i in range(len(epsilons_train))]
    print("clients' noise scales are: {}".format(Z))
    
elif args.method in ['local', 'global', 'MR_MTL', 'f_CDPFL', 'O_CDPFL']:
    Z = [round(compute_z(epsilon=epsilons_train[i], dataset_size=num_samples[i], batch=batches[i], \
                         local_epochs=args.num_local_epochs, global_epochs=args.num_rounds, delta=deltas_input[i]), 2) \
         for i in range(len(epsilons_train))]
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
privacy_engines = [None for model in models]
    

if args.method == 'R_CDPFL':
    privacy_engines = [PrivacyEngine() for model in models]
    for j in range(len(models)):
        models[j], optimizers[j], train_loaders_fullbatch[j] = privacy_engines[j].make_private(
                module = models[j],
                optimizer = optimizers[j],
                data_loader = train_loaders_fullbatch[j],
                noise_multiplier = Z[j],
                max_grad_norm = args.max_per_sample_grad_norm)
elif args.method in ['local', 'global', 'MR_MTL', 'f_CDPFL', 'O_CDPFL']:
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
if args.method in ['f_CDPFL', 'O_CDPFL']:
    central_models = [copy.deepcopy(models[0]) for i in range(args.true_num_clusters)]
if args.method == 'MR_MTL':
    global_model = copy.deepcopy(models[0])  
    
model_path = output_dir  + f'/model_last.pth'
if os.path.exists(model_path):
    ckpt = torch.load(model_path, map_location='cpu')
    start_epoch = ckpt['epoch']
    if args.method == 'local':
        for c in range(0, args.num_clients):
            models[c].load_state_dict(ckpt['state_dict'][c])
    elif args.method == 'MR_MTL':
        for c in range(0, args.num_clients):
            models[c].load_state_dict(ckpt['state_dict'][c])
        global_model.load_state_dict(ckpt['state_dict'][-1])    
    elif args.method in ['FedAvg', 'global']:
        for model in models:
            # model.to(device)
            model.load_state_dict(ckpt['state_dict'])
    elif args.method in ['f_CDPFL', 'O_CDPFL']:
        for c in range(args.true_num_clusters):
            # model.to(device)
            central_models[c].load_state_dict(ckpt['state_dict'][c]) 
        probs = ckpt['probs']
    elif args.method == 'R_CDPFL':
        stage1_rounds = ckpt['stage1_rounds']
        M_opt = ckpt['M_opt']  
        print('stage1_rounds and M_opt are loaded and are equal to: {}, {}'.format(stage1_rounds, M_opt))
        central_models = [copy.deepcopy(models[0]) for i in range(M_opt)]  
        for c in range(M_opt):
            # model.to(device)
            central_models[c].load_state_dict(ckpt['state_dict'][c])
        probs = ckpt['probs']      
          
else:
    start_epoch = 0
    

if args.method == 'R_CDPFL' and start_epoch > 0: # i.e. we have already passed the first round (which uses full batch size)
    privacy_engines = [PrivacyEngine() for model in models]
    for j in range(len(models)):
        models[j], optimizers[j], train_loaders[j] = privacy_engines[j].make_private(
                module = models[j],
                optimizer = optimizers[j],
                data_loader = train_loaders[j],
                noise_multiplier = Z[j],
                max_grad_norm = args.max_per_sample_grad_norm)   
    
####
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
#### 

    
for t in range(start_epoch + 1, args.num_rounds+1):
    old_models = [copy.deepcopy(model) for model in models]
    #--------------------------
    if args.method in ['FedAvg']:
        for i in range(0, args.num_clients):
            individual_train(train_loaders[i], loss_func, optimizers[i], models[i], device=device, \
                             client_id=i, epochs=args.num_local_epochs, output_dir=output_dir, show=False, save=False)
        weights_aggregation = [1/args.num_clients for i in range(args.num_clients)]
        
    #------------------------------
    elif args.method in ['global']:
        for i in range(0, args.num_clients):
            individual_train_PDP(train_loaders[i], loss_func, optimizers[i], models[i], privacy_engines[i], \
                                 args.delta, device=device, client_id=i, epochs=args.num_local_epochs, \
                                 output_dir=output_dir, show=False, save=False)
        weights_aggregation = [1/args.num_clients for i in range(args.num_clients)]
    #------------------------------      
    elif args.method in ['local']:
        for i in range(0, args.num_clients):
            individual_train_PDP(train_loaders[i], loss_func, optimizers[i], models[i], privacy_engines[i], \
                                 args.delta, device=device, client_id=i, epochs=args.num_local_epochs, \
                                 output_dir=output_dir, show=False, save=False)    
    #------------------------------    
    elif args.method in ['MR_MTL']:
        for i in range(0, args.num_clients):
            individual_train_MR_MTL(train_loaders[i], loss_func, optimizers[i], models[i], copy.deepcopy(global_model), \
                                    args.lambda_MR_MTL, privacy_engines[i], args.delta, device=device, client_id=i, \
                                    epochs=args.num_local_epochs, output_dir=output_dir, show=False, save=False)
        weights_aggregation = [1/args.num_clients for i in range(args.num_clients)]
        update_model(global_model, models, weights=weights_aggregation)
        
    #----------------------------------------------------------------- 
    elif args.method in ['f_CDPFL', 'R_CDPFL', 'O_CDPFL']:      
        if t == 1:                
            if args.method == 'f_CDPFL':
                for i in range(0, args.num_clients):
                    individual_train_PDP(train_loaders[i], loss_func, optimizers[i], models[i], privacy_engines[i],
                                            args.delta, device=device, client_id=i,
                                            epochs=args.num_local_epochs, output_dir=output_dir, show=False, save=False)
                # in the second round (t=2) all clients contribute equally likely to all clusters:
                probs = np.ones((args.num_clients, args.true_num_clusters))/args.true_num_clusters   

            elif args.method in ['R_CDPFL']:
                for i in range(0, args.num_clients):
                    individual_train_PDP(train_loaders_fullbatch[i], loss_func, optimizers[i], models[i], privacy_engines[i],
                                            args.delta, device=device, client_id=i,
                                            epochs=args.num_local_epochs, output_dir=output_dir, show=False, save=False)
                delta_models = [find_delta_model(models[m], old_models[m].to(device)) for m in range(len(models))]
                M = models_to_matrix(deepcopy(delta_models))
                del delta_models
                n_pca = args.num_clients
                M_projected = PCA_proj(M.T, n_components=n_pca) # M_projected has shape (n_pca, args.num_clients)
                M_projected = M_projected[0:2,:] # we use the main 2 principal components for all clients
                if args.scaler_type == 'standard':
                    scaling = StandardScaler()
                elif args.scaler_type == 'minmax':
                    scaling = MinMaxScaler()                         
                scaling.fit(M_projected.T)
                M_projected_scaled = scaling.transform(M_projected.T).T
                
                
                # choosing M automatically:
                # candidate_Ms = [2,3,4,5,6,7,8,9,10]
                candidate_Ms = [args.true_num_clusters]
                scores = []
                overlaps = []
                for m in candidate_Ms:
                    probs, GMM_means, GMM_radii = GMM(M_projected_scaled, m, args.seed)
                    pairwise_sepscores, pairwise_overlaps = GMM_overlap(GMM_means, GMM_radii)
                    min_pairwise_sepscore = min(pairwise_sepscores)
                    max_pairwise_overlap = max(pairwise_overlaps)
                    scores.append(min_pairwise_sepscore)
                    overlaps.append(max_pairwise_overlap)
                    
                
                M_idx = np.argmax(np.array(scores))
                M_opt = candidate_Ms[M_idx]
                print('the number of clusters (M) was set to: {}'.format(M_opt))
                central_models = [copy.deepcopy(models[0]) for i in range(M_opt)]
                min_score = scores[M_idx]
                print('minimum pairwise separation score with this M: {}'.format(min_score))
                max_overlap = overlaps[M_idx]
                print('maximum pairwise overlap with this M: {}'.format(max_overlap))
                stage1_rounds = int((1 - max_overlap) * args.num_rounds / 2)
                print('first stage rounds is set to {}'.format(stage1_rounds))
                probs, GMM_means, GMM_radii = GMM(M_projected_scaled, M_opt, args.seed)

                
            # ------------------------------------------
            elif args.method == 'O_CDPFL':
                probs = clusters_to_probs(true_clusters, args.num_clients)
                # probs has shape (args.num_clients, args.true_num_clusters)

        elif t > 1:
            sampled_clusters = []
            for i in range(0, args.num_clients):
                client_probs = probs[i,:]
                c = sample_cluster(client_probs)
                sampled_clusters.append(c)
                assign_model(models[i], central_models[c])
                individual_train_PDP(train_loaders[i], loss_func, optimizers[i], models[i], privacy_engines[i],
                                        args.delta, device=device, client_id=i, 
                                        epochs=args.num_local_epochs, output_dir=output_dir, show=False, save=False)
            print('list of sampled clusters used during round {}: {}'.format(t, sampled_clusters))


                   
    
    if args.method in ['FedAvg', 'MR_MTL', 'local', 'global', 'KM_CDPFL']:
        if args.method in ['FedAvg', 'global']:
            aggregate(models, weights = weights_aggregation) # (for KM_CDPFL and MR_MTL local models are already updated and
            # 'local' does not have any aggregation of local models)
        
        ### Evaluation at the end of t-th round:
        accs_ = accuracies(models, val_loaders, device)
        losses_ = losses(models, train_loaders, loss_func, device)
        print('global epoch: {}'.format(t))
        mean, std = mean_std(accs_)
        print('mean validation accuracy: {}'.format(mean))
        print('mean train loss: {}'.format(np.mean(losses_)))
        print(f'validation accs: {[round(i, 3) for i in accs_]}')
        print(f'train losses: {round_list(losses_)}')
        ### saving loss values and accuracy values at the end of each round:
        accuracy_vals.append(accs_)
        loss_vals.append(losses_)
        with open(acc_file, 'wb') as f_out:
            pickle.dump(accuracy_vals, f_out)
        with open(loss_file, 'wb') as f_out:
            pickle.dump(loss_vals, f_out) 

                
        ### saving the model at the end of t-th round:
        if t % args.save_epoch == 0:
            if args.method in ['local', 'KM_CDPFL']: # save all clients' models one by one
              torch.save({'epoch': t, 'state_dict': [models[i].state_dict() for i in range(args.num_clients)]}, output_dir  + \
                         f'/model_last.pth')
            elif args.method == 'MR_MTL': # save all clients' models as well as the global model
              torch.save({'epoch': t, 'state_dict': [models[i].state_dict() for i in range(args.num_clients)]
                          + [global_model.state_dict()]}, output_dir + f'/model_last.pth')
            else:
              torch.save({'epoch': t, 'state_dict': models[0].state_dict()}, output_dir  + f'/model_last.pth')
               
      
    
    elif args.method in ['f_CDPFL', 'R_CDPFL', 'O_CDPFL']:
        if t==1 and args.method == 'R_CDPFL':
            # update the batch size for the rest of the rounds (t>1):
            privacy_engines = [PrivacyEngine() for model in models]
            for j in range(0, args.num_clients):
                models[j], optimizers[j], train_loaders[j] = privacy_engines[j].make_private(module = models[j], \
                                                                                             optimizer = optimizers[j], \
                                                                                             data_loader = train_loaders[j], \
                                                                                             noise_multiplier = Z[j], \
                                                                                             max_grad_norm = \
                                                                                             args.max_per_sample_grad_norm)
        elif t>1:
            # update the central models
            if args.method in ['f_CDPFL', 'O_CDPFL']:
                for c in range(args.true_num_clusters):
                    models_c = [models[i] for i in range(len(models)) if sampled_clusters[i]==c]
                    weights_c = np.array([1 for i in range(len(models)) if sampled_clusters[i]==c])
                    weights_c = list(weights_c/np.sum(weights_c))
                    if len(models_c) > 0:
                        update_model(central_models[c], models_c, weights=weights_c)
            elif args.method == 'R_CDPFL':
                for c in range(M_opt):
                    models_c = [models[i] for i in range(len(models)) if sampled_clusters[i]==c]
                    weights_c = np.array([1 for i in range(len(models)) if sampled_clusters[i]==c])
                    weights_c = list(weights_c/np.sum(weights_c))
                    if len(models_c) > 0:
                        update_model(central_models[c], models_c, weights=weights_c)
            ### Evaluation at the end of t-th round (t>=2):                    
                    
            
            if args.method == 'f_CDPFL' and t < local_clustering_rounds + 1:
                accs_, losses_, best_clusters = cluster_accuracies_losses(central_models, sampled_clusters, train_loaders,
                                                                     val_loaders, loss_func, device, Gumbel_noise_scales,
                                                                         compute_best_clusters=True)
                probs = clusters_to_probs(best_clusters, args.num_clients) # update the sampling probs with the "best_clusters"
            elif args.method == 'f_CDPFL':
                accs_, losses_, _ = cluster_accuracies_losses(central_models, sampled_clusters, train_loaders, val_loaders,\
                                                              loss_func, device, Gumbel_noise_scales)
                                
            if args.method == 'R_CDPFL' and t in list(np.arange(stage1_rounds, stage1_rounds + local_clustering_rounds + 1)):
                accs_, losses_, best_clusters = cluster_accuracies_losses(central_models, sampled_clusters, train_loaders,
                                                                     val_loaders, loss_func, device, Gumbel_noise_scales,
                                                                         compute_best_clusters=True)
                probs = clusters_to_probs(best_clusters, args.num_clients) # update the sampling probs with the "best_clusters"
            elif args.method == 'R_CDPFL':
                accs_, losses_, _ = cluster_accuracies_losses(central_models, sampled_clusters, train_loaders, val_loaders,\
                                                              loss_func, device, Gumbel_noise_scales)
            
            if args.method == 'O_CDPFL':
                accs_, losses_, _ = cluster_accuracies_losses(central_models, sampled_clusters, train_loaders, val_loaders,\
                                                              loss_func, device, Gumbel_noise_scales)
                
            
            print('at the end of global epoch {}:'.format(t))
            mean, std = mean_std(accs_)
            print('mean validation accuracy: {}'.format(mean))
            print('mean train loss: {}'.format(np.mean(losses_)))
            print(f'validation accs: {[round(i, 3) for i in accs_]}')
            print(f'train losses: {round_list(losses_)}')
            ## saving loss values and accuracy values at the end of each round:
            accuracy_vals.append(accs_)
            loss_vals.append(losses_)
            with open(acc_file, 'wb') as f_out:
                pickle.dump(accuracy_vals, f_out)
            with open(loss_file, 'wb') as f_out:
                pickle.dump(loss_vals, f_out)
                    
            
            
            ### saving the central models at the end of t-th round:
            if t % args.save_epoch == 0:
                if args.method == 'R_CDPFL':
                    torch.save({'epoch': t, 'state_dict': [central_models[c].state_dict() for c in
                                                           range(M_opt)], 'probs': probs, 'stage1_rounds':
                                                           stage1_rounds, 'M_opt': M_opt}, output_dir  + 
                                                           f'/model_last.pth')
                else:
                    torch.save({'epoch': t, 'state_dict': [central_models[c].state_dict() for c in
                                                           range(args.true_num_clusters)], 'probs': probs}, output_dir  +
                                                           f'/model_last.pth')
                       
 
print('----------------------------')
print('Evaluation on the test sets at the end of global epoch {}:'.format(t))
if args.method in ['FedAvg', 'MR_MTL', 'local', 'global', 'KM_CDPFL']:
    accs_ = accuracies(models, test_loaders, device)
    mean, std = mean_std(accs_)
    print('mean test accuracy: {}'.format(mean))
    print(f'test accs: {[round(i, 3) for i in accs_]}')         


elif args.method in ['f_CDPFL', 'R_CDPFL', 'O_CDPFL']:
    
    accs_, losses_, _ = cluster_accuracies_losses(central_models, sampled_clusters, train_loaders, test_loaders, loss_func,\
                                                  device, Gumbel_noise_scales)
    mean, std = mean_std(accs_)
    print('mean test accuracy: {}'.format(mean))
    print(f'test accs: {[round(i, 3) for i in accs_]}')
    