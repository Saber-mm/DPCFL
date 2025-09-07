#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch.utils import data
from torchvision import datasets, transforms
from utils.sampling import iid, clustered_iid
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.transforms.functional import rotate
# import torchvision.transforms.RandomRotation as rotate





class DatasetSplit_augmented(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, tau):
        dataset_aug = copy.deepcopy(dataset)
        if tau > 1:
            for i in range(tau-1):
                dataset_aug = ConcatDataset([dataset_aug, dataset])
        self.dataset = dataset_aug   

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        
        
        return torch.tensor(image), torch.tensor(label)
    
    
class DatasetSplit2_augmented(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, tau, cluster, num_classes=10):
        dataset_aug = copy.deepcopy(dataset)
        if tau > 1:
            for i in range(tau-1):
                dataset_aug = ConcatDataset([dataset_aug, dataset])
        self.dataset = dataset_aug
        self.cluster = cluster
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        label = (label + int(self.cluster)) % 10
    
        return torch.tensor(image), torch.tensor(label)
    

class DatasetSplit3_augmented(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, tau, cluster, num_clusters, num_classes=10):
        dataset_aug = copy.deepcopy(dataset)
        if tau > 1:
            for i in range(tau-1):
                dataset_aug = ConcatDataset([dataset_aug, dataset])
        self.dataset = dataset_aug
        self.cluster = cluster
        self.num_clusters = num_clusters
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        image = rotate(image, self.cluster * (360/self.num_clusters))
    
        return torch.tensor(image), torch.tensor(label)
    
    
    

    
    
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        
        
        return torch.tensor(image), torch.tensor(label)
    
    
class DatasetSplit2(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs, cluster, num_classes=10):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.cluster = cluster
        self.num_classes = num_classes

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        label = (label + int(self.cluster)) % 10
    
        return torch.tensor(image), torch.tensor(label)
    

class DatasetSplit3(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs, cluster, num_clusters, num_classes=10):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.cluster = cluster
        self.num_clusters = num_clusters
        self.num_classes = num_classes

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        image = rotate(image, self.cluster * (360/self.num_clusters))
    
        return torch.tensor(image), torch.tensor(label)
    
    
    

def get_dataset_cluster_split(dataset='MNIST', num_users=20, num_minority_users=2, num_clusters=4, user_max_class=10, p_pure=0.9,
                              seed=0, num_samples_per_client=8000):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if dataset == 'CIFAR10':
        data_dir = '../data/CIFAR10'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)
        
        user_groups, true_clusters, clusters_idx = clustered_iid(train_dataset, num_users, num_minority_users, num_clusters,\
                                                                 seed = seed, num_items = num_samples_per_client)
        user_groups_test, _, _ = clustered_iid(test_dataset, num_users, num_minority_users, num_clusters, seed = seed)
        

        
            
    elif dataset == 'CIFAR100':
        data_dir = '../data/CIFAR100'
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=transforms.ToTensor())

        user_groups, true_clusters, clusters_idx = clustered_iid(train_dataset, num_users, num_minority_users, num_clusters,\
                                                                 seed = seed, num_items = num_samples_per_client)
        user_groups_test, _, _ = clustered_iid(test_dataset, num_users, num_minority_users, num_clusters, seed = seed)
    
    
    elif dataset == 'MNIST' or dataset == 'FMNIST':
        
        if dataset == 'MNIST':
            data_dir = '../data/MNIST'
            
            apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
            
            train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
            test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)
            
            
        else:
            data_dir = '../data/FMNIST'
            
            apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
            
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=apply_transform)
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=apply_transform)
        

        user_groups, true_clusters, clusters_idx = clustered_iid(train_dataset, num_users, num_minority_users, num_clusters,\
                                                                 seed = seed, num_items = num_samples_per_client)
        user_groups_test, _, _ = clustered_iid(test_dataset, num_users, num_minority_users, num_clusters, seed = seed)

    return train_dataset, test_dataset, user_groups, user_groups_test, true_clusters, clusters_idx






