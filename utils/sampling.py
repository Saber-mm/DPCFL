#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
import random
import copy
from torchvision import datasets, transforms



def clustered_iid(dataset, num_users, num_minority_users, num_clusters, seed=0, num_items = None):
    """
    Sample I.I.D. client data from "dataset"
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    np.random.seed(seed)
    num_majority_users = int((num_users - num_minority_users)/(num_clusters-1))
    if num_items == None:
        num_items = int(len(dataset)/num_majority_users)
    
    
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        #if i == num_minority_users or (i-num_minority_users) % num_majority_users == 0:
        #   all_idxs = [i for i in range(len(dataset))]
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        # all_idxs = list(set(all_idxs) - dict_users[i])
    
    clusters_idx = [num_clusters-1] * num_users
    c = 0
    for i in range(num_users):
        if i == num_minority_users or (i-num_minority_users) % num_majority_users == 0:
            c = c+1
        clusters_idx[i] = c

    true_clusters = []
    for c in range(num_clusters):
        cluster = [client for client in range(num_users) if clusters_idx[client]==c]
        true_clusters.append(cluster)
        
    return dict_users, true_clusters, clusters_idx




def oneclient_iid(dataset, seed=0):
    """
    Sample I.I.D. client data from "dataset"
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    np.random.seed(seed)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    dict_users[0] = all_idxs
        
    return dict_users



def noniid(dataset, num_users, user_max_class):
    """
    Sample non-I.I.D client data from "dataset"
    :param dataset:
    :param num_users:
    :param user_max_class:
    :return:
    """
    np.random.seed(0)
    
    if user_max_class == 1:
        num_shards, num_imgs, num_shards_per_user = 20*1, 3000, 1
    elif user_max_class == 2:
        num_shards, num_imgs, num_shards_per_user = 20*2, 1500, 2
    elif user_max_class == 4:
        num_shards, num_imgs, num_shards_per_user = 20*4, 750, 5
    elif user_max_class == 5:
        num_shards, num_imgs, num_shards_per_user = 20*5, 600, 5
    elif user_max_class == 6:
        num_shards, num_imgs, num_shards_per_user = 20*6, 500, 6
    elif user_max_class == 8:
        num_shards, num_imgs, num_shards_per_user = 20*8, 375, 8
    elif user_max_class == 10:
        num_shards, num_imgs, num_shards_per_user = 20*10, 300, 10
    else:
        exit('Error: is not implemented')


    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 6 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_shards_per_user, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    for i in range(num_users):
        np.random.shuffle(dict_users[i])
    for i in range(num_users):
        values, counts = np.unique(labels[list(map(int, list(dict_users[i])))], return_counts=True)
        print(len(values), values, counts)

    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users