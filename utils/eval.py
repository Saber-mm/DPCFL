import torch
from tqdm import tqdm
import numpy as np
import copy as copy

def accuracy(model, loader, device, show=True):
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        if show:
            t = tqdm(loader)
        else:
            t = loader
        for images, target in t:
            images = images.to(device)
            target = target.to(device)
            correct += (model(images).argmax(1) == target).sum().item()
            total += target.numel()
            acc = correct / total
            if show:
                t.set_description(f'test acc: {acc*100:.2f}%')
    return acc * 100


def loss(model, loader, loss_fn, device, show=True):
    loss_total = 0
    total = 0
    model.to(device)
    
    with torch.no_grad():
        if show:
            t = tqdm(loader)
        else:
            t = loader
        for images, target in t:
            images = images.to(device)
            target = target.to(device)
            #target = torch.nn.functional.one_hot(target, num_classes=10).type(torch.cuda.FloatTensor)
            if images.shape[0] == 0: # i.e. empty batch
                break
            outputs = model(images).to(device)
            loss_total += loss_fn(outputs, target) * len(target)
            total += len(target)
        
        loss_avg = loss_total / total
    return loss_avg.item()


def accuracies(models, loaders, device):
    num_clients = len(loaders)
    accs = []
    for i in range(num_clients):
        model, loader = models[i], loaders[i]
        acc = accuracy(model, loader, device, show=False)
        accs.append(acc)
    return np.array(accs)

def losses(models, loaders, loss_fn, device):
    num_clients = len(models)
    losses_ = []
    for i in range(num_clients):
        model, loader = models[i], loaders[i]
        loss_ = loss(model, loader, loss_fn, device, show=False)
        losses_.append(loss_)
    return np.array(losses_)


# def cluster_accuracies_losses(models, sampled_clusters, train_loaders, test_loaders, loss_fn, device):
    
#     num_clients = len(test_loaders)
#     num_clusters = len(models)

#     accs = np.zeros((num_clients, num_clusters))
#     for i in range(num_clients):
#         for c in range(num_clusters):
#             model = copy.deepcopy(models[c])
#             loader = copy.deepcopy(train_loaders[i])
#             acc = accuracy(model, loader, device, show=False)
#             accs[i, c] = acc
#     best_accs = np.max(accs, axis=1)
#     idxs = list(np.argmax(accs, axis=1))
#     print('The best models for clients are: {}'.format(idxs))

#     losses_ = []
#     for i in range(num_clients):
#         model = copy.deepcopy(models[idxs[i]])
#         loader = copy.deepcopy(test_loaders[i])
#         loss_ = loss(model, loader, loss_fn, device, show=False)
#         losses_.append(loss_)
    
#     return best_accs, np.array(losses_)





def cluster_accuracies_losses(models, sampled_clusters, train_loaders, evaluation_loaders, loss_fn, device, noise_scales,\
                              compute_best_clusters=False):
    
    best_clusters = []
    num_clients = len(evaluation_loaders)
    if compute_best_clusters == True:
        num_clusters = len(models)
        losses = np.zeros((num_clients, num_clusters))
        accuracies = np.zeros((num_clients, num_clusters))
        for i in range(num_clients):
            for c in range(num_clusters):
                model = copy.deepcopy(models[c])
                loader = copy.deepcopy(train_loaders[i])
                loss_ = loss(model, loader, loss_fn, device, show=False)
                losses[i, c] = loss_
                acc_ = accuracy(model, loader, device, show=False)
                accuracies[i, c] = acc_ + np.random.gumbel(0, noise_scales[i], 1)
        idxs = list(np.argmax(accuracies, axis=1)) # model selection for clients based on noisy "cluster accruacies"
        print('The best clusters at the end of the round are: {}'.format(idxs))
        
        for c in range(num_clusters):
            cluster = [i for i in range(num_clients) if idxs[i] == c]
            best_clusters.append(cluster)  
    
    
    # Now, evaluating "validation set" performance of clients on their corresponding cluster model, used in the passed round:
    accs_ = []
    losses_ = []
    for i in range(num_clients):
        model = copy.deepcopy(models[sampled_clusters[i]])
        train_loader = copy.deepcopy(train_loaders[i])
        val_loader = copy.deepcopy(evaluation_loaders[i])
        loss_ = loss(model, train_loader, loss_fn, device, show=False)
        losses_.append(loss_)
        acc = accuracy(model, val_loader, device, show=False)
        accs_.append(acc)
        
            
    return np.array(accs_), losses_, best_clusters



def randomize_response(clusters, num_clusters, p):
    
    randomized_clusters = copy.deepcopy(clusters)
    randomized_clusters = [i if np.random.rand(1)[0] > p else np.random.choice(num_clusters) for i in clusters]
    
    return randomized_clusters
    
    
    
def losses(models, loaders, loss_fn, device):
    num_clients = len(models)
    losses_ = []
    for i in range(num_clients):
        model, loader = models[i], loaders[i]
        loss_ = loss(model, loader, loss_fn, device, show=False)
        losses_.append(loss_)
    return np.array(losses_)



def epsilons(priv_engines, delta):
    num_clients = len(priv_engines)
    epsilons_ = []
    for i in range(num_clients):
        epsilon = priv_engines[i].accountant.get_epsilon(delta=delta)
        epsilons_.append(round(epsilon, 3))
    return epsilons_    
        