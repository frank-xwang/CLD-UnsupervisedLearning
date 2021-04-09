import math
import time
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
# from pykeops.torch import LazyTensor

use_cuda = torch.cuda.is_available()
dtype = 'float32' if use_cuda else 'float64'
torchtype = {'float32': torch.float32, 'float64': torch.float64}

def embeddings_to_cosine_similarity(E, sigma=1.0):
    '''
    Build a pairwise symmetrical cosine similarity matrix
    diganal is set as zero
    '''

    dot = torch.abs_(torch.mm(E, E.t())) # E[i]E[j]
    norm = torch.norm(E, 2, 1) # ||E[i]||
    x = torch.div(dot, norm) # E[i]E[j]/||E[j]||
    x = torch.div(x, torch.unsqueeze(norm, 0)) # E[i]E[j]/(||E[j]||*||E[i]||)
    x = x.div_(sigma)

    return torch.max(x, x.t()).fill_diagonal_(0)

def pairwise_cosine_similarity(E, C, temp=1.0):
    '''
    Build a pairwise cosine similarity matrix.
    '''
    dot = torch.abs_(torch.mm(E, C.t())) # E[i]C[j]
    # dot = torch.mm(E, C.t()) # E[i]C[j]
    # print(dot.size())
    E = torch.norm(E, 2, 1) # ||E[j]||
    # print(E.size(), norm_E.size())
    C = torch.norm(C, 2, 1) # ||C[i]||
    # print(C.size(), norm_C.size())
    x = torch.div(dot, E.view(-1, 1)) # E[i]E[j]/||E[j]||
    x = torch.div(x, C.view(1, -1)) # E[i]E[j]/(||E[j]||*||E[i]||)
    return x.div_(temp)

def kway_normcuts(F, K=2, sigma=1.0):
    # Build similarity matrix W, use cosine similarity
    W = embeddings_to_cosine_similarity(F, sigma=sigma)

    # Build defree matrix
    degree = torch.sum(W, dim=0)

    # Construct normalized Laplacian matrix L
    D_pow = torch.diag(degree.pow(-0.5))
    L = torch.matmul(torch.matmul(D_pow, torch.diag(degree)-W), D_pow)

    # Get eigenvectors with torch.symeig()
    _, eigenvector = torch.symeig(L, eigenvectors=True)

    # Normalize eigenvector along each row
    eigvec_norm = torch.matmul(torch.diag(degree.pow(-0.5)), eigenvector)
    eigvec_norm = eigvec_norm/eigvec_norm[0][0]
    k_eigvec = eigvec_norm[:,:K]

    return k_eigvec

def KMeans(x, K=10, Niters=10, verbose=False):
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # K-means loop:
    # - x  is the point cloud,
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    # start = time.time()
    c = x[:K, :].clone()  # Simplistic random initialization
    # x_i = LazyTensor(x[:, None, :])  # (Npoints, 1, D)
    x_i = x[:, None, :]  # (Npoints, 1, D)

    for i in range(Niters):
        c_j = c[None, :, :]  # (1, Nclusters, D)
        # c_j = LazyTensor(c[None, :, :])  # (1, Nclusters, D)
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # Ncl = torch.bincount(cl).type(torchtype[dtype])  # Class weights
        # for d in range(D):  # Compute the cluster centroids with torch.bincount:
        #     c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl
        # print(c)
        Ncl = cl.view(cl.size(0), 1).expand(-1, D)
        unique_labels, labels_count = Ncl.unique(dim=0, return_counts=True)
        # As some clusters don't contain any samples, manually assign count as 1
        labels_count_all = torch.ones([K]).long().cuda()
        labels_count_all[unique_labels[:,0]] = labels_count
        c = torch.zeros([K, D], dtype=torch.float).cuda().scatter_add_(0, Ncl, x)
        c = c / labels_count_all.float().unsqueeze(1)

    return cl, c

def spectral_clustering(F, K=10, clusters=10, Niters=10, sigma=1):
    '''
    Input:
        Sample features F, N x D
        K: Number of eigenvectors for K-Means clustering
        clusters: number of K-Means clusters
        Niters: NUmber of iterations for K-Means clustering
    Output:
        cl: cluster label for each sample, N x 1
        c: centroids of each cluster, clusters x D
    '''
    # Get K eigenvectors with K-way normalized cuts 
    k_eigvec = kway_normcuts(F, K=K, sigma=sigma)

    #  Spectral embedding with K eigen vectors
    cl, _ = KMeans(k_eigvec, K=clusters, Niters=Niters, verbose=False)

    # Get unique labels and samples numbers of each cluster
    Ncl = cl.view(cl.size(0), 1).expand(-1, F.size(1))
    unique_labels, labels_count = Ncl.unique(dim=0, return_counts=True)

    # As some clusters don't contain any samples, manually assign count as 1
    labels_count_all = torch.ones([clusters]).long().cuda()
    labels_count_all[unique_labels[:,0]] = labels_count

    # Calcualte feature centroids
    c = torch.zeros([clusters, F.size(1)], dtype=torch.float).cuda().scatter_add_(0, Ncl, F)
    c = c / labels_count_all.float().unsqueeze(1)

    return cl, c

def test_spectral_clustering():
    SIGMA=1
    cluster = 40
    Niters = 40
    time_list = []
    for K in [40]:
        iters = 100
        F = torch.randn(1024, 128).float().cuda()
        for i in range(iters+1):
            if i == 0:
                time_s = time.time()
            # Test spectral clustering
            cl, c = spectral_clustering(F, K=K, clusters=cluster, Niters=Niters, sigma=SIGMA)
        time_list.append((time.time()-time_s)/iters)
        print('Total time cost is {} when K is {}, # cluster is {} and \
            Niters is {}'.format(time.time()-time_s, K, cluster, Niters))
        print('Avg time cost is: ', (time.time()-time_s)/iters)
    print(time_list)

# test_spectral_clustering()