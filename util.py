import community
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
def node_iter(G):
    if float(nx.__version__)<2.0:
        return G.nodes()
    else:
        return G.nodes

def node_dict(G):
    if float(nx.__version__)>2.1:
        node_dict = G.nodes
    else:
        node_dict = G.node
    return node_dict

def adj_process(adjs):
    g_num, n_num, n_num = adjs.shape
    adjs = adjs.detach()
    for i in range(g_num):
        adjs[i] += torch.eye(n_num).cuda()
        adjs[i][adjs[i]>0.] = 1.
        degree_matrix = torch.sum(adjs[i], dim=-1, keepdim=False)
        degree_matrix = torch.pow(degree_matrix,-1)
        degree_matrix[degree_matrix == float("inf")] = 0.
        degree_matrix = torch.diag(degree_matrix)
        adjs[i] = torch.mm(degree_matrix, adjs[i])
    return adjs


def NormData(adj):
    adj=adj.tolist()
    adj_norm = normalize_adj(adj )
    adj_norm = adj_norm.toarray()
    #adj = adj + sp.eye(adj.shape[0])
    #adj = adj.toarray()
    #feat = feat.toarray()
    return adj_norm



def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)