"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

import pandas as pd
import geopandas as gpd
from scipy.spatial import Delaunay
import random

def load_data(args):
    if args.task == 'nc':
        data = load_data_nc()
        data['adj_train_norm'], data['features'] = process(
            data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
    )

    return data


# ############### FEATURES PROCESSING ####################################


def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


# ############### DATA SPLITS #####################################################


def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)  


def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


# ############### LINK PREDICTION DATA LOADERS ####################################


def load_data_lp(dataset, use_feats, data_path):
    if dataset in ['cora', 'pubmed', 'shapefile']:
        adj, features = load_citation_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'disease_lp':
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'airport':
        adj, features = load_data_airport(dataset, data_path, return_label=False)
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    data = {'adj_train': adj, 'features': features}
    return data


# ############### NODE CLASSIFICATION DATA LOADERS ####################################


def load_data_nc():
    adj, features, labels, idx_train, idx_val, idx_test = load_citation_data()

    labels = torch.LongTensor(labels)
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
    return data


# ############### DATASETS ####################################


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def adjacency(path, TAZ_Points):  # 由TAZ面生成的内部点表示
    # build graph
    print("Loading {} dataset...".format(path))
    file = gpd.read_file(os.path.join(path, TAZ_Points))
    point_XY = file.values[:, 2:4]  # 获取xy值,根据不同情况调节[:,xx]
    point_size = point_XY.shape[0]  # 节点数量
    tri = Delaunay(point_XY)
    relation = np.array(tri.simplices)
    adjacency = np.zeros((point_size, point_size), dtype=np.int)  # 生成point_size * point_size维数据
    for i in range(len(relation)):
        Num_1 = relation[i, 0]
        Num_2 = relation[i, 1]
        Num_3 = relation[i, 2]
        adjacency[Num_1, Num_2] = adjacency[Num_2, Num_1] = 1

        adjacency[Num_2, Num_3] = adjacency[Num_3, Num_2] = 1

        adjacency[Num_1, Num_3] = adjacency[Num_3, Num_1] = 1

    # 获取边连接点对,格式[[0,0,0,1,1,1,2,2,2,3,3,3], [1,2,3,0,2,3,0,1,3,0,1,2]]
    connection = np.zeros((2, sum(sum(adjacency))), dtype=np.int)

    count = 0
    for i in range(adjacency.shape[0]):
        for j in range(adjacency.shape[1]):
            if adjacency[i, j] == 1:
                connection[0][count] = i
                connection[1][count] = j
                count += 1
    # 转换成为torch.tensor
    connection = torch.from_numpy(connection)

    # adjacency转稀疏矩阵
    adjacency = sp.coo_matrix(adjacency)

    # 返回数据为元组形式
    return adjacency, connection


def data_Genearte(path, file_1, file_2):
    # file_1 street_view文件
    print("Loading {} dataset...".format(file_1))

    view_features_label = gpd.read_file(os.path.join(path, file_1))
    _ = view_features_label.groupby("id").agg([np.sum])  # 获取street_view特征
    for i in range(_.shape[0]):
        _.iloc[i, -1] = _.iloc[i, -1][0]  # 获取label数据
    view_features_label = _.iloc[:, 1:]  # _数据的第一列为Object，最后一列为label，中间列为features，view_features从第一列开始取值

    print("Loading {} dataset...".format(file_2))
    _ = gpd.read_file(os.path.join(path, file_2))
    poi_features = _.iloc[:, 1:-1]
    poi_count = poi_features.groupby(["id", "type_class"]).count()
    length_count = poi_count._stat_axis.values.tolist()
    poi_features = pd.DataFrame(np.zeros((2097, 128), dtype=np.int))  # view_features_label与poi_features开始索引名值差1
    result = pd.concat([view_features_label, poi_features], axis=1)  # 按列合并view_features_label和poi_features
    result = result[~result[result.keys()[0]].isin(['nan'])]  # 移除'nan'行
    for i in length_count:
        result.iloc[i[0] - 1, int(i[1]) - 129] = poi_count.iloc[length_count.index(i), 0]
    result.insert(result.shape[1] - 1, 'label', result.pop(result.keys()[-129]))  # 将label换到最后一列

    return result


def load_citation_data(path="./data/shapefile/", dataset_view="Final_Result_Features.shp", poi="feature_poi.shp",
              TAZ_Points="TAZ_Points.shp"):
    """Load citation network dataset"""

    _ = np.array(data_Genearte(path, dataset_view, poi), dtype=np.dtype(np.float64))  # _暂存street_view特征
    idx_features_labels = _

    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float64)
    labels = encode_onehot(idx_features_labels[:, -1])

    """Build graph"""
    # adj需为对称矩阵,该条件在adjacency()函数中生成
    adj_all = adjacency(path, TAZ_Points)
    adj = adj_all[0]  # 获取邻接信息
    connection = adj_all[1]  # 获取边对信息
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # 分别对特征矩阵features和邻接矩阵adj做标准化,normalize()方法
    features = normalize(features)

    # 标准化之前加入自环
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # 分别构建训练集、验证集、测试集，并创建特征矩阵、标签向量和临界矩阵的tensor
    # 此处所有样本参加训练、验证以及测试
    # idx_train = list(random.sample(range(0, adj.shape[0]), 1000))
    # idx_val = list(random.sample(range(0, adj.shape[0]), 1000))
    # idx_test = list(random.sample(range(0, adj.shape[0]), 1000))
    idx_train = range(0, adj.shape[0])
    idx_val = range(0, adj.shape[0])
    idx_test = range(0, adj.shape[0])

    # 输出格式tensor
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    # 此处不对adj进行处理,处理部分在load_data中进行
    # adj = sparse_mx_to_torch_sparse_tensor(adj).to_dense()

    # 此处的adj可以理解为边权重邻接矩阵
    return adj, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels


def load_data_airport(dataset_str, data_path, return_label=False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph.node[u]['feat'] for u in graph.nodes()])
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0/7, 8.0/7, 9.0/7])
        return sp.csr_matrix(adj), features, labels
    else:
        return sp.csr_matrix(adj), features

