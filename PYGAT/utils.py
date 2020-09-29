import numpy as np
import scipy.sparse as sp
import os
import torch
import pandas as pd
import geopandas as gpd
from scipy.spatial import Delaunay


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def adjacency(path):  # 由TAZ面生成的内部点表示
    # build graph
    print("Loading {} dataset...".format(path))
    file = gpd.read_file(os.path.join(path))
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

    # adjacency转稀疏矩阵
    adjacency = sp.coo_matrix(adjacency)

    # 返回数据为元组形式
    return adjacency


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


def load_data(path="./data/shapefile/", dataset_view="Final_Result_Features.shp", poi="feature_poi.shp",
              TAZ_Points="TAZ_Points.shp"):
    """Load citation network dataset"""

    _ = np.array(data_Genearte(path, dataset_view, poi), dtype=np.dtype(np.float64))  # _暂存street_view特征
    idx_features_labels = _

    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float64)
    labels = encode_onehot(idx_features_labels[:, -1])

    """Build graph"""
    # adj需为对称矩阵,该条件在adjacency()函数中生成
    adj = adjacency(os.path.join(path, TAZ_Points))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # 分别对特征矩阵features和邻接矩阵adj做标准化,normalize()方法
    features = normalize_features(features)

    # 标准化之前加入自环
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    # 分别构建训练集、验证集、测试集，并创建特征矩阵、标签向量和临界矩阵的tensor
    # 此处所有样本参加训练、验证以及测试
    idx_train = range(adj.shape[0])
    idx_val = range(adj.shape[0])
    idx_test = range(adj.shape[0])

    # 输出格式tensor
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj).to_dense()

    # 输出格式tensor
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # 此处的adj可以理解为边权重邻接矩阵
    return adj, features, labels, idx_train, idx_val, idx_test

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
