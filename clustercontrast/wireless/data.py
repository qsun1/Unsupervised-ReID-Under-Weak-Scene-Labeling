from __future__ import print_function, absolute_import
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import normalize


def np_filter(arr, *arg):
    temp_arr = arr
    for i_axis, axis_i in enumerate(arg):
        map_list = []
        for i_elem, elem_i in enumerate(axis_i):
            temp_elem_arr = temp_arr[temp_arr[:, i_axis] == elem_i]
            map_list.append(temp_elem_arr)
        temp_arr = np.concatenate(map_list, axis=0)
    return temp_arr
    
def normalize_2d(mx):
    """Row-normalize matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def cosine_distance(x, y):
    if torch.is_tensor(x):
        assert torch.is_tensor(y), type(y)
        assert x.dim() == 2, x.dim()
        assert y.dim() == 2, y.dim()
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        dst = 1 - x.mm(y.t())
        return dst
    elif isinstance(x, np.ndarray):
        assert isinstance(y, np.ndarray), type(y)
        assert x.ndim == 2, x.ndim
        assert y.ndim == 2, y.ndim
        x = normalize(x, norm='l2', axis=1)
        y = normalize(y, norm='l2', axis=1)
        return 1 - x.dot(y.T)
    else:
        raise TypeError(type(x))


def feadict_to_videofea(fea, info):
    feature = []
    for video_info in info:
        video_fea = [fea[k] for k in video_info[0]]
        video_fea = torch.stack(video_fea).mean(0)
        feature.append(video_fea)
    return torch.stack(feature)


def video_fea_to_feadict(fea, info):
    fea = fea.cpu()
    feature = {}
    for i, video_info in enumerate(info):
        for f in video_info[0]:
            feature[f] = fea[i]
    return feature
