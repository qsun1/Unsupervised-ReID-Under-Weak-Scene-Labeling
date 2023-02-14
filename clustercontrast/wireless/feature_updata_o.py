import torch
import numpy as np

from ..utils import to_torch
from .wireless_info_explore_o import mac_info_explore
from .data import cosine_distance, feadict_to_videofea, video_fea_to_feadict
from .tracklet_association import nearest_neighbour_association


def direct_nns(feature, dataset):
    feature = feadict_to_videofea(feature, dataset.extra_info['train'])
    train_info = dataset.extra_info['train_info'].copy()
    feature = to_torch(feature).cuda()
    fea_dist_mat = cosine_distance(feature, feature).cpu().numpy()
    new_train_info, _ = nearest_neighbour_association(train_info.copy(), fea_dist_mat, k=None)
    return get_pseudo_label(train_info.copy(), new_train_info, dataset.extra_info['train']), get_video_label(train_info.copy(), new_train_info)

def get_video_label(raw_info, new_info):
    raw_idx_list = raw_info[:,2].tolist() # (868, 5); start frame of video sequences
    new_idx_list = new_info[:,2].tolist() # (281, 5)
    output = []
    for i, idx in enumerate(raw_idx_list):
        if idx in new_idx_list:
            pid = new_info[new_idx_list.index(idx), 0]
        else:
            pid = -1    # w / o reciprocal Nearest Neighbor
        output.append(pid)
    label = np.asarray(output)
    assert len(label) == raw_info.shape[0]
    return label # pseudo labels for all imgs


def get_pseudo_label(raw_info, new_info, video_info):
    assert raw_info.shape[0] == len(video_info)
    raw_idx_list = raw_info[:,2].tolist()
    new_idx_list = new_info[:,2].tolist()
    outpout = []
    for i, idx in enumerate(raw_idx_list):
        if idx in new_idx_list:
            pid = new_info[new_idx_list.index(idx), 0]
        else:
            pid = -1
        for f_dir in video_info[i][0]:
            outpout.append((f_dir, pid))
    label = np.asarray([f[1] for f in sorted(outpout)])
    return label
        
def fea_update_by_wireless(train_ass_mat, mac_cluster_num, feature, dataset, seed, k, visual_labels):
    # dist_mat weight
    feature = feadict_to_videofea(feature, dataset.extra_info['train'])
    train_info = dataset.extra_info['train_info'].copy()
    video_ass, acc_cluster,  acc_ass = mac_info_explore(train_ass_mat.copy(), train_info.copy(), feature.numpy(), mac_cluster_num, seed)

    video_ass_new = np.sort(video_ass, axis=2)[:,:,-15:]
    video_ass_new = video_ass
    wireless_dist_mat = 1 - np.average(video_ass_new, axis=2)
    feature = to_torch(feature).cuda()
    #fea_dist_mat = cosine_distance(feature, feature).cpu().numpy()
    N = feature.shape[0]
    fea_dist_mat = np.zeros((N, N))
    for i in range(N):
        if visual_labels[i] != -1:
            fea_dist_mat[i, np.where(visual_labels == visual_labels[i])] = 1
        fea_dist_mat[i, i] = 1
    
    assert ((fea_dist_mat - fea_dist_mat.T) == 0).all()

    dist_mat = k * wireless_dist_mat + (1 - k) * fea_dist_mat
    new_train_info, _ = nearest_neighbour_association(train_info.copy(), dist_mat, k=None)

    return get_pseudo_label(train_info.copy(), new_train_info, dataset.extra_info['train'])
