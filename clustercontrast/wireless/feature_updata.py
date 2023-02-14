from sklearn.model_selection import learning_curve
import torch
import numpy as np
from collections import Counter

from ..utils import to_torch
from .wireless_info_explore import mac_info_explore
from .wireless_info_explorev2 import mac_info_explorev2
from .data import cosine_distance, feadict_to_videofea, video_fea_to_feadict
from .tracklet_association import nearest_neighbour_association


def direct_nns(feature, dataset):
    feature = feadict_to_videofea(feature, dataset.extra_info['train']) # video features
    train_info = dataset.extra_info['train_info'].copy()
    feature = to_torch(feature).cuda()
    fea_dist_mat = cosine_distance(feature, feature).cpu().numpy() 
    new_train_info, _ = nearest_neighbour_association(train_info.copy(), fea_dist_mat, k=None) # (281, 5)
    return get_pseudo_label(train_info.copy(), new_train_info, dataset.extra_info['train'])


def get_pseudo_label(raw_info, new_info, video_info):
    assert raw_info.shape[0] == len(video_info)
    raw_idx_list = raw_info[:,2].tolist() # (868, 5); start frame of video sequences
    new_idx_list = new_info[:,2].tolist() # (281, 5)
    outpout = []
    for i, idx in enumerate(raw_idx_list):
        if idx in new_idx_list:
            pid = new_info[new_idx_list.index(idx), 0]
        else:
            pid = -1    # w / o reciprocal Nearest Neighbor
        for f_dir in video_info[i][0]: # list containing all img pths from one tracklet
            outpout.append((f_dir, pid))
    label = np.asarray([f[1] for f in sorted(outpout)])
    return label # pseudo labels for all imgs
        
def fea_update_by_wireless(train_ass_mat, mac_cluster_num, feature, dataset, trainer, seed, alpha):
    feature = feadict_to_videofea(feature, dataset.extra_info['train'])
    train_info = dataset.extra_info['train_info'].copy()
    video_ass, acc_cluster,  acc_ass, _ = mac_info_explore(train_ass_mat.copy(), train_info.copy(), feature.numpy(), mac_cluster_num, seed, alpha)
    feature = to_torch(feature).cuda()
    fea_dist_mat = cosine_distance(feature, feature).cpu().numpy()
    new_train_info, selected_idx = nearest_neighbour_association(train_info.copy(), fea_dist_mat, k=None)

    trainer.init(
            torch.index_select(feature, dim=0, index=torch.from_numpy(selected_idx).long().cuda()),
            video_ass[selected_idx][:, selected_idx].copy(), new_train_info)
    trainer.train()
    feature = trainer.inference(feature, video_ass)

    fea_dist_mat = cosine_distance(feature, feature).cpu().numpy()
    new_train_info, _ = nearest_neighbour_association(train_info.copy(), fea_dist_mat, k=None)

    return get_pseudo_label(train_info.copy(), new_train_info, dataset.extra_info['train'])

# 给每个图片分配无线标签M维的向量
def get_wireless_label(vw_match_score, video_info):
    wireless_labels = []
    for i, (img_pths, _, _) in enumerate(video_info):
        if vw_match_score[:, i].sum() != 0:
            for img_pth in img_pths:
                wireless_labels.append(np.argmax(vw_match_score[:, i]))
        else:
            wireless_labels.append(-1)
    return wireless_labels

def fea_update_by_wireless_v3(train_ass_mat, mac_cluster_num, feature, dataset, seed, match_rate_threshold):
    feature = feadict_to_videofea(feature, dataset.extra_info['train'])
    train_info = dataset.extra_info['train_info'].copy()
    feature = to_torch(feature).cuda()
    fea_dist_mat = cosine_distance(feature, feature).cpu().numpy()
    new_train_info, _ = nearest_neighbour_association(train_info.copy(), fea_dist_mat, k=None)

    wireless_train_info, wireless_video_label = mac_info_explorev2(train_ass_mat.copy(), train_info.copy(), feature.cpu().numpy(), mac_cluster_num, seed, match_rate_threshold)

    visual_video_label = get_video_label(train_info.copy(), new_train_info)
    #wireless_video_label = wireless_train_info[:, 0]

    new_wireless_video_label = wireless_video_label
    # new_wireless_video_label = update_wireless_with_visual(wireless_video_label, visual_video_label, 1)
    visual_video_label = update_wireless_with_visual(visual_video_label, wireless_video_label, 0)
    # new_visual_video_label = update_wireless_with_visual(visual_video_label, wireless_video_label) 
    # new_visual_video_label = update_wireless_with_visual(visual_video_label, new_wireless_video_label)  


    new_train_info = get_info(visual_video_label, train_info)
    new_wireless_train_info = get_info(new_wireless_video_label, train_info)

    return get_pseudo_label(train_info.copy(), new_train_info, dataset.extra_info['train']),\
            get_pseudo_label(train_info.copy(), new_wireless_train_info, dataset.extra_info['train'])

# video label -> video info
def get_info(new_label, raw_info):
    raw_info[:, 0] = new_label
    return raw_info[new_label != -1, :]


# unlabeled -> labeled
def update_wireless_with_visual(wireless_label, visual_label, flag):
    # flag = 1 visual assist wireless; flag = 0 wireless assist visual
    assert wireless_label.shape == visual_label.shape
    num_wireless_label = max(wireless_label)
    new_wireless_label = wireless_label.copy()

    unlabeled_index = np.where(wireless_label == -1)
    visual_cluster_list = np.unique(visual_label[unlabeled_index])
    visual_cluster_list = visual_cluster_list[visual_cluster_list != -1] # 去掉-1

    add_cluster = 0
    for visual_cluster in visual_cluster_list:
        visual_index = np.where(visual_label == visual_cluster)
        visual_correspond_wireless_label = wireless_label[visual_index]
        unlabeled_vsual_correspond_wireless_index = np.where((visual_label == visual_cluster) & (wireless_label == -1))
        final_wireless_label = Counter(visual_correspond_wireless_label).most_common(1)[0][0]
        if final_wireless_label == -1:
            add_cluster += 1
            new_wireless_label[unlabeled_vsual_correspond_wireless_index] = num_wireless_label + add_cluster # unlabeled 自成一类
        else:
            new_wireless_label[unlabeled_vsual_correspond_wireless_index] = final_wireless_label # unlabeled合并
    if flag == 1:
        print(f'visual Add {add_cluster} cluster for wireless of {len(visual_cluster_list)}!')
    else:
        print(f'Wireless Add {add_cluster} cluster for Visual of {len(visual_cluster_list)}!')
    return new_wireless_label


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
    