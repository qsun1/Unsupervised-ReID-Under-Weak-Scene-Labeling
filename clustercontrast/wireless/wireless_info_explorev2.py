from cProfile import label
import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from tabulate import tabulate


__all__ = ['mac_info_explorev2', 'analys_adj', 'histogram_count', 'get_cluster_number']


def mac_info_explorev2(ass_mat, video_info, fea, cluster_num, seed, match_rate_threshold): # MMDA # alpha for percent of match video
    N = video_info.shape[0]
    idx_tmp = np.arange(N)

    macs = np.unique(ass_mat[:, 0]) # 0-31
    acc_box = []
    as_acc = []
    mac_fea_group = {}
    
    for mac in macs:
        mac_info = ass_mat[ass_mat[:, 0] == mac] # 和mac关联的fragment, R_m
        record_num = mac_info.shape[0]
        fea_record_group_idx = {} # 第i个视频，在V_m^r中

        ass_idx_list = [] # T_m 下面所有的idx
        ass_fea_list = [] # T_m 下面所有的feature
        for r, record in enumerate(mac_info):
            ass_i = record[2:] 
            ass_idx = idx_tmp[ass_i > 0] # 对应视频idx，相当于找出V_m^r

            for idx_i in ass_idx:
                if idx_i not in fea_record_group_idx:
                    fea_record_group_idx[idx_i] = []
                    ass_idx_list.append(idx_i) 
                    ass_fea_list.append(fea[idx_i])
                fea_record_group_idx[idx_i].append(r) 

        if len(ass_fea_list) == 0:
            continue
        ass_fea = np.asarray(ass_fea_list)
        assert ass_fea.ndim == 2 and ass_fea.size > 1

        cluster_num_tmp = cluster_num[cluster_num[:, 0] == mac]
        assert cluster_num_tmp.shape[0] == 1
        # ----------------------------
        # fea.shape == (868, 2048), ass_fea:跟mac相关的视频特征, (274, 2048), fea_record_group_idx: {视频idx: fragment}, ass_idx_list: 跟mac相关的视频index
        # clustering with one mac
        ap = KMeans(n_clusters=int(min(cluster_num_tmp[0, 1], ass_fea.shape[0])), random_state=seed).fit(ass_fea)
        labels_fea = ap.labels_ 

        # -----------------------------
        #statistics within one mac
        fea_group = {} # 将一条无线轨迹下的feature group
        for i, label_i in enumerate(labels_fea): # Cm, k中各个视频的属性
            if label_i == -1: # ? 不是没有 -1 类了吗
                continue
            if label_i not in fea_group: # 初始化
                fea_group[label_i] = {'video_idx': [], 'record_id': [], 'feature': []}
            fea_group[label_i]['video_idx'].append(ass_idx_list[i]) # 对应视频idx
            fea_group[label_i]['feature'].append(fea[ass_idx_list[i]]) # 对应feature
            fea_group[label_i]['record_id'].extend(fea_record_group_idx[ass_idx_list[i]]) #在哪个fragment中，相当于r

        for label_i in fea_group: #遍历每个pseudolabel
            mean_feature = np.array(fea_group[label_i]['feature']).mean(axis=0)
            fea_group[label_i]['mean_feature'] = mean_feature / np.linalg.norm(mean_feature)
            # match_rate
            ass_record_num = np.unique(np.asarray(fea_group[label_i]['record_id'])).size
            match_rate = ass_record_num / record_num
            assert match_rate <= 1
            fea_group[label_i]['match_rate'] = match_rate

        if len(fea_group.keys()) > 0:
            mac_fea_group[mac] = fea_group
        else:
            raise ValueError
    # ----------------------------------------------
    # relabeling
    all_fea_group = {} 
    video_group_lbl = {} # 对于所有的视频idx，分别对应的group的label
    lbl = 0
    for mac in macs:
        for group_lbl in mac_fea_group[mac]:
            fea_group = mac_fea_group[mac][group_lbl]

            if fea_group['match_rate'] < match_rate_threshold:
                continue

            all_fea_group[lbl] = fea_group # relabel
            group_video_indexes = fea_group['video_idx']
            for index in group_video_indexes:
                if index not in video_group_lbl:
                    video_group_lbl[index] = []
                video_group_lbl[index].append(lbl)
            lbl += 1
    print(f"get {lbl + 1} wireless clusters!")

    wireless_label = -np.ones(N, dtype=int)
    for video_idx in video_group_lbl:
        video_feature = fea[video_idx]
        video_cluster_lbl = video_group_lbl[video_idx]
        video_cluster_cos_dist = [np.dot(video_feature, all_fea_group[lbl]['mean_feature']) for lbl in video_cluster_lbl]
        final_label = video_cluster_lbl[np.argmin(np.array(video_cluster_cos_dist))]
        wireless_label[video_idx] = final_label
    
    
    #----------
    # relabeling
    all_label = np.unique(wireless_label[wireless_label != -1])
    for final_label, label in enumerate(all_label):
        wireless_label[wireless_label == label] = final_label

    wireless_label_unlabeled = wireless_label.copy()
    
    # ----------
 
    video_info[:, 0] = wireless_label
    wireless_info = video_info[np.array(list(video_group_lbl.keys()))]
    # wireless feature
    wireless_feature = []
    for label_i in range(len(all_fea_group)):
        wireless_feature.append(all_fea_group[label_i]['mean_feature'])
    wireless_feature = np.array(wireless_feature)

    return wireless_info, wireless_label_unlabeled


def get_cluster_number(ass_mat, video_info, alpha):
    logger = logging.getLogger(__name__)

    ass_mat = ass_mat.copy()
    video_info = video_info.copy()
    record_num = ass_mat.shape[0]

    assert ass_mat.shape[1] == video_info.shape[0] + 2

    v_pids = video_info[:, 0]
    macs = np.unique(ass_mat[:, 0])

    r_per_mac = record_num / macs.size

    mac_ass_video_num = []
    mac_ass_person_num = []

    for mac in macs:
        mac_info = ass_mat[ass_mat[:, 0] == mac]

        mac_ass_video_tmp = mac_info.sum(axis=0)[2:]
        mac_ass_video_num.append(mac_ass_video_tmp[mac_ass_video_tmp > 0].size)
        mac_ass_person_num.append(np.unique(v_pids[mac_ass_video_tmp > 0]).size)

    mac_ass_video_num = np.asarray(mac_ass_video_num)
    mac_ass_person_num = np.asarray(mac_ass_person_num)
    e_mac_ass_person_num = mac_ass_video_num / r_per_mac

    log_info = []

    log_info.append(['Video mean/std', '{:.1f}/{:.1f}'.format(mac_ass_video_num.mean(), mac_ass_video_num.std())])
    log_info.append(['Person mean/std', '{:.1f}/{:.1f}'.format(mac_ass_person_num.mean(), mac_ass_person_num.std())])
    log_info.append(['E Person mean/std', '{:.1f}/{:.1f}'.format(e_mac_ass_person_num.mean(), e_mac_ass_person_num.std())])
    log_info.append(['|GT mean -GT| mean', '{:.1f}'.format(np.abs(mac_ass_person_num.mean() - mac_ass_person_num).mean())])
    log_info.append(['|K*1-GT| mean', '{:.1f}'.format(np.abs(e_mac_ass_person_num * 1 - mac_ass_person_num).mean())])
    log_info.append(['|K*2-GT| mean', '{:.1f}'.format(np.abs(e_mac_ass_person_num * 2 - mac_ass_person_num).mean())])
    log_info.append(['|K*3-GT| mean', '{:.1f}'.format(np.abs(e_mac_ass_person_num * 3 - mac_ass_person_num).mean())])
    log_info.append(['|K*4-GT| mean', '{:.1f}'.format(np.abs(e_mac_ass_person_num * 4 - mac_ass_person_num).mean())])
    log_info.append(['|K*5-GT| mean', '{:.1f}'.format(np.abs(e_mac_ass_person_num * 5 - mac_ass_person_num).mean())])
    log_info.append(['Alpha', '{}'.format(alpha)])

    e_mac_ass_person_num = np.ceil(e_mac_ass_person_num* alpha)
    log_info.append(['|K*alpha-GT| mean', '{:.1f}'.format(np.abs(e_mac_ass_person_num - mac_ass_person_num).mean())])
    logger.info('Analy the estimation of K\n{}'.format(tabulate(log_info)))

    return np.stack((macs, e_mac_ass_person_num), axis=1)