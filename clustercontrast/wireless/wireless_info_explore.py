import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from tabulate import tabulate


__all__ = ['mac_info_explore', 'analys_adj', 'histogram_count', 'get_cluster_number']


def mac_info_explore(ass_mat, video_info, fea, cluster_num, seed, alpha): # MMDA # alpha for percent of match video
    N = video_info.shape[0]
    idx_tmp = np.arange(N)

    macs = np.unique(ass_mat[:, 0]) # 0-31
    output = []
    acc_box = []
    as_acc = []
    vw_match_score = []
    
    for mac in macs:
        video_wireless_match_score = np.zeros(N, dtype=np.float32) # for match in this mac
        video_ass_matrix = np.zeros((N, N), dtype=np.float32)

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
        ap = KMeans(n_clusters=int(min(cluster_num_tmp[0, 1], ass_fea.shape[0])), random_state=seed).fit(ass_fea)
        labels_fea = ap.labels_
        acc = adjusted_mutual_info_score(video_info[np.asarray(ass_idx_list), 0][labels_fea != -1], labels_fea[labels_fea != -1]) # what is acc?
        acc_box.append(acc)

        fea_relabel = labels_fea.max() + 1
        for i in range(labels_fea.size):
            if labels_fea[i] == -1: # outliner, 单独成一类
                labels_fea[i] = fea_relabel
                fea_relabel += 1

        fea_group = {} # 将一条无线轨迹下的feature group
        for i, label_i in enumerate(labels_fea): # Cm, k中各个视频的属性
            if label_i == -1: # ? 不是没有 -1 类了吗
                continue
            if label_i not in fea_group: # 初始化
                fea_group[label_i] = {'fea': [], 'record_id': []}
            fea_group[label_i]['fea'].append(ass_idx_list[i]) # 对应视频idx
            fea_group[label_i]['record_id'].extend(fea_record_group_idx[ass_idx_list[i]]) #在哪个fragment中，相当于r

        if len(fea_group.keys()) > 0:

            for label_i, fea_g in fea_group.items():
                ass_record_num = np.unique(np.asarray(fea_g['record_id'])).size
                match_rate = ass_record_num / record_num
                assert match_rate <= 1      

                ass_fea_idx = fea_g['fea']
                for idx in ass_fea_idx:
                    if video_info[idx, 0] == mac:
                        as_acc.append(match_rate) # 和真实做测评
                    video_ass_matrix[idx, ass_fea_idx] = match_rate

                    # new add
                    video_wireless_match_score[idx] = match_rate

        else:
            raise ValueError
        
        output.append(video_ass_matrix)
        assert ((video_ass_matrix - video_ass_matrix.T) == 0).all() # symmetric matrix
    
        vw_match_score.append(video_wireless_match_score)
    vw_match_score = np.stack(vw_match_score, axis=0)

    output = np.stack(output, axis=2)
    row, col = np.diag_indices_from(output[..., 0])
    output[row, col, :] = 1
    acc_box = np.asarray(acc_box).mean()

    return output, acc_box, np.asarray(as_acc).mean(), vw_match_score


def analys_adj(adj, label):
    N = label.size
    match_label = (np.tile(label[:, np.newaxis], N) == np.tile(label[:, np.newaxis], N).T).astype(np.float32)
    match_label_pure = match_label - np.eye(N)
    unmatch_label = 1 - match_label
    positive_v = (adj * match_label).sum() / match_label.sum()
    positive_p = (adj * match_label_pure).sum() / match_label_pure.sum()
    negative_v = (adj * unmatch_label).sum() / unmatch_label.sum()
    negative_p = (adj * unmatch_label).sum() / (adj * unmatch_label>0).sum()

    positive_d = N / match_label.sum()
    return 'PP/P {:.4f}/{:.4f} NP/N {:.4f}/{:.4f} I {:.4f}'.format(positive_p, positive_v, negative_p, negative_v, positive_d)


def histogram_count(adj):
    adj = adj.copy()
    row, col = np.diag_indices_from(adj[..., 0])
    adj[row, col, :] = 0
    adj_max = adj.max()
    adj_min = adj.min()
    adi = adj.reshape(-1)

    assert adj_min >= 0
    assert adj_max <= 1
    N = adj.shape[0]
    M = adj.shape[2]
    bin_num = 32
    bins = bin_num-1 if adj_max < 1 else bin_num

    histogram_all = np.zeros((N, N, bin_num), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i != j:
                elem_his, bin_range = np.histogram(adj[i, j], bins=bins, range=(adj_min, adj_max))
                histogram_all[i, j][:elem_his.size] = elem_his
            else:
                histogram_all[i, j, -1] = M
  
    histogram_all /= M
    return histogram_all


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