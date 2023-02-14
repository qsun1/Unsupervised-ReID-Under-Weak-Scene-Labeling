import numpy as np
from sklearn.cluster import KMeans
from .data import np_filter


__all__ = ['nearest_neighbour_association']


def kmeans_1d_k(data, k):
    init = np.linspace(min(data), max(data), k)
    x = np.reshape(data, (-1,1))
    init = np.reshape(init, (-1,1))
    kmeans = KMeans(n_clusters=k, init=init, n_init=1)
    kmeans.fit(x)
    res = kmeans.predict(x)
    return res


def cross_camera_nearest_neighbour(dist_mat, idx_a, idx_b, k=None):
    m = idx_a.size
    n = idx_b.size
    assert idx_a.size == m and idx_b.size == n
    dist_mat = dist_mat[idx_a][:, idx_b] # (m, n): retrieve columns(camid=0) & row(camid=1); a submatrix of the origin matrix
    inds_1 = np.argsort(dist_mat, 0)
    inds_2 = np.argsort(dist_mat, 1)
    rank_1 = np.full((m, n), max(m, n))
    rank_2 = np.full((m, n), max(m, n))

    for i in range(m):
        for j in range(n):
            rank_1[i, inds_2[i, j]] = j
            rank_2[inds_1[i, j], j] = i

    rank = rank_1 + rank_2

    matched_pairs = []
    matched_pairs_dst = []

    for i in range(m):
        for j in range(n):
            if rank[i, j] <= 0: # rank1==0 && rank2==0
                matched_pairs.append([idx_a[i], idx_b[j]])
                matched_pairs_dst.append(dist_mat[i, j])

    if k is not None:
        res = kmeans_1d_k(matched_pairs_dst, k)
        new_matched_pairs = []
        new_matched_pairs_dst = []
        for i, t_pairs in enumerate(matched_pairs):
            if res[i] == 0:
                new_matched_pairs.append(t_pairs)
                new_matched_pairs_dst.append(matched_pairs_dst[i])
        return new_matched_pairs, new_matched_pairs_dst
    else:
        return matched_pairs, matched_pairs_dst


def get_index(rawset, subset):
    index = []
    for i_probe in range(subset.shape[0]):
        begin = subset[i_probe, 2] # using start frame to search corresponding index
        temp_index = np.where(rawset[:, 2] == begin)[0]
        assert temp_index.size == 1
        index.append(temp_index[0])
    index = np.asarray(index, dtype=np.int64)
    return index


def nearest_neighbour_association(video_info, dist_mat, k=None): # (868, 5) (868, 868)
    cam_ids = np.unique(video_info[:, 1])
    cam_ids.sort()
    idx_list = np.arange(video_info.shape[0], dtype=video_info.dtype)

    cam_dict = {} # cam_dict[0]['idx']: camid == 0 containing tracklet idx; cam_dict[0]['info']: corresponding info
    for cid in cam_ids:
        cam_dict[cid] = {'info': video_info[video_info[:, 1] == cid].copy(),
                         'idx': idx_list[video_info[:, 1] == cid]
                         }

    match_pairs_all = []
    #survey all cross-camera pairs
    for cid_a in cam_ids: 
        for cid_b in cam_ids:
            if cid_a == cid_b:
                continue
            else:
                match_pairs_camera, match_pairs_dist_camera = cross_camera_nearest_neighbour(
                    dist_mat,
                    cam_dict[cid_a]['idx'],
                    cam_dict[cid_b]['idx'], k=k) # camid == 0 tracklet ids & camid == 1 tracklet ids
                match_pairs_all.extend(match_pairs_camera) # gets 400 match pairs in total...

    group_list = [] # group pairs (106)
    for (idx_a, idx_b) in match_pairs_all:
        find_flag = False
        for g in group_list:
            if idx_a in g or idx_b in g:
                g.extend([idx_a, idx_b])
                find_flag = True
                break
        if not find_flag:
            group_list.append([idx_a, idx_b])

    new_video_info = video_info.copy()
    new_video_info[:, 0] = np.arange(new_video_info.shape[0], dtype = new_video_info.dtype)

    for g in group_list:
        g_tmp = np.unique(np.asarray(g))
        g_tmp.sort()
        for idx in g_tmp[1:]:
            new_video_info[idx, 0] = new_video_info[g_tmp[0], 0] # subtitute video pid with the same pid (pseudo label)

    pids = np.unique(new_video_info[:, 0])
    pids.sort()
    output = [] # video info (assigned with pseudolabels)
    new_id = 0
    for pid in pids:
        p_data = new_video_info[new_video_info[:, 0] == pid]
        if p_data.shape[0] > 1: # filtering unclustered tracklets
            p_data = p_data.copy()
            p_data[:, 0] = new_id # assign tracklets with new id (natural sequences)
            output.append(p_data)
            new_id += 1

    output = np.concatenate(output, axis=0) 
    idx = get_index(video_info, output) # indexes for clustered tracklets 
    return output, idx 