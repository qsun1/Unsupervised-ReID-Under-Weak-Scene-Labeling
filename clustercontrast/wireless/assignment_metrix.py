import numpy as np
from copy import deepcopy
from .check_data import check_ass_video_cam, check_person_record, check_time_group


__all__ = ['get_assignment_matrix']


def get_assignment_matrix(extra_data, raw_train_info, raw_test_info, npr, drop_rate, real_record, require_analys=False):
    print(extra_data['info'] + ' Drop {:.2f}'.format(drop_rate)) # gps_info
    extra_data = extra_data['data']
    timestamp = extra_data['timestamp']
    train_wifi = extra_data['train_wifi']
    test_wifi = extra_data['test_wifi']

    train_ass_mat = calculate_assignment_matrix(deepcopy(train_wifi), deepcopy(raw_train_info), # extra_info['train_info']
                                                deepcopy(timestamp), npr, drop_rate, real_record=real_record, 
                                                require_analys=require_analys)
    test_ass_mat = calculate_assignment_matrix(deepcopy(test_wifi), deepcopy(raw_test_info),
                                               deepcopy(timestamp), npr, drop_rate, real_record=real_record, 
                                               require_analys=require_analys)
    return train_ass_mat, test_ass_mat


def calculate_assignment_matrix(wifi_info, video_info, timestamp, npr, drop_rate, real_record, require_analys=False):
    all_assignment_tmp = []
    video_time = get_video_time_info(video_info, timestamp) #(868, 4)

    for c_id, wifi_device in wifi_info.items():
        camera_mac_assignment = get_camera_mac_assignment_matrix(wifi_device, c_id, video_time, real_record)
        for mac in camera_mac_assignment.keys():
            mac_info = camera_mac_assignment[mac]

            for mac_info_i in mac_info:
                mac_info_tmp = np.sum(mac_info_i, axis=0)
                all_assignment_tmp.append([int(mac), int(c_id), mac_info_tmp])
                check_ass_video_cam(mac_info_tmp, video_info, int(c_id))

    record_count = len(all_assignment_tmp)

    all_assignment_matrix = np.zeros((record_count, 2 + video_info.shape[0]), dtype=np.int64)
    for idx, record_i in enumerate(all_assignment_tmp):
        all_assignment_matrix[idx, 0] = record_i[0]  # mac id
        all_assignment_matrix[idx, 1] = record_i[1]  # camera id
        all_assignment_matrix[idx, 2:] = record_i[2]  # assignment_matrix

    mac_id = np.unique(all_assignment_matrix[:, 0])
    video_pids = np.unique(video_info[:, 0])

    print('Video ID lacks wifi: \n ' + str(np.setdiff1d(video_pids, np.intersect1d(mac_id, video_pids))))

    if drop_rate > 0 :
        print('Before dropping part of the associated record.')
    
    if require_analys:
        pass
        # analys_assignment_matrix(all_assignment_matrix, video_info, real_record, drop_rate>0)
    if drop_rate > 0:
        print('After dropping part of the associated record.')
        all_assignment_matrix = random_drop_record_ass(all_assignment_matrix, video_info, npr, drop_rate)
        if require_analys:
            pass
            # analys_assignment_matrix(all_assignment_matrix, video_info, real_record, drop_rate>0)
    return all_assignment_matrix


def get_video_time_info(video_info, timestamp):
    timestamp = np.asarray(timestamp, dtype=np.float32).reshape(-1)
    output = np.zeros((video_info.shape[0], 4), dtype=timestamp.dtype)
    output[:, 0] = video_info[:, 0]
    output[:, 1] = video_info[:, 1]
    output[:, 2] = timestamp[video_info[:, 2]]
    output[:, 3] = timestamp[video_info[:, 3]-1]
    return output


def get_camera_mac_assignment_matrix(wifi_device, camera_id, video_time, real_record):
    '''

    :param wifi_device:
      {'duration': int, the record time duration.'data': np.ndarray, shape [N, 5]}
    :return:
    '''

    record_duration = wifi_device['duration']   # the time period of wifi device record
    all_record = np.asarray(wifi_device['record'])
    # all_record[:, 0] is mac ID,
    # all_record[:, 1] is camera ID,
    # all_record[:, 2] and all_record[:, 3] For pseudo record, it is the index range of the video frames.
    #                                       For real record, it is meaningless and is set to -1.
    # all_record[:, 1] is timestamp.

    check_person_record(all_record)
    assert (all_record[:, 1] == camera_id).all()

    macs = np.unique(all_record[:, 0])

    output = {}

    pids = np.unique(video_time[:, 0]).astype(np.int64)

    for mac in macs:
        mac_info = all_record[all_record[:, 0] == mac]
        time_mark_tmp = mac_info[:, 4]
        time_mark_group = time_mark_split_rule(time_mark_tmp, record_duration*2)

        if int(mac) in pids:
            check_time_group(time_mark_group, video_time.copy(), int(mac), camera_id, real_record)

        mac_assignment_group = []

        for mark_group in time_mark_group:
            time_mark = np.asarray(mark_group)
            mac_assignment = np.zeros((time_mark.size, video_time.shape[0]), dtype=np.int64)
            for i_time, time_i in enumerate(time_mark):
                idx = find_related_video(time_i, video_time, camera_id, record_duration*2)
                if idx is None:
                    continue
                else:
                    mac_assignment[i_time, idx] = 1
            
            if int(mac) in pids and not real_record:
                assert mac_assignment.sum() > 0

            mac_assignment_group.append(mac_assignment)
        output[int(mac)] = mac_assignment_group
        assert len(mac_assignment_group) > 0
    return output


def time_mark_split_rule(time_mark, win):
    time_mark = time_mark.reshape(-1)
    time_mark = np.sort(time_mark)
    output = []
    group_tmp = []
    for idx in range(time_mark.size):
        if idx == 0:
            group_tmp.append(time_mark[idx])
            continue
        if time_mark[idx] - time_mark[idx-1] <= win:
            group_tmp.append(time_mark[idx])
        else:
            output.append(group_tmp)
            group_tmp = [time_mark[idx]]
    assert len(group_tmp) != 0
    output.append(group_tmp)
    return output


def find_related_video(time_mark, video_time, cid, win):
    output = []
    for idx, video_i in enumerate(video_time):
        if video_i[1] != cid:
            continue
        if time_mark >= video_i[2]-win and time_mark <= video_i[3]+win:
            output.append(idx)
    if len(output) == 0:
        return None
    else:
        return np.asarray(output).reshape(-1)


def random_drop_record_ass(ass_mat, video_info, npr, drop_rate):
    macs = np.unique(ass_mat[:,0])
    select_idx = np.arange(macs.size)
    npr.shuffle(select_idx)
    select_idx = select_idx[int(macs.size * drop_rate):]

    output = []
    for i in range(ass_mat.shape[0]):
        if ass_mat[i, 0] in select_idx:
            output.append(ass_mat[i])

    # select_idx = np.arange(ass_mat.shape[0])
    # npr.shuffle(select_idx)
    # select_idx = select_idx[int(ass_mat.shape[0] * drop_rate):]
    # ass_mat_output = ass_mat[select_idx].copy()
    return np.asarray(output)

