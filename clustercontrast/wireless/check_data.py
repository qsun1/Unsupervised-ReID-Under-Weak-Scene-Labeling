import numpy as np
from .data import np_filter


def check_camera_data(record_info, video_info, c_id):
    pids = np.unique(video_info[:, 0])
    macs = np.unique(record_info[:, 0])
    for mac in macs:
        assert mac in pids
        p_info = np_filter(video_info, [mac])
        assert (p_info[:, 1] == c_id).any(), 'mac {} cid {} info {}'.format(int(mac), c_id, str(p_info))


def check_person_record(record_data):
    pids = np.unique(record_data[:, 0])
    for pid in pids:
        p_data = record_data[record_data[:, 0] == pid]
        assert np.unique(p_data[:, 4]).size == p_data.shape[0], 'time: {} video data: {}'.format(np.unique(p_data[:, 4]).size,  p_data.shape)


def check_pseudo_record(wifi_device_all, tracklet_info):
    detected_box = []
    confuse_box = []
    pids = np.unique(tracklet_info[:, 0])

    for c_id, wifi_device in wifi_device_all.items():
        detected_info = wifi_device['detected']
        detected_box.append(detected_info)
        record_info = wifi_device['record']
        check_camera_data(detected_info, tracklet_info, c_id)

        if 'confuse' in wifi_device:
            confuse_info = wifi_device['confuse']
            confuse_box.append(confuse_info)
        
        assert (record_info[:, 1] == c_id).all()
    
    detected_box = np.concatenate(detected_box, axis=0)
    detected_ids = np.unique(detected_box[:, 0])
    assert np.intersect1d(detected_ids, pids).size == detected_ids.size
    
    if len(confuse_box) > 0:
        confuse_box = np.concatenate(confuse_box, axis=0)
        confuse_ids = np.unique(confuse_box[:, 0])
        assert np.intersect1d(confuse_ids, pids).size == 0
        assert np.intersect1d(detected_ids, confuse_ids).size == 0


def check_ass_video_cam(ass_info, video_info, cid):
    p_cid = video_info[ass_info > 0, 1]
    assert (p_cid == cid).all()


def check_time_group(time_mark_group, video_time, pid, cid, real_record):

    p_data = np_filter(video_time, [int(pid)], [cid])

    if not real_record:
        assert len(time_mark_group) <= p_data.shape[0]
    else:
        if len(time_mark_group) > p_data.shape[0]:
            print('Camera {} Person {} with {} video, but obtain {} mac record.'.format(cid, pid, p_data.shape[0], len(time_mark_group)))