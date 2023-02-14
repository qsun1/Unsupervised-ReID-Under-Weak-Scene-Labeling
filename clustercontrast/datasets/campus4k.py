from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
import urllib
import zipfile
import pickle
import json
import numpy as np
from pathlib import Path

from ..utils.data import BaseImageDataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json
from .wpreid import VideoDataset


class Campus4k(VideoDataset):
    dataset_dir = ''

    def __init__(self, root, verbose=True, **kwargs):
        super(Campus4k, self).__init__()
        self.dataset_dir = Path('/data/sunq/wire_visual/')
        self.split_file_dir = self.dataset_dir / 'Campus4k/Campus4k_dict.json'
        self.all_image_num = 521309
        
        data_dict = self._get_dict(self.split_file_dir)
        self.images_dirs = data_dict['dir']
        self.query, probe_video, self.gallery, gallery_video = self._prepare_test(data_dict['dir'], data_dict['probe'], data_dict['gallery'])
        self.train, train_video = self._get_data(data_dict['train'], data_dict['dir'])
        self.extra_info ={'query': probe_video, 
        'gallery':gallery_video, 
        'train':train_video, 
        'train_info':data_dict['train'], 
        'test_info':np.concatenate((data_dict['probe'], data_dict['gallery']), axis=0)}

        assert len(data_dict['dir']) == self.all_image_num
        self.gps_info = data_dict['gps']
    
    def _get_dict(self, file_path):
        with open(file_path, 'rb') as f:
            info = pickle.load(f)
            print('Load data <--- ' + str(file_path), flush=True)
        
        img_dir_list = info['dir']
        img_dir_list = [str(self.dataset_dir) + i for i in img_dir_list]
        gps_info = info['extra_data'][33]
        info = info['split'][0]

        data_dict = {}
        data_dict['dir'] = img_dir_list
        data_dict['train'] = info['train']
        data_dict['probe'] = info['probe']
        data_dict['gallery'] = info['gallery']
        data_dict['info'] = 'Campus4k dataset'
        data_dict['gps'] = gps_info
        return data_dict


class Campus4kOld(VideoDataset):
    dataset_dir = ''

    def __init__(self, root, verbose=True, **kwargs):
        super(Campus4kOld, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        
        self.raw_data_folder = Path('/data/sunq/wire_visual')

        self.train_dir = self.raw_data_folder / 'train'
        self.query_dir = self.raw_data_folder / 'query'
        self.gallery_dir = self.raw_data_folder / 'gallery'

        self.all_image_num = 521309

        data_dict = self._get_dict()
        data_split = data_dict['split'][0]
        self.query, probe_video, self.gallery, gallery_video = self._prepare_test(data_dict['dir'], data_split['probe'], data_split['gallery'])

        self.train, train_video = self._get_data(data_split['train'], data_dict['dir'])

        self.extra_info={'query': probe_video, 'gallery':gallery_video, 'train':train_video}
        assert len(data_dict['dir']) == self.all_image_num

    def _get_dict(self):

        train_dirs, train_pids, train_cids = self._process_dir(self.train_dir)

        query_dirs, query_pids, query_cids = self._process_dir(self.query_dir)

        gallery_dirs, gallery_pids, gallery_cids = self._process_dir(self.gallery_dir)

        pid_map = self._relabel(train_pids, query_pids, gallery_pids)
        cid_map = self._relabel(train_cids, query_cids, gallery_cids)

        images_dirs = []
        timestamps = []
        train_dirs, train_info, train_time = self._process_data(train_dirs, train_pids, pid_map, cid_map, 0)
        images_dirs += train_dirs
        timestamps += train_time
        query_dirs, query_info, query_time = self._process_data(query_dirs, query_pids, pid_map, cid_map, len(images_dirs))
        images_dirs += query_dirs
        timestamps += query_time
        gallery_dirs, gallery_info, gallery_time = self._process_data(gallery_dirs, gallery_pids, pid_map, cid_map, len(images_dirs))
        images_dirs += gallery_dirs
        timestamps += gallery_time


        train_track = np.asarray(train_info, dtype=np.int64)
        query_track = np.asarray(query_info, dtype=np.int64)
        gallery_track = np.asarray(gallery_info, dtype=np.int64)
        test_track = np.asarray(query_info+gallery_info, dtype=np.int64)
        timestamps = np.asarray(timestamps, dtype=np.float) / 30.0  # 30fps


        assert train_track[:, 4].sum() + test_track[:, 4].sum() == len(images_dirs)
        assert train_track[-1, 3] == test_track[0, 2] and test_track[-1, 3] == len(images_dirs)
        assert train_track[0, 2] == 0

        images_dirs = [str(i) for i in images_dirs]

        data_dict = {}
        data_dict['dir'] = images_dirs
        data_split = {}
        data_split['train'] = train_track
        data_split['probe'] = query_track
        data_split['gallery'] = gallery_track
        data_split['info'] = 'Campus4k dataset. Split ID {:2d}'.format(0)
        data_dict['split'] = [data_split]
        data_dict['info'] = 'Campus4k Dataset.'

        return data_dict

    def _relabel(self, train, query, gallery):
        all = list(set(train+query+gallery))
        all.sort()
        id_map = {}
        for idx, old_id in enumerate(all):
            id_map[old_id] = idx
        return id_map

    def _process_data(self, dir_dict, pids, pid_map, cid_map, begin):
        images_dirs = []
        timestamp_list = []
        track_info = []
        idx = begin
        for old_pid in pids:
            new_pid = pid_map[old_pid]
            tids = list(dir_dict[old_pid].keys())
            tids.sort()
            for tid in tids:
                t_data = dir_dict[old_pid][tid]
                cid = t_data[0]
                images_dirs += t_data[1]
                timestamp_list += t_data[2]
                track_info.append([new_pid, cid_map[cid], idx, idx+len(t_data[1]), len(t_data[1])])
                idx += len(t_data[1])
        
        assert len(images_dirs) == len(timestamp_list)
        return images_dirs, track_info, timestamp_list
        
    
    def _process_dir(self, dir_path):
        pid_dirs = list(dir_path.glob('*')) 
        pids = [int(pid_dir.name) for pid_dir in pid_dirs]
        pids.sort()

        cids = set()

        data_dict = {}
        
        for pid_dir in pid_dirs:
            pid = int(pid_dir.name)
            data_dict[pid] = {}
            tracklet_dirs = list(pid_dir.glob('*'))
            tracklets = [int(tracklet_dir.name) for tracklet_dir in tracklet_dirs]
            tracklets.sort()

            for tracklet_dir in tracklet_dirs:
                tid = tracklet_dir.name
                assert tid not in data_dict[pid]

                frame_dirs = list(tracklet_dir.glob('*.jpg'))
                frame_dirs = [str(f.name) for f in frame_dirs]
                frame_dirs.sort()

                assert len(frame_dirs) >= 1
                cid = int(frame_dirs[0][6])
                cids.add(cid)
                timestamps = [int(f[9:14]) for f in frame_dirs]
                new_frame_dir = []
                for frame in frame_dirs:
                    assert int(frame[:4]) == pid
                    assert int(frame[6]) == cid
                    new_frame_dir.append(tracklet_dir / frame)
                
                data_dict[pid][tid] = [cid, new_frame_dir, timestamps]

        return data_dict, pids, list(cids)