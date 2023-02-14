from __future__ import print_function, absolute_import
import pickle
import numpy as np
import random
from pathlib import Path
from ..utils.data import BaseImageDataset
from ..wireless.data import np_filter

class VideoDataset(BaseImageDataset):
    def _get_data(self, info, img_dir_list):
        data_info = []
        video_info = []
        for i in range(info.shape[0]):
            pid = info[i, 0]
            cid = info[i, 1]
            begin = info[i, 2]
            end = info[i, 3]
            for frame_idx in range(begin, end):
                frame_dir = img_dir_list[frame_idx]
                data_info.append((frame_dir, pid, cid))
            video_info.append((img_dir_list[begin:end], pid, cid))
        return data_info, video_info

    def _prepare_test(self, img_dir_list, probe_info, gallery_info):

        probe_data, probe_video = self._get_data(probe_info, img_dir_list)
        gallery_data, gallery_video = self._get_data(gallery_info, img_dir_list)
        return probe_data, probe_video, gallery_data, gallery_video

    def _check(self, train_info, test_info, probe_info, gallery_info, test_only):
        assert np.unique(test_info[:, 2]).size == test_info.shape[0]

        assert np.sum(train_info[:, 3] - train_info[:, 2] - train_info[:, 4]) == 0
        assert np.sum(test_info[:, 3] - test_info[:, 2] - test_info[:, 4]) == 0
        assert np.sum(probe_info[:, 3] - probe_info[:, 2] - probe_info[:, 4]) == 0
        assert np.sum(gallery_info[:, 3] - gallery_info[:, 2] - gallery_info[:, 4]) == 0

        test_id = np.unique(test_info[:, 0])
        probe_id = np.unique(probe_info[:, 0])
        gallery_id = np.unique(gallery_info[:, 0])
        assert -1 not in set(test_id)   # junk id set to be -1, it should have been removed.

        assert np.setdiff1d(probe_id, gallery_id).size == 0
        assert set(test_id) == set(probe_id).union(set(gallery_id))

        for probe_i in range(probe_info.shape[0]):
            data_info = probe_info[probe_i]
            p_id = data_info[0]
            p_cam_id = data_info[1]
            g_info = np_filter(gallery_info, [p_id])
            g_cam_id = np.unique(g_info[:, 1])
            if not np.setdiff1d(g_cam_id, np.asarray([p_cam_id])).size > 0:
                print('All gallery trackets have the same camera id with probe tracklet for ID: ' + str(p_id))

        assert np.unique(test_info[:, 2]).size == np.unique(np.concatenate((probe_info, gallery_info))[:, 2]).size
        if not test_only:
            assert np.intersect1d(train_info[:, 2], test_info[:, 2]).size == 0
        assert np.unique(train_info[:, 2]).size == train_info.shape[0]
        assert np.unique(test_info[:, 2]).size == test_info.shape[0]
        assert np.unique(probe_info[:, 2]).size == probe_info.shape[0]
        assert np.unique(gallery_info[:, 2]).size == gallery_info.shape[0]

        return test_info, probe_info

    def pick_train_data(self, num_picked):
        info = self.extra_info['train_info'].copy()
        img_dir_list = self.images_dirs

        data_info = []
        video_info = []
        for i in range(info.shape[0]):
            pid = info[i, 0]
            cid = info[i, 1]
            begin = info[i, 2]
            end = info[i, 3]
            img_list = img_dir_list[begin:end]
            num_picked_now = len(img_list) if len(img_list) < num_picked else num_picked
            picked_imgs = random.sample(img_list, num_picked_now)

            video_info.append((picked_imgs, pid, cid))

            for frame_dir in picked_imgs:
                data_info.append((frame_dir, pid, cid))
        self.train = data_info
        self.extra_info['train'] = video_info


class WPReID(VideoDataset):
    dataset_dir = ''

    def __init__(self, root, verbose=True, **kwargs):
        super(WPReID, self).__init__()
        self.dataset_dir = Path('examples/data/wpreid') # replace by your WPReID path
        self.split_file_dir = self.dataset_dir / 'WPReID_dict.json'
        self.all_image_num = 106578

        data_dict = self._get_dict(self.split_file_dir)
        self.images_dirs = data_dict['dir']
        self.query, probe_video, self.gallery, gallery_video = self._prepare_test(self.images_dirs, data_dict['probe'], data_dict['gallery'])

        self.train, train_video = self._get_data(data_dict['gallery'], self.images_dirs)

        self.extra_info ={'query': probe_video, 
        'gallery':gallery_video, 
        'train':train_video, 
        'train_info':data_dict['gallery'], 
        'test_info':data_dict['gallery']}
        assert len(self.images_dirs) == self.all_image_num
        self.gps_info = data_dict['gps']

    def _get_dict(self, file_path):
        with open(file_path, 'rb') as f:
            info = pickle.load(f)
            print('Load data <--- ' + str(file_path), flush=True)
        
        img_dir_list = info['dir']

        img_dir_list = [str(self.dataset_dir) + i for i in img_dir_list]
        '''
                img_dir_list is a list of all the images in the dataset.
                Each image is a cropped person image. 
                '''
        gps_info = info['extra_data']['gps_detect'][4]
        info = info['split'][0]
        probe_info = info['probe']
        gallery_info = info['gallery']

        _, probe_info = self._check(gallery_info, gallery_info, probe_info, gallery_info, test_only=True)

        data_dict = {}
        data_dict['dir'] = img_dir_list
        data_dict['probe'] = probe_info
        data_dict['gallery'] = gallery_info
        data_dict['info'] = 'WPReID dataset'
        data_dict['gps'] = gps_info
        return data_dict