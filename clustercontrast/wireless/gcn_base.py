import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
from collections import OrderedDict
from pathlib import Path
import scipy.sparse as sp
from .wireless_info_explore import histogram_count
from .model_init import weights_init_kaiming
from .data import normalize_2d, sparse_mx_to_torch_sparse_tensor
from .optimizer import default_optimizer
from .utils import inference_context


__all__ = ['GCNBase', 'GCNTrainer']


class GCNBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.CrossEntropyLoss()

    def losses(self, outputs, other_info):
        global_feat_bn = outputs

        loss_identity_0 = self.identity(global_feat_bn, other_info)

        loss_dict = OrderedDict({
            'G_I': loss_identity_0
        })

        return loss_dict


def data_preprocess(data, adj):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    
    assert adj.ndim == 3
    adj_mean = adj.mean(axis=2)
    adj_mean = normalize_2d(adj_mean)
    adj_sp = sp.csr_matrix(adj_mean, dtype=np.float32)
    adj_sp = sparse_mx_to_torch_sparse_tensor(adj_sp)
    adj_h = histogram_count(adj.copy())

    if not data.is_cuda:
        data = data.cuda()
        
    adj_h = torch.from_numpy(adj_h)
    adj_mean = torch.from_numpy(adj_mean)
    return data, adj_h, adj_sp, adj_mean


class GCNDataloader(object):
    def __init__(self, fea, adj, video_info=None, adj_type='histogram'):
        fea, adj, adj_sp, adj_mean= data_preprocess(fea, adj)
        
        if video_info is not None:
            self.get_label(video_info)

        self.data = fea
        self.adj_type = adj_type

        if self.adj_type == 'histogram':
            self.adj = adj.cuda()
            self.adj_mean = adj_mean.cuda()
        elif self.adj_type == 'sparse':
            self.adj = adj_sp.cuda()
        elif self.adj_type == 'all':
            self.adj = adj.cuda() 
            self.adj_mean = adj_mean.cuda()
            self.adj_sp = adj_sp.cuda() 
        else:
            raise KeyError
    
    def get_label(self, video_info):
        assert isinstance(video_info, np.ndarray)
        label = video_info[:, 0].copy()
        self.nclass = np.unique(label).size
        self.label = torch.from_numpy(label).cuda()

    def get_input(self):
        if self.adj_type == 'histogram':
            return self.data, self.adj, self.adj_mean
        elif self.adj_type == 'sparse':
            return self.data, self.adj
        elif self.adj_type == 'all':
            return self.data, self.adj, self.adj_sp, self.adj_mean
        else:
            raise KeyError



class GCNTrainer(object):
    def __init__(self, config, gcn_class):
        super().__init__()
        self.gcn_cfg = config
        self.max_iter = self.gcn_cfg['iter']

        # self.checkpointer = Checkpointer(
        #     # Assume you want to save checkpoints together with logs/statistics
        #     model,
        #     gcn_cfg.MODEL.OUTPUT,
        #     save_to_disk=comm.is_main_process(),
        #     optimizer=optimizer,
        #     scheduler=self.scheduler
        # )

        self.epoch = 0
        self.iter = 0

        self.gcn_class = gcn_class
        if self.gcn_class.__name__ == 'GCNModel':
            self.adj_type = 'sparse'
        elif self.gcn_class.__name__ == 'GATModel':
            self.adj_type = 'histogram'
        elif self.gcn_class.__name__ == 'GANModel':
            self.adj_type = 'all'
        else:
            raise KeyError
    
    def init_model(self, gcn_class):
        gcn_cfg = self.gcn_cfg
        model = gcn_class(gcn_cfg['model'])
        model.init_classifier(self.dataloader.nclass)
        self.optimizer = default_optimizer(gcn_cfg['optimizer'], model)
        self.model = model.cuda()
        self.model.train()

    def init(self, fea, adj, video_info):
        self.dataloader = GCNDataloader(fea, adj, video_info, self.adj_type)
        self.init_model(self.gcn_class)

    def resume_or_load(self, path=None, resume=True):
        if path is not None:
            path = Path(path)
            if not path.is_absolute():
                path = self.task_dir / path

        checkpoint = self.checkpointer.resume_or_load(path, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_epoch = checkpoint.get('epoch', 0)
            self.start_iter = checkpoint.get("iteration", 0)

    def train(self):
        # self.logger.info('GCN Train for {} Iteration'.format(self.max_iter))
        # self.loss_result_list = []
        for i in range(self.max_iter):
            self.iter = i
            self.run_step()
        self.epoch += 1
        # self.logger.info('GCN Epoch {} Loss : {}'.format(self.epoch, str(self.loss_result_list)))
    
    def inference(self, data, adj):
        dataloader = GCNDataloader(data, adj, video_info=None, adj_type=self.adj_type)
        with inference_context(self.model), torch.no_grad():
            outputs = self.model.inference(dataloader.get_input())
            return outputs

    def run_step(self):

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"

        outputs = self.model(self.dataloader.get_input())
        loss_dict = self.model.losses(outputs, self.dataloader.label)
        losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        losses.backward()

        self.optimizer.step()

        # losses = losses.detach().cpu().numpy()
        # self.loss_result_list.append(float(losses))
