import torch.optim as optim
import logging


__all__ = ['default_optimizer']


def default_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg['lr']
        weight_decay = cfg['weight_decay']
        if "heads" in key:
            lr *= 1.0
        if "bias" in key:
            lr *= 2.0
            weight_decay = 0.0
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg['opt'] == 'Adam':
        print('Using Adam optimizer. lr {} weight decay {}'.format(cfg['lr'], cfg['weight_decay']))
        return optim.Adam(params)
    elif cfg['opt'] == 'SGD':
        print(
            'Using SGD optimizer. lr {} weight decay {} momentum {}'.format(cfg['lr'], cfg['weight_decay'], cfg['momentum']))
        return optim.SGD(params, momentum=cfg['momentum'])
    else:
        raise KeyError('No optimizer method for {}'.format(cfg['opt']))