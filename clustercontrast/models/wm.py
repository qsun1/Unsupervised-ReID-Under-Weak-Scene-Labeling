import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd


class WM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, score, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets, score)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, score = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * score * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def wm(inputs, indexes, score, features, momentum=0.5):
    return WM.apply(inputs, indexes, score, features, torch.Tensor([momentum]).to(inputs.device))



class WirelessMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(WirelessMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets):
        inputs = F.normalize(inputs, dim=1).cuda()
        for target, score in enumerate(targets):
            if score != 0:            
                outputs = wm(inputs, target, score, self.features, self.momentum)
                outputs /= self.temp
                loss += score * F.cross_entropy(outputs, target)
        return loss