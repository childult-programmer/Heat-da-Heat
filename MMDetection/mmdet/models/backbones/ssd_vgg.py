# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import copy
import torch
import torch.nn as nn
from mmcv.cnn import VGG
from mmcv.runner import BaseModule
import numpy as np
import torchvision

from ..builder import BACKBONES
from ..necks import ssd_neck

import logging

#from .utils import constant_init, kaiming_init, normal_init

def decimate(tensor, m):

    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor



@BACKBONES.register_module()
class SSDVGG(VGG, BaseModule):
    """VGG Backbone network for single-shot-detection.

    Args:
        depth (int): Depth of vgg, from {11, 13, 16, 19}.
        with_last_pool (bool): Whether to add a pooling layer at the last
            of the model
        ceil_mode (bool): When True, will use `ceil` instead of `floor`
            to compute the output shape.
        out_indices (Sequence[int]): Output from which stages.
        out_feature_indices (Sequence[int]): Output from which feature map.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
        input_size (int, optional): Deprecated argumment.
            Width and height of input, from {300, 512}.
        l2_norm_scale (float, optional) : Deprecated argumment.
            L2 normalization layer init scale.

    Example:
        >>> self = SSDVGG(input_size=300, depth=11)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 300, 300)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 1024, 19, 19)
        (1, 512, 10, 10)
        (1, 256, 5, 5)
        (1, 256, 3, 3)
        (1, 256, 1, 1)
    """
    extra_setting = {
        300: (256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256),
        512: (256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128),
    }

    def __init__(self,
                 depth,
                 with_last_pool=False,
                 ceil_mode=True,
                 out_indices=(3, 4),
                 out_feature_indices=(22, 34),
                 pretrained=None,
                 init_cfg=None,
                 input_size=None,
                 l2_norm_scale=None):
        # TODO: in_channels for mmcv.VGG
        super(SSDVGG, self).__init__(
            depth,
            with_last_pool=with_last_pool,
            ceil_mode=ceil_mode,
            out_indices=out_indices)

        self.features.add_module(
            str(len(self.features)),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        self.features.add_module(
            str(len(self.features)),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6))
        self.features.add_module(
            str(len(self.features)), nn.ReLU(inplace=True))
        self.features.add_module(
            str(len(self.features)), nn.Conv2d(1024, 1024, kernel_size=1))
        self.features.add_module(
            str(len(self.features)), nn.ReLU(inplace=True))
 
        
        self.out_feature_indices = out_feature_indices

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'

        if init_cfg is not None:
            self.init_cfg = init_cfg
        elif isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='Conv2d'),
                dict(type='Constant', val=1, layer='BatchNorm2d'),
                dict(type='Normal', std=0.01, layer='Linear'),
            ]
        else:
            raise TypeError('pretrained must be a str or None')

        if input_size is not None:
            warnings.warn('DeprecationWarning: input_size is deprecated')
        if l2_norm_scale is not None:
            warnings.warn('DeprecationWarning: l2_norm_scale in VGG is '
                          'deprecated, it has been moved to SSDNeck.')
            
        self.features_lwir = copy.deepcopy(self.features)
        
        self.init_weights()

        
    def forward(self, x):
        """Forward function."""
        #import pdb;pdb.set_trace()
        vis = x[:, :3, :, :]

        vis_outs = []

        
        for i, layer in enumerate(self.features):
            vis = layer(vis)
            if i in self.out_feature_indices:
                vis_outs.append(vis)


        if len(vis_outs) == 1:
            return vis_outs[0]
        else:
            return tuple(vis_outs)

        
    def init_weights(self):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())
        
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())
        
        for i, param in enumerate(param_names[:26]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]
            
        for i, param in enumerate(param_names[30:-4]):
            #import pdb;pdb.set_trace()
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]
            
        # fc6
        #import pdb;pdb.set_trace()
        conv_fc6_rgb_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
        conv_fc6_rgb_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
        state_dict['features.31.weight'] = decimate(conv_fc6_rgb_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['features.31.bias'] = decimate(conv_fc6_rgb_bias, m=[4])  # (1024)
        
        conv_fc6_lwir_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
        conv_fc6_lwir_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
        state_dict['features_lwir.31.weight'] = decimate(conv_fc6_lwir_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['features_lwir.31.bias'] = decimate(conv_fc6_lwir_bias, m=[4])  # (1024)
        # fc7
        
        #import pdb;pdb.set_trace()
        conv_fc7_rgb_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_rgb_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['features.33.weight'] = decimate(conv_fc7_rgb_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['features.33.bias'] = decimate(conv_fc7_rgb_bias, m=[4])  # (1024)
        
        conv_fc7_lwir_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_lwir_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['features_lwir.33.weight'] = decimate(conv_fc7_lwir_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['features_lwir.33.bias'] = decimate(conv_fc7_lwir_bias, m=[4])  # (1024)
        
        self.load_state_dict(state_dict)


class L2Norm(ssd_neck.L2Norm):

    def __init__(self, **kwargs):
        super(L2Norm, self).__init__(**kwargs)
        warnings.warn('DeprecationWarning: L2Norm in ssd_vgg.py '
                      'is deprecated, please use L2Norm in '
                      'mmdet/models/necks/ssd_neck.py instead')
