# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import os
import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule, ModuleList, Sequential
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils import model_zoo
from collections import OrderedDict

import torch.distributed as dist

from ..builder import BACKBONES
from .resnet import BasicBlock, Bottleneck


#load pretrained weight 

open_mmlab_model_urls = {
  "msra/hrnetv2_w18": "https://download.openmmlab.com/pretrain/third_party/hrnetv2_w18-00eb2006.pth",
  "msra/hrnetv2_w32": "https://download.openmmlab.com/pretrain/third_party/hrnetv2_w32-dc9eeb4f.pth",
  "msra/hrnetv2_w40": "https://download.openmmlab.com/pretrain/third_party/hrnetv2_w40-ed0b031c.pth",
  "msra/hrnetv2_w48": "https://download.openmmlab.com/pretrain/third_party/hrnetv2_w48-d2186c55.pth",
}

def get_dist_info():
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        initialized = dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def load_url_dist(url):
    """ In distributed setting, this function only download checkpoint at
    local rank 0 """
    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    if rank == 0:
        checkpoint = model_zoo.load_url(url)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = model_zoo.load_url(url)
    return checkpoint

class HRModule(BaseModule):
    """High-Resolution Module for HRNet.

    In this module, every branch has 4 BasicBlocks/Bottlenecks. Fusion/Exchange
    is in this module.
    """

    def __init__(self,
                 num_branches,
                 blocks,
                 num_blocks,
                 in_channels,
                 num_channels,
                 multiscale_output=True,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 block_init_cfg=None,
                 init_cfg=None):
        super(HRModule, self).__init__(init_cfg)
        self.block_init_cfg = block_init_cfg
        self._check_branches(num_branches, num_blocks, in_channels,
                             num_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches

        self.multiscale_output = multiscale_output
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp
        self.branches = self._make_branches(num_branches, blocks, num_blocks,
                                            num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(self, num_branches, num_blocks, in_channels,
                        num_channels):
        if num_branches != len(num_blocks):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                        f'!= NUM_BLOCKS({len(num_blocks)})'
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                        f'!= NUM_CHANNELS({len(num_channels)})'
            raise ValueError(error_msg)

        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                        f'!= NUM_INCHANNELS({len(in_channels)})'
            raise ValueError(error_msg)

    def _make_one_branch(self,
                         branch_index,
                         block,
                         num_blocks,
                         num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.in_channels[branch_index] != \
                num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    self.in_channels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, num_channels[branch_index] *
                                 block.expansion)[1])

        layers = []
        layers.append(
            block(
                self.in_channels[branch_index],
                num_channels[branch_index],
                stride,
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg,
                init_cfg=self.block_init_cfg))
        self.in_channels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.in_channels[branch_index],
                    num_channels[branch_index],
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    init_cfg=self.block_init_cfg))

        return Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels[j],
                                in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            build_norm_layer(self.norm_cfg, in_channels[i])[1],
                            nn.Upsample(
                                scale_factor=2**(j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[i],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[i])[1]))
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[j])[1],
                                    nn.ReLU(inplace=False)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = 0
            for j in range(self.num_branches):
                if i == j:
                    y += x[j]
                else:
                    y += self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


@BACKBONES.register_module()
class MS_HRNet(BaseModule):
    """HRNet backbone.

    `High-Resolution Representations for Labeling Pixels and Regions
    arXiv: <https://arxiv.org/abs/1904.04514>`_.

    Args:
        extra (dict): Detailed configuration for each stage of HRNet.
            There must be 4 stages, the configuration for each stage must have
            5 keys:

                - num_modules(int): The number of HRModule in this stage.
                - num_branches(int): The number of branches in the HRModule.
                - block(str): The type of convolution block.
                - num_blocks(tuple): The number of blocks in each branch.
                    The length must be equal to num_branches.
                - num_channels(tuple): The number of channels in each branch.
                    The length must be equal to num_branches.
        in_channels (int): Number of input image channels. Default: 3.
        conv_cfg (dict): Dictionary to construct and config conv layer.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: False.
        multiscale_output (bool): Whether to output multi-level features
            produced by multiple branches. If False, only the first level
            feature will be output. Default: True.
        pretrained (str, optional): Model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.

    Example:
        >>> from mmdet.models import HRNet
        >>> import torch
        >>> extra = dict(
        >>>     stage1=dict(
        >>>         num_modules=1,
        >>>         num_branches=1,
        >>>         block='BOTTLENECK',
        >>>         num_blocks=(4, ),
        >>>         num_channels=(64, )),
        >>>     stage2=dict(
        >>>         num_modules=1,
        >>>         num_branches=2,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4),
        >>>         num_channels=(32, 64)),
        >>>     stage3=dict(
        >>>         num_modules=4,
        >>>         num_branches=3,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4, 4),
        >>>         num_channels=(32, 64, 128)),
        >>>     stage4=dict(
        >>>         num_modules=3,
        >>>         num_branches=4,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4, 4, 4),
        >>>         num_channels=(32, 64, 128, 256)))
        >>> self = HRNet(extra, in_channels=1)
        >>> self.eval()
        >>> inputs = torch.rand(1, 1, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 32, 8, 8)
        (1, 64, 4, 4)
        (1, 128, 2, 2)
        (1, 256, 1, 1)
    """

    blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}

    def __init__(self,
                 extra,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=True,
                 with_cp=False,
                 zero_init_residual=False,
                 multiscale_output=True,
                 pretrained=None,
                 init_cfg=None):
        super(MS_HRNet, self).__init__(init_cfg)

        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        # Assert configurations of 4 stages are in extra
        assert 'stage1' in extra and 'stage2' in extra \
               and 'stage3' in extra and 'stage4' in extra
        # Assert whether the length of `num_blocks` and `num_channels` are
        # equal to `num_branches`
        for i in range(4):
            cfg = extra[f'stage{i + 1}']
            assert len(cfg['num_blocks']) == cfg['num_branches'] and \
                   len(cfg['num_channels']) == cfg['num_branches']

        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        # stem net
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(self.norm_cfg, 64, postfix=2)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            self.conv_cfg,
            64,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)
        
        # stage 1
        self.stage1_cfg = self.extra['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]
        block_type = self.stage1_cfg['block']
        num_blocks = self.stage1_cfg['num_blocks'][0]

        block = self.blocks_dict[block_type]
        stage1_out_channels = num_channels * block.expansion
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)

        # stem net_lwir
        self.norm3_name, norm3 = build_norm_layer(self.norm_cfg, 64, postfix=3)
        self.norm4_name, norm4 = build_norm_layer(self.norm_cfg, 64, postfix=4)
        
        self.conv3 = build_conv_layer(
            self.conv_cfg,
            3,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)
        self.conv4 = build_conv_layer(
            self.conv_cfg,
            64,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.add_module(self.norm4_name, norm4)
        
        self.relu = nn.ReLU(inplace=True)

        # stage 1_lwir
        self.stage1_cfg = self.extra['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]
        block_type = self.stage1_cfg['block']
        num_blocks = self.stage1_cfg['num_blocks'][0]

        block = self.blocks_dict[block_type]
        stage1_out_channels = num_channels * block.expansion
        self.layer1_lwir = self._make_layer(block, 64, num_channels, num_blocks)
        
        # stage 2
        self.stage2_cfg = self.extra['stage2']
        num_channels = self.stage2_cfg['num_channels']
        block_type = self.stage2_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition1 = self._make_transition_layer([stage1_out_channels],
                                                       num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        # stage 3
        self.stage3_cfg = self.extra['stage3']
        num_channels = self.stage3_cfg['num_channels']
        block_type = self.stage3_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        # stage 4
        self.stage4_cfg = self.extra['stage4']
        num_channels = self.stage4_cfg['num_channels']
        block_type = self.stage4_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multiscale_output=multiscale_output)
        
        # Fusion layer 
        self.feat_1 = nn.Conv2d(stage1_out_channels*2, stage1_out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_1_bn = nn.BatchNorm2d(stage1_out_channels, momentum=0.01)
        
        # Load pretrained layers
        self.load_pretrained_layers()
        
    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: the normalization layer named "norm3" """
        return getattr(self, self.norm3_name)

    @property
    def norm4(self):
        """nn.Module: the normalization layer named "norm3" """
        return getattr(self, self.norm4_name)
    
    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False),
                            build_norm_layer(self.norm_cfg,
                                             num_channels_cur_layer[i])[1],
                            nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False),
                            build_norm_layer(self.norm_cfg, out_channels)[1],
                            nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1])

        layers = []
        block_init_cfg = None
        if self.pretrained is None and not hasattr(
                self, 'init_cfg') and self.zero_init_residual:
            if block is BasicBlock:
                block_init_cfg = dict(
                    type='Constant', val=0, override=dict(name='norm2'))
            elif block is Bottleneck:
                block_init_cfg = dict(
                    type='Constant', val=0, override=dict(name='norm3'))
        layers.append(
            block(
                inplanes,
                planes,
                stride,
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg,
                init_cfg=block_init_cfg,
            ))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    init_cfg=block_init_cfg))

        return Sequential(*layers)

    def _make_stage(self, layer_config, in_channels, multiscale_output=True):
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = self.blocks_dict[layer_config['block']]

        hr_modules = []
        block_init_cfg = None
        if self.pretrained is None and not hasattr(
                self, 'init_cfg') and self.zero_init_residual:
            if block is BasicBlock:
                block_init_cfg = dict(
                    type='Constant', val=0, override=dict(name='norm2'))
            elif block is Bottleneck:
                block_init_cfg = dict(
                    type='Constant', val=0, override=dict(name='norm3'))

        for i in range(num_modules):
            # multi_scale_output is only used for the last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            hr_modules.append(
                HRModule(
                    num_branches,
                    block,
                    num_blocks,
                    in_channels,
                    num_channels,
                    reset_multiscale_output,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    block_init_cfg=block_init_cfg))

        return Sequential(*hr_modules), in_channels

    def load_pretrained_layers(self, filename = 'open-mmlab://msra/hrnetv2_w32',
                        map_location=None,
                        strict=False,
                        logger=None):

        """Load checkpoint from a file or URI.
        Args:
            model (Module): Module to load checkpoint.
            filename (str): Either a filepath or URL or modelzoo://xxxxxxx.
            map_location (str): Same as :func:`torch.load`.
            strict (bool): Whether to allow different params for the model and
                checkpoint.
            logger (:mod:`logging.Logger` or None): The logger for error message.
        Returns:
            dict or OrderedDict: The loaded checkpoint.
        """

        # load checkpoint from modelzoo or file or url
        if filename.startswith('modelzoo://'):
            import torchvision
            model_urls = dict()
            for _, name, ispkg in pkgutil.walk_packages(
                    torchvision.models.__path__):
                if not ispkg:
                    _zoo = import_module('torchvision.models.{}'.format(name))
                    if hasattr(_zoo, 'model_urls'):
                        _urls = getattr(_zoo, 'model_urls')
                        model_urls.update(_urls)
            model_name = filename[11:]
            checkpoint = load_url_dist(model_urls[model_name])
        elif filename.startswith('open-mmlab://'):
            model_name = filename[13:]
            checkpoint = load_url_dist(open_mmlab_model_urls[model_name])
        elif filename.startswith(('http://', 'https://')):
            checkpoint = load_url_dist(filename)
        else:
            if not osp.isfile(filename):
                raise IOError('{} is not a checkpoint file'.format(filename))
            checkpoint = torch.load(filename, map_location=map_location)

        # get state_dict from checkpoint
        if isinstance(checkpoint, OrderedDict):
            state_dict = checkpoint
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            raise RuntimeError(
                'No state_dict found in checkpoint file {}'.format(filename))

        # strip prefix of state_dict
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}

        pretrained_state_dict = state_dict
        pretrained_param_names = list(pretrained_state_dict.keys())

        # Current state of base
        if hasattr(self, 'module'):
            state_dict = self.module.state_dict()
        else :
            state_dict = self.state_dict()
        param_names = list(state_dict.keys())


        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[:90]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        for i, param in enumerate(param_names[90:180]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]


        for i, param in enumerate(param_names[181:param_names.index('feat_1.weight')-1]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i+91]]

        if hasattr(self, 'module'):
            self.module.load_state_dict(state_dict)
        else :
            self.load_state_dict(state_dict)

        print("\nLoaded base model.\n")
        
        return

    def forward(self, x):

        """Forward function."""

        vis = x[:,:3,:,:]
        lwir = x[:,3:,:,:]

        vis = self.conv1(vis)
        vis = self.norm1(vis)
        vis = self.relu(vis)
        vis = self.conv2(vis)
        vis = self.norm2(vis)
        vis = self.relu(vis)
        vis = self.layer1(vis)

        lwir = self.conv3(lwir)
        lwir = self.norm3(lwir)
        lwir = self.relu(lwir)
        lwir = self.conv4(lwir)
        lwir = self.norm4(lwir)
        lwir = self.relu(lwir)
        lwir = self.layer1_lwir(lwir)

        x = torch.cat([vis, lwir], dim=1)
        x = self.relu(self.feat_1_bn(self.feat_1(x)))

        # x = self.conv1(x)
        # x = self.norm1(x)
        # x = self.relu(x)
        # x = self.conv2(x)
        # x = self.norm2(x)
        # x = self.relu(x)
        # x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        return y_list

    def train(self, mode=True):
        """Convert the model into training mode will keeping the normalization
        layer freezed."""
        super(MS_HRNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()