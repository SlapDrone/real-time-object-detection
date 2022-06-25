"""
ESNet Backbone network from:

https://arxiv.org/pdf/2111.00902.pdf
"""
from pathlib import Path
from typing import Sequence, Optional, Union
from pydantic.dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmdet.models.utils import make_divisible
from mmcv.cnn.bricks.activation import ACTIVATION_LAYERS
from mmdet.models.utils.inverted_residual import EnhancedInvertedResidualDS, EnhancedInvertedResidual  
from torch.nn import Hardswish


@dataclass
class ESNetSize:
    scale: float
    channel_ratios: list[float]
        
Small = ESNetSize(
    scale=0.75,
    channel_ratios=[
        0.875, 0.5, 0.5, 0.5, 0.625, 0.5, 0.625,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5
    ]
)

Medium = ESNetSize(
    scale=1.0,
    channel_ratios=[
        0.875, 0.5, 1.0, 0.625, 0.5, 0.75, 0.625, 
        0.625, 0.5, 0.625, 1.0, 0.625, 0.75
    ]
)

Large = ESNetSize(
    scale=1.25,
    channel_ratios=[
        0.875, 0.5, 1.0, 0.625, 0.5, 0.75, 0.625, 
        0.625, 0.5, 0.625, 1.0, 0.625, 0.75
    ]
)

ACTIVATION_LAYERS.register_module(Hardswish, force=True)


class ESNet(BaseModule):
    """
    Enhanced ShuffleNet used in PicoDet
    
    Parameters
    ----------
    model_size : 
        String "s", "m", or "l"
    out_indices : 
        Output from which stages
    frozen_stages : 
        Stages to be frozen (stop grad and set eval mode). -1 => none.
    conv_cfg : 
        Optional config dict for convolution layer.
    norm_cfg : 
        Dict config for norm layer
    act_cfg : 
        Dict config for activation layer
    norm_eval : 
        Set norm layers to eval mode (freeze BN running stats)
    se_cfg : 
        Config dict from SE layer
    with_cp : 
        Use (weight) checkpointing or not (save VRAM, slow down training)
    pretrained : 
        Optional path to pretrained model weights
    init_cfg : 
        Initialisation config dict.
    """
    stage_repeats = [3, 7, 3]
    
    def __init__(
        self,
        model_size: ESNetSize = Small,
        frozen_stages: int = -1,
        conv_cfg: Optional[dict] = None,
        norm_cfg: dict = dict(type="BN", requires_grad=True),
        act_cfg: dict = dict(type="Hardswish"),
        norm_eval: bool = False,
        se_cfg: dict = dict(
            conv_cfg=None,
            ratio=4,
            act_cfg=(
                dict(type="ReLU"),
                dict(type="HSigmoid")
            )
        ),
        with_cp: bool = False,
        init_cfg: Optional[Union[str, dict, list[dict]]] = None
    ):
        # weight initialisation from MMCV basemodule
        super().__init__(init_cfg)
        # set initialisations for different layers
        if isinstance(init_cfg, str):
            self.init_cfg = dict(type='Pretrained', checkpoint=init_cfg)
        elif init_cfg is None:
            self.init_cfg = self._init_cfg_default()
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.se_cfg = se_cfg
        self.with_cp = with_cp
        self.model_size = model_size
        # define layers
        # initial (downsampling) conv
        self.conv_initial = ConvModule(
            in_channels=3,
            out_channels=self.stage_out_channels[0],
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.blocks = self._create_bottleneck_blocks()
        
    @property
    def scale(self):
        return self.model_size.scale
    
    @property
    def channel_ratios(self):
        return self.model_size.channel_ratios
    
    @property
    def frozen_stages(self):
        return self._frozen_stages
    
    @property
    def stage_out_channels(self):
        return [
            24,
            make_divisible(128 * self.scale, divisor=16),
            make_divisible(256 * self.scale, divisor=16),
            make_divisible(512 * self.scale, divisor=16),
            1024
        ]
    
        
    @frozen_stages.setter
    def frozen_stages(self, value):
        self._validate_frozen_stages(value)
        self._frozen_stages = value
        
    def _validate_out_indices(self, value):
        if not set(value).issubset(set(range(1, 15))):
            raise ValueError('out_indices must be a subset of range'
                             f'[1, 15). But received {value}')
            
    def _validate_frozen_stages(self, value):
        if value not in range(-1, 4):
            raise ValueError('frozen_stages must be in range(-1, 4). '
                             f'But received {value}')
            
    def _init_cfg_default(self):
        return [
            dict(type='Kaiming', layer='Conv2d'),
            dict(
                type='Constant',
                val=1,
                layer=['_BatchNorm', 'GroupNorm']
            )
        ]
        
    def _create_bottleneck_blocks(self):
        blocks = []
        # linear index (points back to architecture's channel_ratios)
        arch_ix = 0
        self.out_ixs = []
        # always 3 stages with (3, 7, 3) repeated blocks respectively
        for stage_ix, num_repeats in enumerate(self.stage_repeats):
            for repeat_ix in range(num_repeats):
                channel_scale_factor = self.channel_ratios[arch_ix]
                mid_channels = make_divisible(
                    int(self.stage_out_channels[stage_ix + 2] * channel_scale_factor),
                    divisor=8
                )
                # first block in each stage is special case: downsampling
                if not repeat_ix:
                    # TODO: do i need to assign this to attr?
                    self.se_cfg["channels"] = mid_channels // 2
                    block = EnhancedInvertedResidualDS(
                        in_channels=self.stage_out_channels[stage_ix],
                        mid_channels=mid_channels,
                        out_channels=self.stage_out_channels[stage_ix + 1],
                        stride=2,
                        se_cfg=self.se_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        with_cp=self.with_cp,
                        init_cfg=self.init_cfg
                    )
                # no downsampling
                else:
                    self.se_cfg["channels"] = mid_channels
                    block = EnhancedInvertedResidual(
                        in_channels=self.stage_out_channels[stage_ix + 1],
                        mid_channels=mid_channels,
                        out_channels=self.stage_out_channels[stage_ix + 1],
                        stride=1,
                        se_cfg=self.se_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        with_cp=self.with_cp,
                        init_cfg=self.init_cfg
                    )
                name = str(stage_ix + 1) + "_" + str(repeat_ix + 1)
                setattr(self, name, block)
                block.name = name
                blocks.append(block)
                arch_ix += 1
            # after each set of repeating blocks; output
            self.out_ixs.append(arch_ix)
        return blocks
    
    def forward(self, x):
        out = self.conv_initial(x)
        out = self.max_pool(out)
        outs = []
        for i, block in enumerate(self.blocks):
            out = block(out)
            # skip lines for multi-scale interception in neck
            if i in self.out_ixs:
                outs.append(out)
        return outs
    
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            block_num = self.stage_repeats[i]
            for num in range(block_num):
                layer = getattr(self, f'{i + 1}_{num + 1}')
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False
    
    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        super(ESNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()



# refactor to unit test
if __name__ == "__main__":
    # Small
    esnet = ESNet()
    test_input = torch.from_numpy(np.random.rand(1, 3, 320, 320).astype(np.float32))
    test_outputs = esnet(test_input)
    print([a.shape for a in test_output])