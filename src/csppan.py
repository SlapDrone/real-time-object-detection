from typing import Optional, Union

import torch.nn as nn
from pydantic import Field
from pydantic.dataclasses import dataclass

from mmcv.cnn import ConvModule


@dataclass
class ChannelEqualiser(nn.Module):
    """
    Given a list of feature maps, uses a kernel size 1 convolution
    to equalise the channel numbers across them.
    """
    in_channels: list[int, int, int]
    out_channels: int
    act_cfg: dict = Field(default_factory=(lambda: dict(type="LeakyReLU")))
    conv_cfg: Optional[dict] = None
    norm_cfg: Optional[dict] = None
        
    def __post_init__(self) -> None:
        super().__init__()
        
    def __post_init_post_parse__(self) -> None:
        self.convs = nn.ModuleList([
            ConvModule(
                in_channels=c,
                out_channels=self.out_channels,
                kernel_size=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False
            )
            for ix, c in enumerate(self.in_channels)
        ])
        
    def forward(self, x):
        return [c(x[i]) for i, c in enumerate(self.convs)]