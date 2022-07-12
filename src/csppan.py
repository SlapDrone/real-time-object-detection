from typing import Optional, Union

import torch.nn as nn
from pydantic import Field
from pydantic.dataclasses import dataclass

from mmcv.cnn import ConvModule


@dataclass
class DarknetBottleneck(BaseModule):
    in_channels: int = Field(
        description="The input channels of this Module.")
    out_channels: int = Field(
        description="The output channels of this Module.")
    kernel_size: int = Field(
        default=1, description="The kernel size of the convolution.")
    expansion: int = Field(
        default=0.5, description="Hidden conv block channels relative to output")
    add_identity: bool = Field(
        default=True, description="Whether to add identity to the out.")
    use_depthwise: bool = Field(
        default=False, description="Whether to use depthwise separable convolution.")
    conv_cfg: dict = Field(
        default=None, description="Config dict for 2D convolution layer.")
    norm_cfg: dict = Field(
        default_factory=lambda: dict(type='BN'), description="Config dict for 2D convolution layer.") 
    act_cfg: dict = Field(
        default_factory=lambda: dict(type='Swish'), description="Config dict for activation layer.")

    def __post_init__(self) -> None:
        super().__init__()
        
    def __post_init_post_parse__(self) -> None:
        self.hidden_channels = int(self.out_channels * self.expansion)
        self.conv = DepthwiseSeparableConvModule if self.use_depthwise else ConvModule
        self.conv1 = ConvModule(
            self.in_channels,
            self.hidden_channels,
            kernel_size=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        self.conv2 = self.conv(
            self.hidden_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=(self.kernel_size - 1) // 2,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
    
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        input_ = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.add_identity and self.in_channels == self.out_channels:
            return out + input_
        return out


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
            for c in self.in_channels
        ])
        
    def forward(self, x) -> list[torch.Tensor]:
        return [c(x[i]) for i, c in enumerate(self.convs)]