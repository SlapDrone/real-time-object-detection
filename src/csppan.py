from typing import Optional, Union

import torch
import torch.nn as nn
from pydantic import Field
from pydantic.dataclasses import dataclass

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule


@dataclass
class DarknetBottleneck(BaseModule):
    in_channels: int = Field(
        description="The input channels of this Module.")
    out_channels: int = Field(
        description="The output channels of this Module.")
    kernel_size: int = Field(
        default=1, description="The kernel size of the convolution.")
    squeeze_ratio: float = Field(
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
        self.hidden_channels = int(self.out_channels * self.squeeze_ratio)
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


@dataclass
class CSPLayer(BaseModule):
    """
    Cross-stage partial layer
    """
    in_channels: int = Field(
        description="The input channels of this Module.")
    out_channels: int = Field(
        description="The output channels of this Module.")
    kernel_size: int = Field(
        default=1, description="The kernel size of the convolution.")
    num_blocks: int = Field(
        default=1, description="Number of darknet blocks")
    squeeze_ratio: float = Field(
        default=0.5, description="Hidden conv block channels relative to output")
    darknet_squeeze_ratio: float = Field(
        default=0.5, description="Darknet block hidden conv block channels relative to output")
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
    init_cfg: Optional[dict] = Field(
        default=None, description="Weight initialisation config dict")
        
    def __post_init__(self):
        pass
    
    def __post_init_post_parse__(self):
        super().__init__(self.init_cfg)
        self.mid_channels = int(self.out_channels * self.squeeze_ratio)
        conv_module_kwargs = dict(
            in_channels=self.in_channels,
            out_channels=self.mid_channels,
            kernel_size=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        self.main_conv = ConvModule(**conv_module_kwargs)
        self.short_conv = ConvModule(**conv_module_kwargs)
        conv_module_kwargs.update(
            in_channels=2*self.mid_channels,
            out_channels=self.out_channels
        )
        self.final_conv = ConvModule(**conv_module_kwargs)
        self.dn_blocks = nn.Sequential(*[
            DarknetBottleneck(
                self.mid_channels,
                self.mid_channels,
                kernel_size=self.kernel_size,
                squeeze_ratio=self.darknet_squeeze_ratio,
                use_depthwise=self.use_depthwise,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
            for _ in range(self.num_blocks)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        x_main = self.dn_blocks(x_main)
        x_final = torch.cat([x_main, x_short], dim=1)
        return self.final_conv(x_final)
        

@dataclass
class CSPPAN(BaseModule):
    """
    Path aggregation network with CSP module
    """
    in_channels: list[int] = Field(description="Number of input channels per scale")
    out_channels: int = Field(description="Number of output channels (same at each scale)")
    kernel_size: int = Field(default=5, description="Conv2D kernel size")
    squeeze_ratio: float = Field(default=0.5, description="Conv block mid channels as fraction of output")
    num_csp_blocks: int = Field(default=1, description="Number of bottlenecks in CSPLayer")
    use_depthwise: bool = Field(default=True, description="Use depthwise separable in blocks")
    spatial_scales: list[int] = Field(default=[1/8, 1/16, 1/32], description="")
    upsample_cfg: dict = Field(
        default_factory=lambda: dict(scale_factor=2, mode="nearest"),
        description="Constructor kwargs for upsampling layers"
    )
    conv_cfg: Optional[dict] = Field(default=None)
    norm_cfg: dict = Field(default_factory=lambda: dict(type="BN"))
    act_cfg: dict = Field(default_factory=lambda: dict(type='LeakyReLU'))
        
    def __post_init__(self) -> None:
        super().__init__()
    
    def __post_init_post_parse__(self) -> None:
        # layer to normalise channels of backbone inputs
        self.ch_equaliser = ChannelEqualiser(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            act_cfg=self.act_cfg,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg
        )
        # downsampling convolution class
        self.conv = DepthwiseSeparableConvModule if self.use_depthwise else ConvModule
        # simple upsampling
        self.upsample = nn.Upsample(**self.upsample_cfg)
        csp_kwargs = dict(
            in_channels=2*self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            squeeze_ratio=self.squeeze_ratio,
            num_blocks=self.num_csp_blocks,
            add_identity=False,
            use_depthwise=self.use_depthwise,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        # Cross-stage pooling layers for top down/bottom up feature fusion
        self.top_down_blocks = nn.ModuleList([
            CSPLayer(**csp_kwargs) for _ in range(len(self.in_channels))
        ])
        self.bottom_up_blocks = nn.ModuleList([
            CSPLayer(**csp_kwargs) for _ in range(len(self.in_channels))
        ])
        # downsampling convolutions for bottom-up path
        ds_conv_kwargs = dict(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=2,
            padding=(self.kernel_size - 1) // 2,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        self.downsamples = nn.ModuleList([
            self.conv(**ds_conv_kwargs) for _ in range(len(self.in_channels))
        ])
        # top convolutions (downsampling pre- and post-fusion top layers for final output)
        self.top_conv_pre_fusion = self.conv(**ds_conv_kwargs)
        self.top_conv_post_fusion = self.conv(**ds_conv_kwargs)
        
    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        inputs are L->R bottom to top (large->small)
        
        See Also
        --------
        Fig. 2 in https://arxiv.org/pdf/2111.00902.pdf
        """
        assert len(inputs) == len(self.in_channels)
        # extract the same number of channels from each input fmap
        inputs = self.ch_equaliser(inputs)
        # convention: fmaps are stacked in analogy w/ pyramid: top deepest
        # top-down feature fusion
        td_outputs = []
        for ix, fmap in enumerate(reversed(inputs)):
            # at top of pyramid multi-stage fusion impossible, assign output & go down a level
            if not ix:
                td_outputs.append(fmap)
                continue
            # upsample output feature map of previous stage, concat with current
            td_input = torch.cat([self.upsample(td_outputs[-1]), fmap], axis=1)
            # each set of multi-stage features go into their own CSP block, assign output
            # NOTE: sign difference from mmdet implementation here; first top down block is now the top!
            td_outputs.append(self.top_down_blocks[ix-1](td_input))
        # bottom-up feature fusion (with outputs from prev top-down stage)
        bu_outputs = []
        for ix, fmap in enumerate(reversed(td_outputs)):
            if not ix:
                bu_outputs.append(fmap)
                continue
            bu_input = torch.cat([self.downsamples[ix](bu_outputs[-1]), fmap], axis=1)
            bu_outputs.append(self.bottom_up_blocks[ix-1](bu_input))
        # top features
        final_output = self.top_conv_pre_fusion(inputs[-1]) + self.top_conv_post_fusion(bu_outputs[-1])
        return (*bu_outputs, final_output)
            