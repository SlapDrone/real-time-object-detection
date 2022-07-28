from typing import Sequence, Optional, Union

import torch
import torch.nn as nn
from pydantic import Field
from pydantic.dataclasses import dataclass
from mmcv.cnn import DepthwiseSeparableConvModule, ConvModule
from mmcv.runner import BaseModule


@dataclass
class PicoDetHead(BaseModule):
    """
    Head of PP-PicoDet (v1)
    
    GFL: generalised focal loss 
    VFL: varifocal loss (classification loss)
    DFL: distribution focal loss (localisation loss)
    
    """
    num_classes: int = Field(
        description="Number of categories excluding the background category"
    )
    in_channels: int = Field(
        description="Number of channels in the input feature map"
    )
    feat_channels: int = Field(
        default=96,
        description="Number of hidden channels in stacking convs"
    )
    stacked_convs: int = Field(
        default=2,
        description="Number of stacked convolutions in the head"
    )
    strides: tuple[int] = Field(
        default=(8, 16, 32, 64),
        description="Downsample factor of each feature map"
    )
    use_depthwise: bool = Field(
        default=True,
        description="Enable depthwise-separable convolutions"
    )
    kernel_size: int = Field(default=5, description="Kernel size of conv layers")
    share_cls_reg: bool = Field(
        default=True,
        description="Flag to share weights between regression and classificaiton branches"
    )
    sigmoid_classifier: bool = Field(
        default=True,
        description="Whether sigmoid loss will be used. This will reduce output channels by 1."
    )
    # I think this is the number of objects allowed assigned to the same prior point?
    reg_max: int = Field(
        default=7,
        description="Max value of integral set {0, ..., reg_max} in DFL setting."
    )
    conv_cfg: dict = Field(
        default=None,
        description="Config dict for 2D convolution layer."
    )
    norm_cfg: dict = Field(
        default_factory=lambda: dict(type='BN', requires_grad=True),
        description="Config dict for 2D convolution layer."
    ) 
    act_cfg: dict = Field(
        default_factory=lambda: dict(type='HSwish'),
        description="Config dict for activation layer."
    )    
    init_cfg: Optional[dict] = Field(
        default=None, description="Weight initialisation config dict"
    )    

    def __post_init__(self):
        pass

    @property
    def ConvModule(self):
        return DepthwiseSeparableConvModule if self.use_depthwise else ConvModule
    
    @property
    def output_channels_classification(self):
        return self.num_classes + (not self.sigmoid_classifier)
    
    def __post_init_post_parse__(self):
        super().__init__(self.init_cfg)
        # build conv layers for interpreting neck outputs - classification
        self.classification_convs = nn.ModuleList([self._build_convs() for _ in self.strides])
        # regression (optional, if not sharing weights from classification)
        self.regression_convs = nn.ModuleList([
            self._build_convs() if not self.share_cls_reg else None for _ in self.strides
        ])
        # generalised focal loss head classification
        self.output_channels_regression = 4 * (self.reg_max + 1)
        # if sharing the weights between classification and regression,
        # the GFL head will calculate both together then split
        self.gfl_classification_conv_out_channels = self.output_channels_classification
        if self.share_cls_reg:
            # C + (x1, x2, y1, y2 + ?)
            self.gfl_classification_conv_out_channels += self.output_channels_regression
        # classification
        self.gfl_classification_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.feat_channels,
                out_channels=self.gfl_classification_conv_out_channels, 
                kernel_size=1,
                padding=0
            )
            for _ in self.strides
        ])
        # regression (optional, if weights shared done together by classificaiton conv)
        self.gfl_regression_convs = nn.ModuleList([
            (
                nn.Conv2d(
                    in_channels=self.feat_channels,
                    out_channels=self.output_channels_regression,
                    kernel_size=1,
                    padding=0
                ) if not self.share_cls_reg else None
            )
            for _ in self.strides
            
        ])
    
    def _build_convs(self):
        """Create a list of self.stacked_convs conv blocks"""
        chn = (lambda i: self.in_channels if not i else self.feat_channels)
        return nn.ModuleList([
            self.ConvModule(
                in_channels=chn(i),
                out_channels=self.feat_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=(self.kernel_size-1) // 2,
                act_cfg=self.act_cfg,
                norm_cfg=self.norm_cfg,
                bias=self.norm_cfg is None
            )
            for i in range(self.stacked_convs)
        ])
    
    def forward(self, features: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        """
        One independent forward pass from each scale branch
        """
        return tuple(
            self.forward_at_scale_index(single_scale_feature, ix)
            for ix, single_scale_feature in enumerate(features)
        )

    def forward_at_scale_index(
        self,
        x: torch.Tensor,
        scale_index: int
    ) -> tuple[torch.Tensor]:
        """
        Compute the forward pass for a given scale branch
        
        Returns
        -------
        (class score, bbox prediction) tensors for scale_index 
        """
        clf_convs = self.classification_convs[scale_index]
        reg_convs = self.regression_convs[scale_index]
        gfl_clf_conv = self.gfl_classification_convs[scale_index]
        gfl_reg_conv = self.gfl_regression_convs[scale_index]
        cls_feat = x
        # classification convs
        for cls_conv in clf_convs:
            cls_feat = cls_conv(cls_feat)
        # if sharing convblocks for classifcation and regression,
        # apply final conv and split out class and location components
        if self.share_cls_reg:
            feat = gfl_clf_conv(cls_feat)
            cls_score, bbox_pred = torch.split(
                feat,
                [
                    self.output_channels_classification,
                    self.output_channels_regression
                ],
                dim=1
            )
        # otherwise compute the separate features for location regression
        else:
            reg_feat = x
            for reg_conv in reg_convs:
                reg_feat = reg_conv(reg_feat)
            cls_score, bbox_pred = gfl_clf_conv(cls_feat), gfl_reg_conv(reg_feat)
        return cls_score, bbox_pred
            