# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.backbones import *
from mmdet.models.backbones import __all__ as __mmdet_all__

from .shufflenetv2 import ShuffleNetV2
from .esnet import ESNet

__all__ = __mmdet_all__ + [
    'ShuffleNetV2', 'ESNet'
]
