# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.dense_heads import *
from mmdet.models.dense_heads import __all__ as __mmdet_all__

from .nanodet_head import NanoDetHead
from .picodet_head import PicoDetHead

__all__ = ['NanoDetHead', 'PicoDetHead']
