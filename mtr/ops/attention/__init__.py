"""
Mostly copy-paste from https://github.com/dvlab-research/DeepVision3D/blob/master/EQNet/eqnet/ops/attention
"""

from . import attention_utils
from . import attention_utils_v2

__all__ = {
    'v1': attention_utils,
    'v2': attention_utils_v2,
}