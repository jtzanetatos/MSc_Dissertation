# -*- coding: utf-8 -*-
"""

"""

from ._histogram_operations import KernelsHist
from ._histogram_operations import HistBpf
from ._histogram_operations import HistNorm
from ._histogram_operations import HistWindows


__all__ = [
    'KernelsHist',
    'HistBpf',
    'HistNorm',
    'HistWindows'
]