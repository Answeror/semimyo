from __future__ import division
import os
from .. import base
from .... import CACHE


class Dataset(base.SemiDownsampleMixin, base.PrevSemiDataset):

    root = os.path.join(CACHE, 'dba')
    subjects = list(range(1, 19))
    gestures = list(range(1, 9))
