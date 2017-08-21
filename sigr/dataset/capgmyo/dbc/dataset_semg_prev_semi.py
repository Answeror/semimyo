from __future__ import division
import os
from ..base import PrevSemiDataset
from .... import CACHE


class Dataset(PrevSemiDataset):

    root = os.path.join(CACHE, 'dbc')
    subjects = list(range(1, 11))
    gestures = list(range(1, 13))
