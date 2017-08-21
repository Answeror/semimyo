from __future__ import division
import os
from logbook import Logger
from ..base import BaseDataset
from .... import CACHE


logger = Logger(__name__)


class Dataset(BaseDataset):

    root = os.path.join(CACHE, 'dba')
    subjects = list(range(1, 19))
    gestures = list(range(1, 9))
