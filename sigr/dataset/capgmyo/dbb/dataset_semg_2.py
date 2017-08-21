from __future__ import division
import os
from logbook import Logger
from .base import InterSessionMixin
from ..base import BaseDataset
from .... import CACHE


logger = Logger(__name__)


class Dataset(InterSessionMixin, BaseDataset):

    root = os.path.join(CACHE, 'dbb')
    subjects = list(range(1, 21))
    gestures = list(range(1, 9))
