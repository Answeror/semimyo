from __future__ import division
from .. import base
from .dataset_semg import Dataset as BaseDataset


class Dataset(base.DownsampleMixin, BaseDataset):
    pass
