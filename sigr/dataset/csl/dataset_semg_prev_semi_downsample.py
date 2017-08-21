from __future__ import division
from .. import base
from .dataset_semg_prev_semi import Dataset as BaseDataset


class Dataset(base.SemiDownsampleMixin, BaseDataset):
    pass
