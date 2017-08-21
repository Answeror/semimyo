from __future__ import division
from . import base


class Dataset(base.PrevWindowRowMixin, base.StepMixin, base.SDataDataset):
    pass
