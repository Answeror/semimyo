from __future__ import division
from . import base


class Dataset(base.PrevWindowMixin, base.StepMixin, base.SDataDataset):
    pass
