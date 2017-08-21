from __future__ import division
from . import base


class Dataset(base.SemiDownsampleMixin, base.PrevMixin, base.StepMixin, base.SDataDataset):
    pass
