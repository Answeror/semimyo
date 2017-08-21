from __future__ import division
from . import base


class Dataset(
    base.CaputoMixin,
    base.PosePrevMixin,
    base.StepMixin,
    base.BaseDataset
):
    pass
