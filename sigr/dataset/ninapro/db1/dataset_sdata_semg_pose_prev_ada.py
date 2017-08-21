from __future__ import division
from . import base


class Dataset(
    base.AdaMixin('gesture', 'pose'),
    base.PosePrevMixin,
    base.StepMixin,
    base.SDataDataset
):
    pass
