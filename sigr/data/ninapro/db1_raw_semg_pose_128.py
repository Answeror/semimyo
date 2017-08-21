from __future__ import division
from .db1_raw_semg_pose import Dataset as Base


class Dataset(Base):

    name = 'ninapro-db1-raw/semg-pose-128'
    num_pose = 128
