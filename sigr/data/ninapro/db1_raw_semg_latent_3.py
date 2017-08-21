from __future__ import division
from .db1_raw_semg_latent import Dataset as Base


class Dataset(Base):

    name = 'ninapro-db1-raw/semg-latent-3'
    num_latent = 3
