from __future__ import division
import six


def get_data_iter(dataset, partition, **kargs):
    if isinstance(dataset, six.string_types):
        from .get_dataset import get_dataset
        dataset = get_dataset(dataset)
    return getattr(dataset, 'get_' + partition)(**kargs)
