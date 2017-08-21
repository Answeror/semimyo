from __future__ import division
from . import utils


_modstack = utils.Modstack(
    formula='.dataset.{name}',
    target='Dataset',
    package=__package__
)


def get_dataset(name, **kargs):
    try:
        Dataset = _modstack.get(name=_parse_name(name))
    except ImportError:
        raise
        #  Deprecated approach
        from .data import Dataset
        return Dataset.parse(name)
    return Dataset(**kargs)


def _parse_name(name):
    parts = name.split('.')
    parts[-1] = 'dataset_' + parts[-1]
    return '.'.join(parts)
