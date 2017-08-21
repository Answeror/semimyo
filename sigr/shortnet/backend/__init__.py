from __future__ import division
from ..utils import Modstack


_modstack = Modstack(
    formula='.backend_{name}',
    target='Backend',
    package=__package__
)


def Backend(name):
    return _modstack.get(name=name)


__all__ = 'Backend'
