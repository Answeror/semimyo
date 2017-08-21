from __future__ import division
from . import utils


_modstack = utils.Modstack(
    formula='.symbol.symbol_{name}',
    target='get_symbol',
    package=__package__
)


def get_symbol(name, **kargs):
    return _modstack.get(name=name)(**kargs)
