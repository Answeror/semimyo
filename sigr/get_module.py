from __future__ import division
from . import utils


_modstack = utils.Modstack(
    formula='.module.module_{name}',
    target='Module',
    package=__package__
)
_modstack_runtime = utils.Modstack(
    formula='.module.module_{name}',
    target='RuntimeModule',
    package=__package__
)


def get_module(name, runtime=False, **kargs):
    modstack = _modstack if not runtime else _modstack_runtime
    return modstack.get(name=name)(**kargs)
