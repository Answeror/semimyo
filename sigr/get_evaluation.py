from __future__ import division
from . import utils


_modstack = utils.Modstack(
    formula='.evaluation.evaluation_{name}',
    target='Evaluation',
    package=__package__
)


def get_evaluation(name, **kargs):
    return _modstack.get(name=name)(**kargs)
