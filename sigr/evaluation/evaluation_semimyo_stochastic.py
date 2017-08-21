from __future__ import division
from .base_evaluation import (
    BaseEvaluation,
    MxNetMixin,
    CacheMixin,
    CrossvalMixin,
    StochasticPredictionMixin
)


class Evaluation(
    StochasticPredictionMixin,
    CrossvalMixin,
    MxNetMixin,
    CacheMixin,
    BaseEvaluation
):
    pass
