from __future__ import division
from .base_evaluation import BaseEvaluation, MxNetMixin, CacheMixin, CrossvalMixin


class Evaluation(CrossvalMixin, MxNetMixin, CacheMixin, BaseEvaluation):
    pass
