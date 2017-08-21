from __future__ import division
from .. import utils
from .base import BaseGetTrial
import numpy as np


class PreprocessMixin(object):

    def __init__(self, **kargs):
        self.preprocess = kargs.pop('preprocess', None)
        super(PreprocessMixin, self).__init__(**kargs)


class MemoMixin(object):

    @property
    def memo_room(self):
        return self.dataset.num_trial

    def __init__(self, *args, **kargs):
        super(MemoMixin, self).__init__(*args, **kargs)
        self.memo = utils.make_lru_dict(self.memo_room)

    def has_memo(self, obj):
        return utils.hash(obj) in self.memo

    def get_memo(self, obj):
        if not self.has_memo(obj):
            raise KeyError(str(obj))
        return self.memo[utils.hash(obj)]

    def set_memo(self, obj, value):
        self.memo[utils.hash(obj)] = value

    def __call__(self, combo):
        if not self.has_memo(combo):
            self.update_memo(combo)
        return self.get_memo(combo)


class RandomStateMixin(object):

    def __init__(self, **kargs):
        super(RandomStateMixin, self).__init__(**kargs)
        self.random_state = kargs.pop('random_state', np.random.RandomState(1927))


__all__ = ['BaseGetTrial', 'PreprocessMixin', 'MemoMixin']
