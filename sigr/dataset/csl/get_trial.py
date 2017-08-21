from __future__ import division
from ..get_trial import BaseGetTrial as Base, PreprocessMixin, MemoMixin


class BaseGetTrial(PreprocessMixin, MemoMixin, Base):

    memo_room = 30
