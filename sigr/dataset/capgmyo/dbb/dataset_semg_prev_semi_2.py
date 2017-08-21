from __future__ import division
import os
from .base import InterSessionMixin
from ..base import PrevSemiDataset
from .... import CACHE, utils


class Dataset(InterSessionMixin, PrevSemiDataset):

    root = os.path.join(CACHE, 'dbb')
    subjects = list(range(1, 21))
    gestures = list(range(1, 9))

    def get_inter_session_ada_data(self, fold, **kargs):
        subject_train = self.subjects[fold * 2]
        subject_val = self.subjects[fold * 2 + 1]
        load = utils.F(self.get_dataiter, **kargs)
        train = load(cls=self.train_dataiter_cls,
                     get_trial_cls=self.get_ada_get_trial_cls(subject_val),
                     shuffle=True,
                     combos=self.get_combos(subject=[subject_train, subject_val],
                                            gesture=self.gestures,
                                            trial=self.trials))
        val = load(cls=self.val_dataiter_cls,
                   get_trial_cls=self.val_get_trial_cls,
                   shuffle=False,
                   combos=self.get_combos(subject=[subject_val],
                                          gesture=self.gestures,
                                          trial=self.trials))
        return train, val
