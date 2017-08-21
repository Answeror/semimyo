from __future__ import division
from .... import utils


class InterSessionMixin(object):

    def get_inter_session_data(self, fold, **kargs):
        subject_train = self.subjects[fold * 2]
        subject_val = self.subjects[fold * 2 + 1]
        load = utils.F(self.get_dataiter, **kargs)
        train = load(cls=self.train_dataiter_cls,
                     get_trial_cls=self.train_get_trial_cls,
                     shuffle=True,
                     combos=self.get_combos(subject=[subject_train],
                                            gesture=self.gestures,
                                            trial=self.trials))
        val = load(cls=self.val_dataiter_cls,
                   get_trial_cls=self.val_get_trial_cls,
                   shuffle=False,
                   combos=self.get_combos(subject=[subject_val],
                                          gesture=self.gestures,
                                          trial=self.trials))
        return train, val

    def get_inter_session_val(self, fold, **kargs):
        subject_val = self.subjects[fold * 2 + 1]
        load = utils.F(self.get_dataiter, **kargs)
        val = load(cls=self.val_dataiter_cls,
                   get_trial_cls=self.val_get_trial_cls,
                   shuffle=False,
                   combos=self.get_combos(subject=[subject_val],
                                          gesture=self.gestures,
                                          trial=self.trials))
        return val
