from __future__ import division
from ... import utils
from .dataset_semg_prev_semi_downsample import Dataset as BaseDataset


class Dataset(BaseDataset):

    def get_intra_session_data(self, fold, **kargs):
        subjects = [self.subjects[(fold // self.num_trial) // self.num_session]]
        sessions = [self.sessions[(fold // self.num_trial) % self.num_session]]
        fold = fold % self.num_trial

        nonrest_trial = self.nonrest_trials[fold]
        rest_trial = self.rest_trials[fold]
        load = utils.F(self.get_dataiter, **kargs)
        train = load(cls=self.train_dataiter_cls,
                     get_trial_cls=self.train_get_trial_cls,
                     shuffle=True,
                     combos=(self.get_combos(subjects=subjects,
                                             sessions=sessions,
                                             gestures=self.nonrest_gestures,
                                             trials=[nonrest_trial]) +
                             self.get_combos(subjects=subjects,
                                             sessions=sessions,
                                             gestures=self.rest_gestures,
                                             trials=[rest_trial])))
        val = load(cls=self.val_dataiter_cls,
                   get_trial_cls=self.val_get_trial_cls,
                   shuffle=False,
                   combos=(self.get_combos(subjects=subjects,
                                           sessions=sessions,
                                           gestures=self.nonrest_gestures,
                                           trials=[nonrest_trial]) +
                           self.get_combos(subjects=subjects,
                                           sessions=sessions,
                                           gestures=self.rest_gestures,
                                           trials=[rest_trial])))
        return train, val
