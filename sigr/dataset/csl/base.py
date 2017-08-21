from __future__ import division
import os
from itertools import product
from functools import partial
from logbook import Logger
from ... import CACHE, utils
from ...data import Dataset as _BaseDataset
from .. import base, data_iter as di
from .get_trial import BaseGetTrial


logger = Logger(__name__)
Combo = base.Combo
StepMixin = base.StepMixin
DownsampleMixin = base.DownsampleMixin
SemiDownsampleMixin = base.SemiDownsampleMixin
AdaMixin = base.AdaMixin


class BaseIter(di.IndexMixin, di.BalanceGestureMixin, di.NDArrayIter):
    pass


class BaseDataset(
    base.DeterministicMixin,
    base.GetTrialMixin,
    base.DataIterMixin,
    _BaseDataset
):

    framerate = 2048
    num_semg_row = 24
    num_semg_col = 7
    num_trial = 10
    nonrest_trials = list(range(1, 11))
    rest_trials = [2, 4, 7, 8, 11, 13, 19, 25, 26, 30]
    subjects = list(range(1, 6))
    sessions = list(range(1, 6))
    gestures = list(range(27))
    nonrest_gestures = list(range(1, 27))
    rest_gestures = [0]
    num_session = 5
    num_subject = 5
    root = os.path.join(CACHE, 'csl')
    num_trial = 10

    def __init__(self, *args, **kargs):
        self.__dict__.update(kargs)
        kargs = {}
        super(BaseDataset, self).__init__(*args, **kargs)

    @utils.classproperty
    def num_semg_pixel(cls):
        return cls.num_semg_row * cls.num_semg_col

    @classmethod
    def get_preprocess_kargs(cls):
        return dict(
            framerate=cls.framerate,
            num_semg_row=cls.num_semg_row,
            num_semg_col=cls.num_semg_col
        )

    @utils.return_list
    def get_combos(self, subjects, sessions, gestures, trials):
        for subject, session, gesture, trial in product(subjects, sessions, gestures, trials):
            if (subject, session, gesture, trial) in [(4, 4, 8, 10),
                                                      (4, 4, 9, 10)]:
                continue
            yield Combo(subject=subject,
                        session=session,
                        gesture=gesture,
                        trial=trial)

    def good_combo(self, combo):
        if combo.subject in self.subjects:
            if combo.session in self.sessions:
                if combo.gesture in self.nonrest_gestures:
                    return combo.trial in self.nonrest_trials
                elif combo.gesture in self.rest_gestures:
                    return combo.trial in self.rest_trials
        return False

    def get_universal_intra_session_data(self, fold, **kargs):
        #  return self._get_intra_session_data(
            #  fold % self.num_trial,
            #  [self.subjects[fold // self.num_trial]],
            #  self.sessions,
            #  **kargs
        #  )
        return self._get_intra_session_data(
            fold,
            self._parload(self.subjects),
            self._parload(self.sessions),
            **kargs
        )

    def _parload(self, a):
        return a if not getattr(self, 'parload', False) else utils.shuf(a)

    def get_intra_session_data(self, fold, **kargs):
        return self._get_intra_session_data(
            fold % self.num_trial,
            [self.subjects[(fold // self.num_trial) // self.num_session]],
            [self.sessions[(fold // self.num_trial) % self.num_session]],
            **kargs
        )

    def get_intra_session_val(self, fold, **kargs):
        return self._get_intra_session_val(
            fold % self.num_trial,
            [self.subjects[(fold // self.num_trial) // self.num_session]],
            [self.sessions[(fold // self.num_trial) % self.num_session]],
            **kargs
        )

    def _get_intra_session_data(self, fold, subjects, sessions, **kargs):
        nonrest_trial = self.nonrest_trials[fold]
        rest_trial = self.rest_trials[fold]
        load = partial(self.get_dataiter, **kargs)
        train = load(cls=self.train_dataiter_cls,
                     get_trial_cls=self.train_get_trial_cls,
                     shuffle=True,
                     combos=(self.get_combos(subjects=subjects,
                                             sessions=sessions,
                                             gestures=self.nonrest_gestures,
                                             trials=[t for t in self.nonrest_trials if t != nonrest_trial]) +
                             self.get_combos(subjects=subjects,
                                             sessions=sessions,
                                             gestures=self.rest_gestures,
                                             trials=[t for t in self.rest_trials if t != rest_trial])))
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

    def _get_intra_session_val(self, fold, subjects, sessions, **kargs):
        nonrest_trial = self.nonrest_trials[fold]
        rest_trial = self.rest_trials[fold]
        load = partial(self.get_dataiter, **kargs)
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
        return val

    get_trial_cls = BaseGetTrial

    @property
    def dataiter_cls(dataset):
        class cls(
            di.InitMixin(data_shapes=[('semg', (-1, 1, dataset.num_semg_row, dataset.num_semg_col))],
                         label_shapes=[('gesture', (-1,)),
                                       ('subject', (-1,)),
                                       ('session', (-1,))],
                         indices=['gesture', 'subject', 'session']),
            BaseIter
        ):
            pass

        return cls
