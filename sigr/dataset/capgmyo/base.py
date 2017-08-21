from __future__ import division
from functools import partial
from itertools import product
from logbook import Logger
import nose.tools as nt
import scipy.io as sio
import numpy as np
import os
from ...data.capgmyo import Dataset as _BaseDataset
from ... import utils
from .. import base, data_iter as di


logger = Logger(__name__)
Combo = base.Combo
StepMixin = base.StepMixin
DownsampleMixin = base.DownsampleMixin
SemiDownsampleMixin = base.SemiDownsampleMixin


class BaseGetTrial(base.BaseGetTrial):

    def __init__(self, preprocess=None, **kargs):
        super(BaseGetTrial, self).__init__(**kargs)
        self.preprocess = preprocess


class MemoGetTrialMixin(object):

    def __init__(self, *args, **kargs):
        super(MemoGetTrialMixin, self).__init__(*args, **kargs)
        self.memo = utils.make_lru_dict(2 * len(self.dataset.trials))


class RawGetTrialMixin(object):

    n = 1000

    def get_path(self, combo):
        return os.path.join(
            self.dataset.root,
            '{c.subject:03d}-{c.gesture:03d}.mat'.format(c=combo)
        )

    def _make_trial(self, gesture):
        breaks = list(np.where(gesture[:-1] != gesture[1:])[0] + 1)
        if gesture[0] > 0:
            breaks.append(0)
        if gesture[-1] > 0:
            breaks.append(len(gesture))
        trial = np.zeros(len(gesture), dtype=np.int)
        count = 0
        for begin, end in zip(breaks[:-1], breaks[1:]):
            g = gesture[end - 1]
            if g > 0:
                if end - begin < self.n:
                    logger.info('Ignore short segment: {}', end - begin)
                else:
                    count += 1
                    trial[begin:end] = count
        return trial


class BaseDataset(base.DeterministicMixin, _BaseDataset):

    framerate = 1000
    num_semg_row = 16
    num_semg_col = 8
    num_trial = 10
    trials = list(range(1, 11))
    train_trials = list(range(1, 11, 2))
    val_trials = list(range(2, 11, 2))

    @property
    def num_semg_pixel(self):
        return self.num_semg_row * self.num_semg_col

    def __init__(self, *args, **kargs):
        self.__dict__.update(kargs)
        kargs = {}
        kargs['root'] = self.__class__.root
        super(BaseDataset, self).__init__(*args, **kargs)

    @classmethod
    def get_preprocess_kargs(cls):
        return dict(
            framerate=cls.framerate,
            num_semg_row=cls.num_semg_row,
            num_semg_col=cls.num_semg_col
        )

    @utils.return_list
    def get_combos(self, subject, gesture, trial):
        for subject, gesture, trial in product(subject, gesture, trial):
            yield Combo(subject=subject, gesture=gesture, trial=trial)

    def get_universal_intra_subject_data(self, fold, **kargs):
        nt.assert_equal(fold, 0)
        return self._get_intra_subject_data(self.subjects, **kargs)

    def get_intra_subject_data(self, fold, **kargs):
        subject = self.subjects[fold]
        return self._get_intra_subject_data([subject], **kargs)

    def get_intra_subject_val(self, fold, **kargs):
        subject = self.subjects[fold]
        return self._get_intra_subject_val([subject], **kargs)

    def _get_intra_subject_data(self, subjects, **kargs):
        load = partial(self.get_dataiter, **kargs)
        train = load(cls=self.train_dataiter_cls,
                     get_trial_cls=self.train_get_trial_cls,
                     shuffle=True,
                     combos=self.get_combos(subject=subjects,
                                            gesture=self.gestures,
                                            trial=self.train_trials))
        val = load(cls=self.val_dataiter_cls,
                   get_trial_cls=self.val_get_trial_cls,
                   shuffle=False,
                   combos=self.get_combos(subject=subjects,
                                          gesture=self.gestures,
                                          trial=self.val_trials))
        return train, val

    def get_inter_subject_data(self, fold, **kargs):
        subject = self.subjects[fold]
        load = partial(self.get_dataiter, **kargs)
        train = load(cls=self.train_dataiter_cls,
                     get_trial_cls=self.train_get_trial_cls,
                     shuffle=True,
                     combos=self.get_combos(subject=utils.exclude(self.subjects, [subject]),
                                            gesture=self.gestures,
                                            trial=self.trials))
        val = load(cls=self.val_dataiter_cls,
                   get_trial_cls=self.val_get_trial_cls,
                   shuffle=False,
                   combos=self.get_combos(subject=[subject],
                                          gesture=self.gestures,
                                          trial=self.trials))
        return train, val

    def get_inter_subject_val(self, fold, **kargs):
        subject = self.subjects[fold]
        load = partial(self.get_dataiter, **kargs)
        val = load(cls=self.val_dataiter_cls,
                   get_trial_cls=self.val_get_trial_cls,
                   shuffle=False,
                   combos=self.get_combos(subject=[subject],
                                          gesture=self.gestures,
                                          trial=self.trials))
        return val

    def _get_intra_subject_val(self, subjects, **kargs):
        load = partial(self.get_dataiter, **kargs)
        val = load(cls=self.val_dataiter_cls,
                   get_trial_cls=self.val_get_trial_cls,
                   shuffle=False,
                   combos=self.get_combos(subject=subjects,
                                          gesture=self.gestures,
                                          trial=self.val_trials))
        return val

    def get_dataiter(self, **kargs):
        cls = kargs.pop('cls', self.dataiter_cls)
        return cls(dataset=self, last_batch_handle='pad', **kargs)

    @property
    def train_get_trial_cls(self):
        return self.get_trial_cls

    @property
    def val_get_trial_cls(self):
        return self.get_trial_cls

    class get_trial_cls(BaseGetTrial):

        def __call__(self, combo):
            path = self.get_path(combo)
            semg = sio.loadmat(path)['data'].astype(np.float32)
            if self.preprocess:
                semg = self.preprocess(semg, **self.get_preprocess_kargs())
            gesture = np.repeat(combo.gesture, len(semg))
            subject = np.repeat(combo.subject, len(semg))
            return self.trial_cls(semg=semg, gesture=gesture, subject=subject, meta=combo.copy())

        def get_path(self, combo):
            return os.path.join(
                self.dataset.root,
                '{c.subject:03d}-{c.gesture:03d}-{c.trial:03d}.mat'.format(c=combo)
            )

    @property
    def train_dataiter_cls(self):
        return self.dataiter_cls

    @property
    def val_dataiter_cls(self):
        return self.dataiter_cls

    @property
    def dataiter_cls(dataset):
        class cls(
            di.InitMixin(data_shapes=[('semg', (-1, 1, dataset.num_semg_row, dataset.num_semg_col))],
                         label_shapes=[('gesture', (-1,)),
                                       ('subject', (-1,))],
                         indices=['gesture', 'subject']),
            di.IndexMixin,
            di.BalanceGestureMixin,
            di.NDArrayIter
        ):
            pass

        return cls


class PrevMixin(object):

    @property
    def dataiter_cls(dataset):
        class cls(
            di.InitMixin(data_shapes=[('semg', (-1, 1, dataset.num_semg_row, dataset.num_semg_col))],
                         label_shapes=[('gesture', (-1,)),
                                       ('prev', (-1, 1, dataset.num_semg_row, dataset.num_semg_col)),
                                       ('subject', (-1,))],
                         indices=['gesture', 'subject']),
            di.PrevMixin,
            di.IndexMixin,
            di.BalanceGestureMixin,
            di.NDArrayIter
        ):
            pass

        return cls

    class get_trial_cls(BaseGetTrial):

        def __call__(self, combo):
            path = self.get_path(combo)
            semg = sio.loadmat(path)['data'].astype(np.float32)
            if self.preprocess:
                semg = self.preprocess(semg, **self.get_preprocess_kargs())
            gesture = np.repeat(combo.gesture, len(semg))
            subject = np.repeat(combo.subject, len(semg))
            prev = utils.get_semg_prev(semg, self.dataset.steps)
            return self.trial_cls(semg=semg, prev=prev, gesture=gesture, subject=subject, meta=combo.copy())

        def get_path(self, combo):
            return os.path.join(
                self.dataset.root,
                '{c.subject:03d}-{c.gesture:03d}-{c.trial:03d}.mat'.format(c=combo)
            )


class PrevSemiMixin(PrevMixin):

    class train_get_trial_cls(RawGetTrialMixin, MemoGetTrialMixin, BaseGetTrial):

        def __call__(self, combo):
            if utils.hash(combo) not in self.memo:
                self.update_memo(combo)
            semg, prev, gesture, subject = self.memo[utils.hash(combo)]
            return self.trial_cls(semg=semg, prev=prev, gesture=gesture, subject=subject, meta=combo.copy())

        def update_memo(self, combo):
            path = self.get_path(combo)
            logger.debug('Load subject {} gesture {}', combo.subject, combo.gesture)
            mat = _load_semi(combo, path, self.dataset.framerate)
            semg = mat['data'].astype(np.float32)
            if self.preprocess:
                semg = self.preprocess(semg, **self.get_preprocess_kargs())
            gesture = mat['gesture'].astype(np.int32).ravel()
            subject = np.repeat(combo.subject, len(semg))
            nt.assert_equal(combo.subject, mat['subject'].flat[0])
            prev = utils.get_semg_prev(semg, self.dataset.steps)

            for trial, data in self.segment(semg, prev, gesture, subject, self._make_trial(gesture)).items():
                combo = combo.copy()
                combo.trial = trial
                self.memo[utils.hash(combo)] = data

        def segment(self, semg, prev, gesture, subject, trial):
            nt.assert_equal(len(set(trial[trial > 0])), len(self.dataset.trials))
            data = {}
            for current in self.dataset.trials:
                mask = trial == current
                nt.assert_greater_equal(mask.sum(), self.n)
                gesture_masked = gesture[mask]
                begin = (len(gesture_masked) - self.n) // 2
                end = begin + self.n
                gesture_masked[:begin] = -1
                gesture_masked[end:] = -1
                data[current] = self._shorten(begin, end, (semg[mask],
                                                           prev[mask],
                                                           gesture_masked,
                                                           subject[mask]))
            return data

        def _shorten(self, begin, end, data):
            if not hasattr(self.dataset, 'expand'):
                return data
            begin, end = utils.expand(begin, end, len(data[0]), self.dataset.expand)
            return tuple(a[begin:end] for a in data)


@utils.cached
def _load_semi(combo, path, framerate):
    mat = sio.loadmat(path)
    semg = mat['data'].astype(np.float32)
    semg = np.transpose([utils.butter_bandstop_filter(ch, 45, 55, framerate, 2)
                         for ch in semg.T])
    mat['data'] = semg
    return mat


class RawMidMixin(object):

    class get_trial_cls(RawGetTrialMixin, MemoGetTrialMixin, BaseGetTrial):

        def __call__(self, combo):
            if utils.hash(combo) not in self.memo:
                self.update_memo(combo)
            semg, gesture, subject = self.memo[utils.hash(combo)]
            return self.trial_cls(semg=semg, gesture=gesture, subject=subject, meta=combo.copy())

        def update_memo(self, combo):
            path = self.get_path(combo)
            logger.debug('Load subject {} gesture {}', combo.subject, combo.gesture)
            mat = sio.loadmat(path)
            semg = mat['data'].astype(np.float32)
            if self.preprocess:
                semg = self.preprocess(semg, **self.get_preprocess_kargs())
            gesture = mat['gesture'].astype(np.int32).ravel()
            subject = np.repeat(combo.subject, len(semg))
            nt.assert_equal(combo.subject, mat['subject'].flat[0])

            for trial, data in self.segment(semg, gesture, subject, self._make_trial(gesture)).items():
                combo = combo.copy()
                combo.trial = trial
                self.memo[utils.hash(combo)] = data

        def segment(self, semg, gesture, subject, trial):
            nt.assert_equal(len(set(trial[trial > 0])), len(self.dataset.trials))
            data = {}
            for current in self.dataset.trials:
                mask = trial == current
                nt.assert_greater_equal(mask.sum(), self.n)
                begin = (mask.sum() - self.n) // 2
                end = begin + self.n
                data[current] = (semg[mask][begin:end], gesture[mask][begin:end], subject[mask][begin:end])
            return data


class PrevSemiDataset(PrevSemiMixin, StepMixin, BaseDataset):

    def get_ada_get_trial_cls(self, subject_val):
        class cls(self.train_get_trial_cls):

            def update_memo(self, combo):
                path = self.get_path(combo)
                logger.debug('Load subject {} gesture {}', combo.subject, combo.gesture)
                semg, gesture, subject = _load_long(path, self.dataset.framerate)
                if self.preprocess:
                    semg = self.preprocess(semg, **self.get_preprocess_kargs())
                nt.assert_equal(combo.subject, subject)
                subject = np.repeat(combo.subject, len(semg))
                prev = utils.get_semg_prev(semg, self.dataset.steps)

                for trial, data in self.segment(semg, prev, gesture, subject, self._make_trial(gesture)).items():
                    combo = combo.copy()
                    combo.trial = trial

                    nt.assert_equal(len(data), 4)

                    if combo.subject == subject_val:
                        data[2][:] = -1
                    else:
                        if getattr(self.dataset, 'ada_label_trainset', False):
                            mask = data[2] >= 0
                            data = tuple(x[mask] for x in data)

                        if getattr(self.dataset, 'ada_downsample_trainset', True):
                            index = np.arange(0, len(data[0]), self.dataset.num_subject - 1)
                            data = tuple(x[index] for x in data)

                    self.memo[utils.hash(combo)] = data

        return cls

    def get_inter_subject_ada_data(self, fold, **kargs):
        subject = self.subjects[fold]
        load = partial(self.get_dataiter, **kargs)
        train = load(cls=self.train_dataiter_cls,
                     get_trial_cls=self.get_ada_get_trial_cls(subject),
                     shuffle=True,
                     combos=self.get_combos(subject=self.subjects,
                                            gesture=self.gestures,
                                            trial=self.trials))
        val = load(cls=self.val_dataiter_cls,
                   get_trial_cls=self.val_get_trial_cls,
                   shuffle=False,
                   combos=self.get_combos(subject=[subject],
                                          gesture=self.gestures,
                                          trial=self.trials))
        return train, val


@utils.cached
def _load_long(path, framerate):
    mat = sio.loadmat(path)
    semg = mat['data'].astype(np.float32)
    semg = np.transpose([utils.butter_bandstop_filter(ch, 45, 55, framerate, 2)
                         for ch in semg.T])
    gesture = mat['gesture'].astype(np.int32).ravel()
    subject = int(mat['subject'].flat[0])
    return semg, gesture, subject
