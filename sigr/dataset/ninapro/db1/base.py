from __future__ import division
from functools import partial
from itertools import product
import nose.tools as nt
from logbook import Logger
import scipy.io as sio
import numpy as np
import os
from collections import OrderedDict
from .... import utils, CACHE
from ....data.ninapro import Dataset as _BaseDataset
from ... import base
from ... import get_trial as gt
from ... import data_iter as di


logger = Logger(__name__)
Combo = base.Combo
Trial = base.Trial
StepMixin = base.StepMixin
DownsampleMixin = base.DownsampleMixin
SemiDownsampleMixin = base.SemiDownsampleMixin


class BaseIter(di.IndexMixin, di.BalanceGestureMixin, di.NDArrayIter):
    pass


class BaseGetTrial(gt.PreprocessMixin, gt.MemoMixin, gt.BaseGetTrial):

    @property
    def memo_room(self):
        return self.dataset.num_gesture * self.dataset.num_trial

    @property
    def gesture_and_trials(self):
        return list(product(self.dataset.gestures, self.dataset.trials))

    def __init__(self, norest=True, **kargs):
        super(BaseGetTrial, self).__init__(**kargs)
        self.norest = norest

    def get_path(self, combo):
        return os.path.join(
            self.dataset.root,
            's{c.subject:d}',
            'S{c.subject:d}_A1_E{e:d}.mat').format(
                c=combo, e=self._get_execise(combo.gesture))

    def _get_execise(self, gesture):
        assert gesture >= 1 and gesture <= 52
        if gesture <= 12:
            return 1
        if gesture <= 29:
            return 2
        return 3

    def _as_local(self, gesture):
        assert gesture >= 1 and gesture <= 52
        if gesture <= 12:
            return gesture
        if gesture <= 29:
            return gesture - 12
        return gesture - 29

    def _load_raw(self, path):
        semg = sio.loadmat(path)['emg'].astype(np.float32)
        gesture = sio.loadmat(path)['restimulus'].astype(np.int32).ravel()
        frame = np.arange(len(semg))
        return OrderedDict([('semg', semg), ('gesture', gesture), ('frame', frame)])

    def update_memo(self, combo):
        path_memo = {}
        for gesture, trial in self.gesture_and_trials:
            combo = combo.copy()
            combo.gesture = gesture
            combo.trial = trial
            path = self.get_path(combo)

            if path not in path_memo:
                logger.debug('Load {}', path)
                d = self._load_raw(path)
                if self.preprocess:
                    d = type(d)(zip(
                        d.keys(),
                        self.preprocess(list(d.values()),
                                        framerate=self.dataset.framerate,
                                        num_semg_row=self.dataset.num_semg_row,
                                        num_semg_col=self.dataset.num_semg_col)))
                d['subject'] = np.repeat(combo.subject, len(d['semg']))
                path_memo[path] = self._segment(combo, **d)

            d = path_memo[path][(self._as_local(combo.gesture), combo.trial)].copy()
            n = len(d.semg)
            gesture = np.repeat(combo.gesture, n)
            assert self.norest
            gesture[d.gesture <= 0] = -1
            d.gesture = gesture
            self.set_memo(combo, d)

    def _segment(self, combo, gesture, **data):
        if self.norest:
            breaks = list(np.where(gesture[:-1] != gesture[1:])[0] + 1)
            if gesture[0] > 0:
                breaks.append(0)
            if gesture[-1] > 0:
                breaks.append(len(gesture))
        else:
            breaks = [0] + list(np.where((gesture[:-1] > 0) & (gesture[1:] == 0))[0] + 1)
            if gesture[-1] > 0:
                breaks.append(len(gesture))

        trials = {}
        memo = {}
        for begin, end in zip(breaks[:-1], breaks[1:]):
            g = gesture[end - 1]
            assert self.norest or g > 0
            if g > 0:
                trials[g] = trials.get(g, 0) + 1
                d = dict(gesture=gesture[begin:end].copy())
                for key in data:
                    d[key] = data[key][begin:end].copy()
                memo[(g, trials[g])] = self.trial_cls(**d)
        return memo


class BaseDataset(
    base.DeterministicMixin,
    base.GetTrialMixin,
    base.DataIterMixin,
    _BaseDataset
):

    root = os.path.join(CACHE, 'ninapro-db1-raw')
    subjects = list(range(1, 28))
    gestures = list(range(1, 53))
    trials = list(range(1, 11))
    circular = 'col'

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

    def get_one_fold_intra_subject_trials(self):
        return [1, 3, 4, 6, 8, 9, 10], [2, 5, 7]

    @utils.return_list
    def get_combos(self, subjects, gestures, trials):
        for subject, gesture, trial in product(subjects, gestures, trials):
            yield Combo(subject=subject, gesture=gesture, trial=trial)

    def get_universal_intra_subject_data(self, fold, **kargs):
        return self.get_universal_one_fold_intra_subject_data(fold, **kargs)

    def get_universal_one_fold_intra_subject_data(self, fold, **kargs):
        nt.assert_equal(fold, 0)
        load = partial(self.get_dataiter, **kargs)
        train_trials, val_trials = self.get_one_fold_intra_subject_trials()
        train = load(cls=self.train_dataiter_cls,
                     get_trial_cls=self.train_get_trial_cls,
                     shuffle=True,
                     combos=self.get_combos(subjects=self.subjects,
                                            gestures=self.gestures,
                                            trials=[i for i in train_trials]))
        val = load(cls=self.val_dataiter_cls,
                   get_trial_cls=self.val_get_trial_cls,
                   shuffle=False,
                   combos=self.get_combos(subjects=self.subjects,
                                          gestures=self.gestures,
                                          trials=[i for i in val_trials]))
        return train, val

    def get_intra_subject_val(self, *args, **kargs):
        return self.get_one_fold_intra_subject_val(*args, **kargs)

    def get_intra_subject_data(self, *args, **kargs):
        return self.get_one_fold_intra_subject_data(*args, **kargs)

    def get_one_fold_intra_subject_data(self, fold, **kargs):
        load = partial(self.get_dataiter, **kargs)
        subject = self.subjects[fold]
        train_trials, val_trials = self.get_one_fold_intra_subject_trials()
        train = load(cls=self.train_dataiter_cls,
                     get_trial_cls=self.train_get_trial_cls,
                     shuffle=True,
                     combos=self.get_combos(subjects=[subject],
                                            gestures=self.gestures,
                                            trials=[i for i in train_trials]))
        val = load(cls=self.val_dataiter_cls,
                   get_trial_cls=self.val_get_trial_cls,
                   shuffle=False,
                   combos=self.get_combos(subjects=[subject],
                                          gestures=self.gestures,
                                          trials=[i for i in val_trials]))
        return train, val

    def get_one_fold_intra_subject_val(self, fold, **kargs):
        load = partial(self.get_dataiter, **kargs)
        subject = self.subjects[fold]
        _, val_trials = self.get_one_fold_intra_subject_trials()
        val = load(cls=self.val_dataiter_cls,
                   get_trial_cls=self.val_get_trial_cls,
                   shuffle=False,
                   combos=self.get_combos(subjects=[subject],
                                          gestures=self.gestures,
                                          trials=[i for i in val_trials]))
        return val

    def get_inter_subject_data(self, fold, **kargs):
        load = partial(self.get_dataiter, **kargs)
        subject = self.subjects[fold]
        train = load(cls=self.train_dataiter_cls,
                     get_trial_cls=self.train_get_trial_cls,
                     shuffle=True,
                     combos=self.get_combos(subjects=utils.exclude(self.subjects, [subject]),
                                            gestures=self.gestures,
                                            trials=self.trials))
        val = load(cls=self.val_dataiter_cls,
                   get_trial_cls=self.val_get_trial_cls,
                   shuffle=False,
                   combos=self.get_combos(subjects=[subject],
                                          gestures=self.gestures,
                                          trials=self.trials))
        return train, val

    def get_inter_subject_val(self, fold, **kargs):
        load = partial(self.get_dataiter, **kargs)
        subject = self.subjects[fold]
        val = load(cls=self.val_dataiter_cls,
                   get_trial_cls=self.val_get_trial_cls,
                   shuffle=False,
                   combos=self.get_combos(subjects=[subject],
                                          gestures=self.gestures,
                                          trials=self.trials))
        return val

    get_trial_cls = BaseGetTrial

    @property
    def dataiter_cls(dataset):
        class cls(
            di.InitMixin(data_shapes=[('semg', (-1, 1, dataset.num_semg_row, dataset.num_semg_col))],
                         label_shapes=[('gesture', (-1,)),
                                       ('subject', (-1,))],
                         indices=['gesture', 'subject']),
            BaseIter
        ):
            pass

        return cls


class SDataMixin(object):

    @property
    def get_trial_cls(dataset):
        class cls(super(SDataMixin, dataset).get_trial_cls):

            def _load_raw(self, path):
                from ....data import Preprocess
                lowpass = Preprocess.parse('ninapro-lowpass')
                d = super(cls, self)._load_raw(path).copy()
                d['semg'] = lowpass(d['semg'], **self.dataset.get_preprocess_kargs())
                return d

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
            BaseIter
        ):
            pass

        return cls

    @property
    def get_trial_cls(dataset):
        class cls(super(PrevMixin, dataset).get_trial_cls):

            def _load_raw(self, path):
                d = super(cls, self)._load_raw(path).copy()
                d['prev'] = utils.get_semg_prev(d['semg'], self.dataset.steps)
                return d

        return cls


class PrevWindowMixin(PrevMixin):

    @property
    def dataiter_cls(dataset):
        class cls(
            di.InitMixin(data_shapes=[('semg', (-1, 1, dataset.num_semg_row, dataset.num_semg_col))],
                         label_shapes=[('gesture', (-1,)),
                                       ('prev', (-1, 1, dataset.num_semg_row, dataset.num_semg_col)),
                                       ('subject', (-1,))],
                         indices=['gesture', 'subject']),
            di.PrevMixin,
            di.WindowMixin,
            BaseIter
        ):

            @property
            def provide_data(self):
                shapes = [(k, (v[0] // self.window,) + v[1:]) for k, v
                          in super(cls, self).provide_data]
                return self._promote_shape(shapes)

            @property
            def provide_label(self):
                shapes = [(k, (v[0] // self.window,) + v[1:]) for k, v
                          in super(cls, self).provide_label]
                return self._promote_shape(shapes)

            @utils.return_list
            def _promote_shape(self, shapes):
                for name, shape in shapes:
                    if name in ('semg', 'prev'):
                        nt.assert_equal(shape[2:], (dataset.num_semg_row,
                                                    dataset.num_semg_col))
                        yield name, (shape[0],
                                     dataset.window,
                                     dataset.num_semg_row,
                                     dataset.num_semg_col)
                    else:
                        yield name, shape

            def getdata(self):
                return self._promote_data(self.provide_data, super(cls, self).getdata())

            def getlabel(self):
                return self._promote_data(self.provide_label, super(cls, self).getlabel())

            @utils.return_list
            def _promote_data(self, shapes, data):
                for (name, shape), item in zip(shapes, data):
                    if name in ('semg', 'prev'):
                        yield item.reshape(shape)
                    else:
                        nt.assert_equal(len(shape), 1)
                        yield item.reshape((shape[0], dataset.window)).T[0]

        return cls


class WindowMixin(object):

    @property
    def dataiter_cls(dataset):
        class cls(
            di.InitMixin(data_shapes=[('semg', (-1, 1, dataset.num_semg_row, dataset.num_semg_col))],
                         label_shapes=[('gesture', (-1,)),
                                       ('subject', (-1,))],
                         indices=['gesture', 'subject']),
            di.WindowMixin,
            BaseIter
        ):

            @property
            def provide_data(self):
                shapes = [(k, (v[0] // self.window,) + v[1:]) for k, v
                          in super(cls, self).provide_data]
                return self._promote_shape(shapes)

            @property
            def provide_label(self):
                shapes = [(k, (v[0] // self.window,) + v[1:]) for k, v
                          in super(cls, self).provide_label]
                return self._promote_shape(shapes)

            @utils.return_list
            def _promote_shape(self, shapes):
                for name, shape in shapes:
                    if name in ('semg',):
                        nt.assert_equal(shape[2:], (dataset.num_semg_row,
                                                    dataset.num_semg_col))
                        yield name, (shape[0],
                                     dataset.window,
                                     dataset.num_semg_row,
                                     dataset.num_semg_col)
                    else:
                        yield name, shape

            def getdata(self):
                return self._promote_data(self.provide_data, super(cls, self).getdata())

            def getlabel(self):
                return self._promote_data(self.provide_label, super(cls, self).getlabel())

            @utils.return_list
            def _promote_data(self, shapes, data):
                for (name, shape), item in zip(shapes, data):
                    if name in ('semg',):
                        yield item.reshape(shape)
                    else:
                        nt.assert_equal(len(shape), 1)
                        yield item.reshape((shape[0], dataset.window)).T[0]

        return cls


class PrevWindowRowMixin(PrevMixin):

    @property
    def dataiter_cls(dataset):
        class cls(
            di.InitMixin(data_shapes=[('semg', (-1, 1, dataset.num_semg_row, dataset.num_semg_col))],
                         label_shapes=[('gesture', (-1,)),
                                       ('prev', (-1, 1, dataset.num_semg_row, dataset.num_semg_col)),
                                       ('subject', (-1,))],
                         indices=['gesture', 'subject']),
            di.PrevMixin,
            di.WindowMixin,
            BaseIter
        ):

            @property
            def provide_data(self):
                shapes = [(k, (v[0] // self.window,) + v[1:]) for k, v
                          in super(cls, self).provide_data]
                return self._promote_shape(shapes)

            @property
            def provide_label(self):
                shapes = [(k, (v[0] // self.window,) + v[1:]) for k, v
                          in super(cls, self).provide_label]
                return self._promote_shape(shapes)

            @utils.return_list
            def _promote_shape(self, shapes):
                for name, shape in shapes:
                    if name in ('semg', 'prev'):
                        nt.assert_equal(shape[2:], (dataset.num_semg_row,
                                                    dataset.num_semg_col))
                        yield name, (shape[0],
                                     1,
                                     dataset.window * dataset.num_semg_row,
                                     dataset.num_semg_col)
                    else:
                        yield name, shape

            def getdata(self):
                return self._promote_data(self.provide_data, super(cls, self).getdata())

            def getlabel(self):
                return self._promote_data(self.provide_label, super(cls, self).getlabel())

            @utils.return_list
            def _promote_data(self, shapes, data):
                for (name, shape), item in zip(shapes, data):
                    if name in ('semg', 'prev'):
                        yield item.reshape(shape)
                    else:
                        nt.assert_equal(len(shape), 1)
                        yield item.reshape((shape[0], dataset.window)).T[0]

        return cls


class SDataDataset(SDataMixin, BaseDataset):
    pass


def AdaMixin(*ignores):
    class ada(object):

        def train_get_trial_cls(dataset, subject_val):
            class cls(super(ada, dataset).train_get_trial_cls):

                def trial_cls(self, **kargs):
                    trial = super(cls, self).trial_cls(**kargs)
                    if trial['subject'][0] == subject_val:
                        for ignore in ignores:
                            a = trial[ignore].copy()
                            a[:] = -1
                            trial[ignore] = a
                    return trial

            return cls

        def get_inter_subject_data(self, fold, **kargs):
            subject_val = self.subjects[fold]
            load = partial(self.get_dataiter, **kargs)
            train = load(cls=self.train_dataiter_cls,
                         get_trial_cls=self.train_get_trial_cls(subject_val),
                         shuffle=True,
                         combos=self.get_combos(subjects=self.subjects,
                                                gestures=self.gestures,
                                                trials=self.trials))
            val = load(cls=self.val_dataiter_cls,
                       get_trial_cls=self.val_get_trial_cls,
                       shuffle=False,
                       combos=self.get_combos(subjects=[subject_val],
                                              gestures=self.gestures,
                                              trials=self.trials))
            return train, val

    return ada


class PosePrevMixin(object):

    def __init__(self, *args, **kargs):
        self.num_pose = kargs.pop('num_pose')
        self.tag = kargs.pop('tag', '%d' % self.num_pose)
        super(PosePrevMixin, self).__init__(*args, **kargs)

    @property
    def dataiter_cls(dataset):
        class cls(
            di.InitMixin(data_shapes=[('semg', (-1, 1, dataset.num_semg_row, dataset.num_semg_col))],
                         label_shapes=[('gesture', (-1,)),
                                       ('pose', (-1,)),
                                       ('prev', (-1, 1, dataset.num_semg_row, dataset.num_semg_col)),
                                       ('subject', (-1,))],
                         indices=['gesture', 'subject', 'pose']),
            di.PrevMixin,
            BaseIter
        ):
            pass

        return cls

    @property
    def get_trial_cls(dataset):
        class cls(super(PosePrevMixin, dataset).get_trial_cls):

            def _load_raw(self, path):
                d = super(cls, self)._load_raw(path).copy()

                pose = sio.loadmat(self._get_pose_path(path))['pose'].astype(np.int32).ravel()
                nt.assert_equal(len(d['semg']), len(pose))
                nt.assert_equal(pose.ndim, 1)
                d['pose'] = pose

                d['prev'] = utils.get_semg_prev(d['semg'], self.dataset.steps)
                return d

            def _get_pose_path(self, path):
                parts = path.split(os.sep)
                return os.sep.join(parts[:-2] + ['pose-' + self.dataset.tag] + parts[-2:])

        return cls


class CaputoMixin(DownsampleMixin):

    downsample = 16

    def get_one_fold_intra_subject_trials(self):
        return [1, 3, 4, 5, 9], [2, 6, 7, 8, 10]


class PoseMixin(object):

    def __init__(self, *args, **kargs):
        self.num_pose = kargs.pop('num_pose')
        self.tag = kargs.pop('tag', '%d' % self.num_pose)
        super(PoseMixin, self).__init__(*args, **kargs)

    @property
    def dataiter_cls(dataset):
        class cls(
            di.InitMixin(data_shapes=[('semg', (-1, 1, dataset.num_semg_row, dataset.num_semg_col))],
                         label_shapes=[('gesture', (-1,)),
                                       ('pose', (-1,)),
                                       ('subject', (-1,))],
                         indices=['gesture', 'subject', 'pose']),
            BaseIter
        ):
            pass

        return cls

    @property
    def get_trial_cls(dataset):
        class cls(super(PoseMixin, dataset).get_trial_cls):

            def _load_raw(self, path):
                d = super(cls, self)._load_raw(path).copy()
                pose = sio.loadmat(self._get_pose_path(path))['pose'].astype(np.int32).ravel()
                nt.assert_equal(len(d['semg']), len(pose))
                nt.assert_equal(pose.ndim, 1)
                d['pose'] = pose
                return d

            def _get_pose_path(self, path):
                parts = path.split(os.sep)
                return os.sep.join(parts[:-2] + ['pose-' + self.dataset.tag] + parts[-2:])

        return cls


__all__ = ['BaseDataset', 'BaseGetTrial', 'Combo', 'Trial', 'StepMixin']
