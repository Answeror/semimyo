from __future__ import division
import os
from nose.tools import assert_equal
from functools import partial
from logbook import Logger
import mxnet as mx
import numpy as np
from itertools import product
from collections import OrderedDict
from .base import BaseDataset, Combo, Trial
from .get_data import get_ninapro_db1_semg_pose_data
from ...amplitude_weighting import get_amplitude_weight
from .... import utils, constant, CACHE


logger = Logger(__name__)


class Dataset(BaseDataset):

    root = os.path.join(CACHE, 'ninapro-db1-raw')
    subjects = list(range(1, 28))
    gestures = list(range(1, 53))
    trials = list(range(1, 11))

    def __init__(self, num_pose, tag=None):
        self.num_pose = num_pose
        self.tag = tag if tag else '%d' % num_pose

    @classmethod
    def get_preprocess_kargs(cls):
        return dict(
            framerate=cls.framerate,
            num_semg_row=cls.num_semg_row,
            num_semg_col=cls.num_semg_col
        )

    def get_trial_func(self, **kargs):
        return GetTrial(self.root, self.gestures, self.trials, self.tag, **kargs)

    def get_dataiter(self, combos, **kargs):
        get_trial = kargs.pop('get_trial', None)
        if get_trial is None:
            get_trial = self.get_trial_func(
                preprocess=kargs.pop('preprocess'),
                norest=kargs.pop('norest')
            )

        combos = list(combos)

        semg = []
        pose = []
        gesture = []
        subject = []
        segment = []

        for combo in combos:
            trial = get_trial(combo=combo)
            semg.append(trial.data[0])
            pose.append(trial.data[1])
            gesture.append(trial.gesture)
            subject.append(trial.subject)
            segment.append(np.repeat(len(segment), len(semg[-1])))

        logger.debug('MAT loaded')
        assert semg and pose, 'Empty data'

        index = []
        n = 0
        for seg in semg:
            index.append(np.arange(n, n + len(seg)))
            n += len(seg)
        index = np.hstack(index)
        logger.debug('Index made')

        logger.debug('Segments: {}', len(semg))
        logger.debug('First segment shape: {}', semg[0].shape)

        semg = np.vstack(semg).reshape(-1, 1, self.num_semg_row, self.num_semg_col)
        pose = np.hstack(pose)
        assert_equal(len(semg), len(pose))
        assert_equal(pose.ndim, 1)
        logger.debug('Data stacked')

        gesture = utils.get_index(np.hstack(gesture), ignores=[-1])
        subject = utils.get_index(np.hstack(subject), ignores=[-1])
        segment = np.hstack(segment)

        logger.debug('Make data iter')
        return DataIter(
            data=OrderedDict([('semg', semg)]),
            label=OrderedDict([('gesture', gesture), ('pose', pose)]),
            gesture=gesture.copy(),
            subject=subject.copy(),
            segment=segment.copy(),
            index=index,
            num_gesture=gesture.max() + 1,
            num_subject=subject.max() + 1,
            framerate=self.framerate,
            **kargs
        )

    def get_one_fold_intra_subject_trials(self):
        return [1, 3, 4, 6, 8, 9, 10], [2, 5, 7]

    def get_universal_one_fold_intra_subject_data(
        self,
        fold,
        batch_size,
        preprocess,
        num_mini_batch,
        **kargs
    ):
        assert_equal(fold, 0)
        get_trial = self.get_trial_func(preprocess=preprocess, norest=True)
        load = partial(self.get_dataiter,
                       get_trial=get_trial,
                       last_batch_handle='pad',
                       batch_size=batch_size)
        train_trials, val_trials = self.get_one_fold_intra_subject_trials()
        train = load(
            combos=self.get_combos(self.subjects, self.gestures, [i for i in train_trials]),
            num_mini_batch=num_mini_batch,
            shuffle=True,
            **kargs
        )
        val = load(
            combos=self.get_combos(self.subjects, self.gestures, [i for i in val_trials]),
            shuffle=False,
            **kargs
        )
        return train, val

    def get_one_fold_intra_subject_data(
        self,
        fold,
        batch_size,
        preprocess,
        num_mini_batch,
        **kargs
    ):
        assert_equal(num_mini_batch, 1)
        get_trial = self.get_trial_func(preprocess=preprocess, norest=True)
        load = partial(self.get_dataiter,
                       get_trial=get_trial,
                       last_batch_handle='pad',
                       batch_size=batch_size)
        subject = self.subjects[fold]
        train_trials, val_trials = self.get_one_fold_intra_subject_trials()
        train = load(
            combos=self.get_combos([subject], self.gestures, [i for i in train_trials]),
            shuffle=True,
            **kargs
        )
        val = load(
            combos=self.get_combos([subject], self.gestures, [i for i in val_trials]),
            shuffle=False,
            **kargs
        )
        return train, val

    def get_one_fold_intra_subject_val(
        self,
        fold,
        batch_size,
        preprocess,
        **kargs
    ):
        get_trial = self.get_trial_func(preprocess=preprocess, norest=True)
        load = partial(self.get_dataiter,
                       get_trial=get_trial,
                       last_batch_handle='pad',
                       batch_size=batch_size)
        subject = self.subjects[fold]
        train_trials, val_trials = self.get_one_fold_intra_subject_trials()
        val = load(
            combos=self.get_combos([subject], self.gestures, [i for i in val_trials]),
            shuffle=False,
            **kargs
        )
        return val

    def get_inter_subject_data(
        self,
        fold,
        batch_size,
        preprocess,
        num_mini_batch,
        **kargs
    ):
        load = partial(self.get_dataiter,
                       preprocess=preprocess,
                       norest=True,
                       last_batch_handle='pad',
                       batch_size=batch_size)
        subject = self.subjects[fold]
        train = load(
            combos=self.get_combos([i for i in self.subjects if i != subject],
                                   self.gestures,
                                   self.trials),
            num_mini_batch=num_mini_batch,
            shuffle=True,
            **kargs)
        val = load(
            combos=self.get_combos([subject], self.gestures, self.trials),
            shuffle=False,
            **kargs
        )
        return train, val


class DataIter(mx.io.NDArrayIter):

    def __init__(self, **kargs):
        self.framerate = kargs.pop('framerate')
        self.amplitude_weighting = kargs.pop('amplitude_weighting', False)
        self.amplitude_weighting_sort = kargs.pop('amplitude_weighting_sort', False)
        self.downsample = kargs.pop('downsample', None)
        self.shuffle = kargs.pop('shuffle')
        self._gesture = kargs.pop('gesture')
        self._subject = kargs.pop('subject')
        self._segment = kargs.pop('segment')
        self._index_orig = kargs.pop('index')
        self._index = np.copy(self._index_orig)
        self.num_gesture = kargs.pop('num_gesture')
        self.num_subject = kargs.pop('num_subject')
        self.num_mini_batch = kargs.pop('num_mini_batch', constant.NUM_MINI_BATCH)
        self.random_state = kargs.pop('random_state', np.random)
        self.balance_gesture = kargs.pop('balance_gesture', 0)

        super(DataIter, self).__init__(**kargs)

        self.data = [(k, self._asnumpy(v)) for k, v in self.data]
        self.label = [(k, self._asnumpy(v)) for k, v in self.label]
        self.num_data = len(self._index)
        self.reset()

    @property
    def num_sample(self):
        return self.num_data

    @property
    def gesture(self):
        return self._gesture[self._index]

    @property
    def subject(self):
        return self._subject[self._index]

    @property
    def segment(self):
        return self._segment[self._index]

    def _asmxnd(self, a):
        return a if isinstance(a, mx.nd.NDArray) else mx.nd.array(a)

    def _asnumpy(self, a):
        return a if not isinstance(a, mx.nd.NDArray) else a.asnumpy()

    def _getdata(self, data_source):
        assert self.cursor < self.num_data, "DataIter needs reset."

        if self.cursor + self.batch_size <= self.num_data:
            res = [(x[1][self._index[self.cursor:self.cursor+self.batch_size]]) for x in data_source]
        else:
            pad = self.batch_size - self.num_data + self.cursor
            res = [(np.concatenate((x[1][self._index[self.cursor:]],
                                    x[1][self._index[:pad]]), axis=0)) for x in data_source]

        return [self._asmxnd(a) for a in res]

    def reset(self):
        self._reset()
        super(DataIter, self).reset()

    def _shuffle(self):
        if self.num_mini_batch <= 1 or len(set(self._subject)) == 1:
            self.random_state.shuffle(self._index)
        else:
            assert self.batch_size % self.num_mini_batch == 0
            mini_batch_size = self.batch_size // self.num_mini_batch

            logger.info('AdaBN shuffle with a mini batch size of {}', mini_batch_size)
            self.random_state.shuffle(self._index)
            subject_shuffled = self._subject[self._index]
            index_batch = []
            for i in sorted(set(self._subject)):
                index = self._index[subject_shuffled == i]
                index = index[:len(index) // mini_batch_size * mini_batch_size]
                index_batch.append(index.reshape(-1, mini_batch_size))
            index_batch = np.vstack(index_batch)
            index = np.arange(len(index_batch))
            self.random_state.shuffle(index)
            self._index = index_batch[index, :].ravel()

            for i in range(0, len(self._subject), mini_batch_size):
                # Make sure that the samples in one batch are from the same subject
                assert np.all(self._subject[self._index[i:i + mini_batch_size - 1]] ==
                              self._subject[self._index[i + 1:i + mini_batch_size]])

            if mini_batch_size != self.batch_size:
                assert self.batch_size % mini_batch_size == 0
                self._index = self._index[:len(self._index) // self.batch_size * self.batch_size].reshape(
                    -1, self.num_mini_batch, mini_batch_size).transpose(0, 2, 1).ravel()

    @property
    def _data_emg(self):
        return self.data[0][1]

    def _amplitude_weighting_by_sampling(self):
        assert np.all(self._index[:-1] < self._index[1:])
        if not hasattr(self, 'amplitude_weight'):
            self.amplitude_weight = get_amplitude_weight(
                self._data_emg, self._segment, self.framerate)
        if self.shuffle:
            random_state = self.random_state
        else:
            random_state = np.random.RandomState(677)
        self._index = random_state.choice(
            self._index, len(self._index), p=self.amplitude_weight)
        if self.amplitude_weighting_sort:
            logger.debug('Amplitude weighting sort')
            self._index.sort()

    def _downsample(self):
        if callable(self.downsample):
            self.downsample(self)
        else:
            samples = np.arange(len(self._index))
            np.random.RandomState(667).shuffle(samples)
            assert self.downsample > 0 and self.downsample <= 1
            samples = samples[:int(np.round(len(samples) * self.downsample))]
            assert len(samples) > 0
            self._index = self._index[samples]

    def _balance_gesture(self):
        num_sample_per_gesture = int(np.round(self.balance_gesture *
                                              len(self._index) / self.num_gesture))
        choice = []
        for gesture in set(self.gesture):
            mask = self._gesture[self._index] == gesture
            choice.append(self.random_state.choice(np.where(mask)[0],
                                                   num_sample_per_gesture))
        choice = np.hstack(choice)
        self._index = self._index[choice]

    def _reset(self):
        self._index = np.copy(self._index_orig)

        if self.amplitude_weighting:
            self._amplitude_weighting_by_sampling()

        if self.downsample:
            self._downsample()

        if self.balance_gesture:
            self._balance_gesture()

        if self.shuffle:
            self._shuffle()

        self.num_data = len(self._index)


class GetTrial(object):

    def __init__(self, root, gestures, trials, tag, preprocess=None, norest=False):
        self.root = root
        self.preprocess = preprocess
        self.memo = utils.make_lru_dict(3)
        self.gesture_and_trials = list(product(gestures, trials))
        self.norest = norest
        self.tag = tag

    def get_path(self, combo):
        return os.path.join(
            self.root,
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

    def __call__(self, combo):
        path = self.get_path(combo)
        if path not in self.memo:
            logger.debug('Load subject {}', combo.subject)
            paths = sorted(set(self.get_path(Combo(subject=combo.subject, gesture=gesture, trial=trial))
                               for gesture, trial in self.gesture_and_trials))
            self.memo.update({path: self._segment(*data) for path, data in
                              zip(paths, get_ninapro_db1_semg_pose_data(
                                  paths, self.preprocess, self.tag))})
        data = self.memo[path]
        semg, pose, local_gesture = data[
            (self._as_local(combo.gesture), combo.trial)]
        n = len(semg)
        gesture = np.repeat(combo.gesture, n)
        gesture[local_gesture == 0] = -1
        subject = np.repeat(combo.subject, n)
        return Trial(data=(semg, pose), gesture=gesture, subject=subject)

    def _segment(self, semg, pose, gesture):
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

        data = {}
        trials = {}
        for begin, end in zip(breaks[:-1], breaks[1:]):
            g = gesture[end - 1]
            assert self.norest or g > 0
            if g > 0:
                trials[g] = trials.get(g, 0) + 1
                data[(g, trials[g])] = (semg[begin:end], pose[begin:end], gesture[begin:end])

        return data
