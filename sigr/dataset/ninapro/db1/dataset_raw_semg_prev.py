from __future__ import division
import os
from logbook import Logger
import numpy as np
import nose.tools as nt
from collections import OrderedDict
from .base import BaseDataset, BaseGetTrial, Combo, Trial
from .get_data import get_ninapro_db1_semg_data
from .... import utils, CACHE
from ... import data_iter as di


logger = Logger(__name__)


class Dataset(BaseDataset):

    root = os.path.join(CACHE, 'ninapro-db1-raw')
    subjects = list(range(1, 28))
    gestures = list(range(1, 53))
    trials = list(range(1, 11))

    @property
    def num_step(self):
        return len(self.steps)

    def __init__(self, *args, **kargs):
        if 'step' in kargs:
            self.steps = [kargs.pop('step')]
        else:
            self.steps = kargs.pop('steps', [1])
        self.semi_downsample = kargs.pop('semi_downsample', None)
        self.semi_downsample_keep = kargs.pop('semi_downsample_keep', False)
        super(Dataset, self).__init__(*args, **kargs)

    def get_trial_func(self, **kargs):
        return GetTrial(self.root, self.gestures, self.trials, dataset=self, **kargs)

    def get_dataiter(self, combos, **kargs):
        get_trial = kargs.pop('get_trial', None)
        if get_trial is None:
            get_trial = self.get_trial_func(
                preprocess=kargs.pop('preprocess'),
                norest=kargs.pop('norest')
            )

        combos = list(combos)

        semg = []
        prev = []
        gesture = []
        subject = []
        segment = []

        for combo in combos:
            trial = get_trial(combo=combo)
            semg.append(trial.data[0])
            prev.append(trial.data[1])
            gesture.append(trial.gesture)
            subject.append(trial.subject)
            segment.append(np.repeat(len(segment), len(semg[-1])))

        logger.debug('MAT loaded')
        assert semg and prev, 'Empty data'

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
        prev = np.vstack(prev).reshape(-1, len(self.steps), self.num_semg_row, self.num_semg_col)
        nt.assert_equal(len(semg), len(prev))
        logger.debug('Data stacked')

        gesture = utils.get_index(np.hstack(gesture), ignores=[-1])
        subject = utils.get_index(np.hstack(subject), ignores=[-1])
        segment = np.hstack(segment)

        if self.semi_downsample:
            logger.info('Semi-downsample')
            if self.semi_downsample < 1:
                samples = np.arange(len(gesture))
                np.random.RandomState(184).shuffle(samples)
                assert self.semi_downsample > 0 and self.semi_downsample <= 1
                samples = samples[int(np.round(len(samples) * self.semi_downsample)):]
                assert len(samples) > 0
                gesture[samples] = -1
            else:
                step = int(self.semi_downsample)
                nt.assert_equal(step, self.semi_downsample)
                from ...downsample import downsample_even_segment as des
                mask = np.ones(len(gesture), dtype=np.bool)
                mask[des(step, segment)] = False
                gesture[mask] = -1
                logger.info('Selected {}%'.format(100 * (1 - np.sum(mask) / len(mask))))
                if self.semi_downsample_keep:
                    logger.info('Semi-upsample from {} to {}', len(gesture), len(gesture) + (step - 1) * np.sum(~mask))
                    upsample = lambda data: np.concatenate([data] + [data[~mask]] * (step - 1), axis=0)
                    semg = upsample(semg)
                    gesture = upsample(gesture)
                    prev = upsample(prev)
                    subject = upsample(subject)
                    segment = upsample(segment)
                    index = np.arange(len(semg))

        logger.debug('Make data iter')
        return DataIter(
            data=OrderedDict([('semg', semg)]),
            label=OrderedDict([('gesture', gesture),
                               ('prev', prev),
                               ('subject', subject),
                               ('segment', segment)]),
            index=index,
            num_gesture=gesture.max() + 1,
            num_subject=subject.max() + 1,
            dataset=self,
            **kargs
        )


class DataIter(
    di.IndexMixin,
    di.BalanceGestureMixin,
    di.NDArrayIter
):
    def __init__(self, **kargs):
        self.num_gesture = kargs.pop('num_gesture')
        self.num_subject = kargs.pop('num_subject')
        super(DataIter, self).__init__(**kargs)

    @property
    def provide_label(self):
        shapes = super(DataIter, self).provide_label
        if self.dataset.num_step > 1:
            shapes[1] = (shapes[1][0], self.provide_data[0][1])
        return shapes

    def reset(self):
        if hasattr(self, 'prev_orig'):
            self.label[1] = (self.label[1][0], self.prev_orig)
        else:
            self.prev_orig = self.label[1][1].copy()

        super(DataIter, self).reset()

        if self.dataset.num_step > 1:
            prev = self.label[1][1]
            select = self.random_state.randint(prev.shape[1], size=prev.shape[0])
            mask = np.zeros(prev.shape, dtype=np.bool)
            for i in range(mask.shape[0]):
                mask[i, select[i], :, :] = True
            prev = prev[mask].reshape(self.data[0][1].shape)
            self.label[1] = (self.label[1][0], prev)


class GetTrial(BaseGetTrial):

    def __call__(self, combo):
        path = self.get_path(combo)
        if path not in self.memo:
            logger.debug('Load subject {}', combo.subject)
            paths = sorted(set(self.get_path(Combo(combo.subject, gesture, trial))
                               for gesture, trial in self.gesture_and_trials))
            self.memo.update({path: self._segment(*data) for path, data in
                              zip(paths, get_ninapro_db1_semg_data(paths, self.preprocess))})
        semg, prev, local_gesture = self.memo[path][
            (self._as_local(combo.gesture), combo.trial)]
        #  nt.assert_equal(prev.shape, semg.shape)
        n = len(semg)
        gesture = np.repeat(combo.gesture, n)
        gesture[local_gesture == 0] = -1
        subject = np.repeat(combo.subject, n)
        return Trial(data=(semg, prev), gesture=gesture, subject=subject)

    def _segment(self, semg, gesture):
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

        prevs = []
        for step in self.dataset.steps:
            prev = np.vstack([semg[:step], semg])
            prev = prev[:-step]
            nt.assert_equal(prev.shape, semg.shape)
            prev = np.expand_dims(prev, axis=1)
            prevs.append(prev)
        prev = np.concatenate(prevs, axis=1)

        data = {}
        trials = {}
        for begin, end in zip(breaks[:-1], breaks[1:]):
            g = gesture[end - 1]
            assert self.norest or g > 0
            if g > 0:
                trials[g] = trials.get(g, 0) + 1
                data[(g, trials[g])] = (semg[begin:end], prev[begin:end], gesture[begin:end])
        return data
