from __future__ import division
import os
from logbook import Logger
import numpy as np
import nose.tools as nt
from collections import OrderedDict
from .base import BaseDataset, BaseGetTrial, Combo, Trial
from .get_data import get_ninapro_db1_semg_pose_data
from .... import utils, CACHE
from ... import data_iter as di


logger = Logger(__name__)


class Dataset(BaseDataset):

    root = os.path.join(CACHE, 'ninapro-db1-raw')
    subjects = list(range(1, 28))
    gestures = list(range(1, 53))
    trials = list(range(1, 11))

    def __init__(self, *args, **kargs):
        self.num_pose = kargs.pop('num_pose')
        self.tag = kargs.pop('tag', '%d' % self.num_pose)
        self.step = kargs.pop('step', 1)
        self.semi_downsample = kargs.pop('semi_downsample', None)
        self.semi_downsample_keep = kargs.pop('semi_downsample_keep', False)
        self.atzori12_downsample = kargs.pop('atzori12_downsample', False)
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
        pose = []
        prev = []
        gesture = []
        subject = []
        segment = []

        for combo in combos:
            trial = get_trial(combo=combo)
            semg.append(trial.data[0])
            pose.append(trial.data[1])
            prev.append(trial.data[2])
            gesture.append(trial.gesture)
            subject.append(trial.subject)
            segment.append(np.repeat(len(segment), len(semg[-1])))

        logger.debug('MAT loaded')
        assert semg and pose and prev, 'Empty data'

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
        nt.assert_equal(len(semg), len(pose))
        nt.assert_equal(pose.ndim, 1)
        prev = np.vstack(prev).reshape(-1, 1, self.num_semg_row, self.num_semg_col)
        nt.assert_equal(len(semg), len(prev))
        logger.debug('Data stacked')

        if self.atzori12_downsample:
            logger.info('Atzori12 downsmaple')
            assert not self.semi_downsample
            for part in gesture:
                part[:int(len(part) / 3)] = -1
                part[int(len(part) / 3 * 2):] = -1

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
                    pose = upsample(pose)
                    prev = upsample(prev)
                    subject = upsample(subject)
                    segment = upsample(segment)
                    index = np.arange(len(semg))

        logger.debug('Make data iter')
        return DataIter(
            data=OrderedDict([('semg', semg)]),
            label=OrderedDict([('gesture', gesture),
                               ('pose', pose),
                               ('prev', prev),
                               ('subject', subject),
                               ('segment', segment)]),
            index=index,
            num_gesture=gesture.max() + 1,
            num_subject=subject.max() + 1,
            **kargs
        )


class DataIter(
    di.IndexMixin,
    di.BalanceGestureMixin,
    di.DownsampleMixin,
    di.NDArrayIter
):
    def __init__(self, **kargs):
        self.num_gesture = kargs.pop('num_gesture')
        self.num_subject = kargs.pop('num_subject')
        super(DataIter, self).__init__(**kargs)


class GetTrial(BaseGetTrial):

    def __call__(self, combo):
        path = self.get_path(combo)
        if path not in self.memo:
            logger.debug('Load subject {}', combo.subject)
            paths = sorted(set(self.get_path(Combo(combo.subject, gesture, trial))
                               for gesture, trial in self.gesture_and_trials))
            self.memo.update({path: self._segment(*data) for path, data in
                              zip(paths, get_ninapro_db1_semg_pose_data(paths, self.preprocess, self.dataset.tag))})
        semg, pose, prev, local_gesture = self.memo[path][
            (self._as_local(combo.gesture), combo.trial)]
        nt.assert_equal(pose.ndim, 1)
        nt.assert_equal(len(pose), len(semg))
        nt.assert_equal(prev.shape, semg.shape)
        n = len(semg)
        gesture = np.repeat(combo.gesture, n)
        gesture[local_gesture == 0] = -1
        subject = np.repeat(combo.subject, n)
        return Trial(data=(semg, pose, prev), gesture=gesture, subject=subject)

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

        prev = np.vstack([semg[:self.dataset.step], semg])
        prev = prev[:-self.dataset.step]
        nt.assert_equal(prev.shape, semg.shape)

        data = {}
        trials = {}
        for begin, end in zip(breaks[:-1], breaks[1:]):
            g = gesture[end - 1]
            assert self.norest or g > 0
            if g > 0:
                trials[g] = trials.get(g, 0) + 1
                data[(g, trials[g])] = (semg[begin:end], pose[begin:end], prev[begin:end], gesture[begin:end])
        return data
