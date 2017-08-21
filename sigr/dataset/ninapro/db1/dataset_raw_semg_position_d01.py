from __future__ import division
import os
from logbook import Logger
import numpy as np
import nose.tools as nt
from collections import OrderedDict
from .base import BaseDataset, BaseGetTrial, Combo, Trial
from .get_data import get_ninapro_db1_semg_position_data
from .... import utils, CACHE
from ... import data_iter as di


logger = Logger(__name__)


class Dataset(BaseDataset):

    root = os.path.join(CACHE, 'ninapro-db1-raw')
    subjects = list(range(1, 28))
    gestures = list(range(1, 53))
    trials = list(range(1, 11))
    num_position = 81 * 2

    def get_trial_func(self, **kargs):
        return GetTrial(self.root, self.gestures, self.trials, **kargs)

    def get_dataiter(self, combos, **kargs):
        get_trial = kargs.pop('get_trial', None)
        if get_trial is None:
            get_trial = self.get_trial_func(
                preprocess=kargs.pop('preprocess'),
                norest=kargs.pop('norest')
            )

        combos = list(combos)

        semg = []
        position = []
        gesture = []
        subject = []
        segment = []

        for combo in combos:
            trial = get_trial(combo=combo)
            semg.append(trial.data[0])
            position.append(trial.data[1])
            gesture.append(trial.gesture)
            subject.append(trial.subject)
            segment.append(np.repeat(len(segment), len(semg[-1])))

        logger.debug('MAT loaded')
        assert semg and position, 'Empty data'

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
        position = np.vstack(position)
        nt.assert_equal(len(semg), len(position))
        nt.assert_equal(position.ndim, 2)
        logger.debug('Data stacked')

        gesture = utils.get_index(np.hstack(gesture), ignores=[-1])
        subject = utils.get_index(np.hstack(subject), ignores=[-1])
        segment = np.hstack(segment)

        logger.debug('Make data iter')
        return DataIter(
            data=OrderedDict([('semg', semg)]),
            label=OrderedDict([('gesture', gesture),
                               ('position', position),
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
                              zip(paths, get_ninapro_db1_semg_position_data(paths, self.preprocess))})
        semg, position, local_gesture = self.memo[path][
            (self._as_local(combo.gesture), combo.trial)]
        nt.assert_equal(position.shape[1], Dataset.num_position)
        n = len(semg)
        gesture = np.repeat(combo.gesture, n)
        gesture[local_gesture == 0] = -1
        subject = np.repeat(combo.subject, n)
        return Trial(data=(semg, position), gesture=gesture, subject=subject)

    def _segment(self, semg, position, gesture):
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

        position = np.hstack([position, np.diff(np.vstack([position[:1], position]), axis=0)])

        data = {}
        trials = {}
        for begin, end in zip(breaks[:-1], breaks[1:]):
            g = gesture[end - 1]
            assert self.norest or g > 0
            if g > 0:
                trials[g] = trials.get(g, 0) + 1
                data[(g, trials[g])] = (semg[begin:end], position[begin:end], gesture[begin:end])
        return data
