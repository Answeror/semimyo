from __future__ import division
import os
from logbook import Logger
import numpy as np
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

    def __init__(self, *args, **kargs):
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
        gesture = []
        subject = []
        segment = []

        for combo in combos:
            trial = get_trial(combo=combo)
            semg.append(trial.data)
            gesture.append(trial.gesture)
            subject.append(trial.subject)
            segment.append(np.repeat(len(segment), len(semg[-1])))

        logger.debug('MAT loaded')
        assert semg, 'Empty data'

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
        logger.debug('Data stacked')

        gesture = utils.get_index(np.hstack(gesture), ignores=[-1])
        subject = utils.get_index(np.hstack(subject), ignores=[-1])
        segment = np.hstack(segment)

        logger.debug('Make data iter')
        return DataIter(
            data=OrderedDict([('semg', semg)]),
            label=OrderedDict([('gesture', gesture),
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
                              zip(paths, get_ninapro_db1_semg_data(paths, self.preprocess))})
        semg, local_gesture = self.memo[path][
            (self._as_local(combo.gesture), combo.trial)]
        n = len(semg)
        gesture = np.repeat(combo.gesture, n)
        gesture[local_gesture == 0] = -1
        subject = np.repeat(combo.subject, n)
        return Trial(data=semg, gesture=gesture, subject=subject)

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

        data = {}
        trials = {}
        for begin, end in zip(breaks[:-1], breaks[1:]):
            g = gesture[end - 1]
            assert self.norest or g > 0
            if g > 0:
                trials[g] = trials.get(g, 0) + 1
                n = end - begin
                begin += n // 3
                end -= n // 3
                data[(g, trials[g])] = (semg[begin:end], gesture[begin:end])
        return data
