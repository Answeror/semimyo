from __future__ import division
import numpy as np
from logbook import Logger
import nose.tools as nt
from collections import OrderedDict
import six
from .. import utils
from . import data_iter as di


logger = Logger(__name__)
Combo = utils.Bunch


class Trial(utils.Bunch):

    def __init__(self, *args, **kargs):
        if args:
            nt.assert_equal(len(args), 1)
            assert not kargs
            self.__dict__.update(meta=getattr(args[0], 'meta', None))
        else:
            self.__dict__.update(meta=kargs.pop('meta', None))
        super(Trial, self).__init__(*args, **kargs)

    @property
    def num_sample(self):
        return len(self.semg)

    def copy(self):
        return Trial(self)

    def asscalar(self, key):
        value = self[key]
        if not isinstance(value, (tuple, list, np.ndarray)):
            return value
        assert np.all(value[:-1] == value[1:])
        return value[0]


class BaseGetTrial(object):

    trial_cls = Trial

    def __init__(self, dataset):
        self.dataset = dataset


class StepMixin(object):

    @property
    def num_step(self):
        return len(self.steps)

    def __init__(self, *args, **kargs):
        if 'step' in kargs:
            self.steps = [kargs.pop('step')]
        else:
            self.steps = kargs.pop('steps', [1])
        super(StepMixin, self).__init__(*args, **kargs)


class SemgPrevMixin(object):

    class dataiter_cls(
        di.IndexMixin,
        di.BalanceGestureMixin,
        di.NDArrayIter
    ):

        def __init__(self, combos, dataset, preprocess=None, **kargs):
            get_trial = dataset.get_trial_func(preprocess=preprocess)
            combos = list(combos)
            semg = []
            prev = []
            gesture = []
            subject = []
            segment = []

            for combo in combos:
                trial = get_trial(combo=combo)
                semg.append(trial.semg)
                prev.append(trial.prev)
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

            semg = np.vstack(semg).reshape(-1, 1, dataset.num_semg_row, dataset.num_semg_col)
            prev = np.vstack(prev).reshape(-1, len(dataset.steps), dataset.num_semg_row, dataset.num_semg_col)
            nt.assert_equal(len(semg), len(prev))
            logger.debug('Data stacked')

            gesture = utils.get_index(np.hstack(gesture), ignores=[-1])
            subject = utils.get_index(np.hstack(subject), ignores=[-1])
            segment = np.hstack(segment)

            logger.debug('Make data iter')

            self.num_gesture = gesture.max() + 1
            self.num_subject = subject.max() + 1

            super(self.__class__, self).__init__(
                data=OrderedDict([('semg', semg)]),
                label=OrderedDict([('gesture', gesture),
                                   ('prev', prev),
                                   ('subject', subject),
                                   ('segment', segment)]),
                index=index,
                dataset=dataset,
                **kargs
            )

        @property
        def provide_label(self):
            shapes = super(self.__class__, self).provide_label
            if self.dataset.num_step > 1:
                shapes[1] = (shapes[1][0], self.provide_data[0][1])
            return shapes

        def reset(self):
            if hasattr(self, 'prev_orig'):
                self.label[1] = (self.label[1][0], self.prev_orig)
            else:
                self.prev_orig = self.label[1][1].copy()

            super(self.__class__, self).reset()

            if self.dataset.num_step > 1:
                prev = self.label[1][1]
                select = self.random_state.randint(prev.shape[1], size=prev.shape[0])
                mask = np.zeros(prev.shape, dtype=np.bool)
                for i in range(mask.shape[0]):
                    mask[i, select[i], :, :] = True
                prev = prev[mask].reshape(self.data[0][1].shape)
                self.label[1] = (self.label[1][0], prev)


class DownsampleGetTrialMixin(object):

    def get_downsample_index(self, trial):
        return utils.get_downsample_index(
            trial.num_sample,
            self.dataset.downsample,
            self.dataset.get_random_state(trial.meta)  # if trial.meta is not None else trial)
        )


class DeterministicMixin(object):
    '''
    Ensure DownsampleMixin and SemiDownsampleMixin select same indices for each
    trial over different crossval (universal or non-universal).
    '''

    def get_random_state(self, obj):
        return np.random.RandomState(int(utils.hash(obj)[:7], base=16))


class DownsampleMixin(object):
    '''Downsample label and data'''

    @property
    def train_get_trial_cls(dataset):
        class cls(DownsampleGetTrialMixin, super(DownsampleMixin, dataset).train_get_trial_cls):

            def trial_cls(self, **kargs):
                trial = super(cls, self).trial_cls(**kargs)
                index = self.get_downsample_index(trial)
                for key in list(trial):
                    trial[key] = trial[key][index]
                return trial

        return cls


class SemiDownsampleMixin(object):
    '''Only downsample gesture labels, keep data and label it as -1'''

    @property
    def train_get_trial_cls(dataset):
        class cls(DownsampleGetTrialMixin, super(SemiDownsampleMixin, dataset).train_get_trial_cls):

            def trial_cls(self, **kargs):
                trial = super(cls, self).trial_cls(**kargs)
                index = self.get_downsample_index(trial)
                mask = np.ones(trial.num_sample, dtype=np.bool)
                mask[index] = False
                trial['gesture'][mask] = -1
                return trial

        return cls


class DataIterMixin(object):

    def get_dataiter(self, **kargs):
        cls = kargs.pop('cls', self.dataiter_cls)
        return cls(dataset=self, last_batch_handle='pad', **kargs)

    @property
    def train_dataiter_cls(self):
        su = super(DataIterMixin, self)
        return su.train_dataiter_cls if hasattr(su, 'train_dataiter_cls') else self.dataiter_cls

    @property
    def val_dataiter_cls(self):
        su = super(DataIterMixin, self)
        return su.val_dataiter_cls if hasattr(su, 'val_dataiter_cls') else self.dataiter_cls


class GetTrialMixin(object):

    @property
    def train_get_trial_cls(self):
        su = super(GetTrialMixin, self)
        return su.train_get_trial_cls if hasattr(su, 'train_get_trial_cls') else self.get_trial_cls

    @property
    def val_get_trial_cls(self):
        su = super(GetTrialMixin, self)
        return su.val_get_trial_cls if hasattr(su, 'val_get_trial_cls') else self.get_trial_cls


def AdaMixin(*ignores, **kargs):
    '''Remove gesture (or other field) label for adaptive learning'''
    if 'ignores' in kargs:
        assert not ignores
        ignores = kargs.pop('ignores')

    if 'ignore' in kargs:
        assert not ignores
        ignores = kargs.pop('ignore')

    if isinstance(ignores, six.string_types):
        ignores = [ignores]

    when = kargs.pop('when', lambda trial: True)

    class ada(object):

        def get_ada_train_get_trial_cls(dataset, *when_args, **when_kargs):
            class cls(super(ada, dataset).train_get_trial_cls):

                def trial_cls(self, **kargs):
                    trial = super(cls, self).trial_cls(**kargs)
                    if when(trial, *when_args, **when_kargs):
                        for ignore in ignores:
                            a = trial[ignore].copy()
                            a[:] = -1
                            trial[ignore] = a
                    return trial

            return cls

    return ada
