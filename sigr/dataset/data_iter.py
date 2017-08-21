from __future__ import division
import mxnet as mx
import numpy as np
import six
from logbook import Logger
import nose.tools as nt
from collections import OrderedDict
from .. import constant, utils


logger = Logger(__name__)


class NDArrayIter(mx.io.NDArrayIter):

    @property
    def num_sample(self):
        return self.num_data

    def __init__(self, **kargs):
        self.shuffle = kargs.pop('shuffle')
        self.random_state = kargs.pop('random_state', np.random)
        if 'dataset' in kargs:
            self.dataset = kargs.pop('dataset')

        names = ['data', 'label', 'batch_size', 'shuffle',
                 'last_batch_handle', 'label_name']
        for key in list(kargs):
            if key not in names:
                logger.debug('Unused param: ' + key)
                del kargs[key]

        super(NDArrayIter, self).__init__(shuffle=False, **kargs)

        self.data = [(k, self.asnumpy(v)) for k, v in self.data]
        self.label = [(k, self.asnumpy(v)) for k, v in self.label]
        self.reset()

    def asmxnd(self, a):
        return a if isinstance(a, mx.nd.NDArray) else mx.nd.array(a)

    def asnumpy(self, a):
        return a if not isinstance(a, mx.nd.NDArray) else a.asnumpy()

    def _getdata(self, data_source):
        assert self.cursor < self.num_data, "DataIter needs reset."

        if self.cursor + self.batch_size <= self.num_data:
            res = [x[1][self.get_inner_index(slice(self.cursor, self.cursor + self.batch_size))]
                   for x in data_source]
        else:
            pad = self.batch_size - self.num_data + self.cursor
            res = [np.concatenate((x[1][self.get_inner_index(slice(self.cursor, None))],
                                   x[1][self.get_inner_index(slice(None, pad))]),
                                  axis=0) for x in data_source]

        res = self._getdata_postprocess(data_source, res)
        return [self.asmxnd(a) for a in res]

    def _getdata_postprocess(self, data_source, res):
        return res

    def do_shuffle(self):
        index = np.arange(self.num_data)
        self.random_state.shuffle(index)
        self.select(index)

    @property
    def input_names(self):
        return [name for name, _ in self.provide_data + self.provide_label]

    def get_input(self, name):
        return dict(self.data + self.label)[name]

    @property
    def data_names(self):
        return [name for name, _ in self.provide_data]

    @property
    def label_names(self):
        return [name for name, _ in self.provide_label]

    def get_input_from_batch(self, name, batch):
        if name in self.data_names:
            return batch.data[self.data_names.index(name)]
        if name in self.label_names:
            return batch.label[self.label_names.index(name)]
        raise Exception('No field named: ' + name)

    def get_input_from_batch_all(self, name):
        self.reset()
        values = []
        for batch in self:
            value = self.get_input_from_batch(name, batch).asnumpy()
            if batch.pad:
                value = value[:-batch.pad]
            values.append(value)
        self.reset()
        return np.concatenate(values, axis=0)

    @property
    def standard_names(self):
        return ['data', 'label', 'provide_data', 'provide_label', 'batch_size']

    def __getattr__(self, name):
        try:
            if name in self.standard_names or name not in self.input_names:
                return self.__getattribute__(name)
            return self.get_input(name)
        except:
            if hasattr(self, 'dataset'):
                return getattr(self.dataset, name)
            raise

    def reset(self):
        super(NDArrayIter, self).reset()
        if self.shuffle:
            self.do_shuffle()

    def get_inner_index(self, index):
        return index

    def select(self, index):
        self.data = [(k, v[index]) for k, v in self.data]
        self.label = [(k, v[index]) for k, v in self.label]


class MiniBatchMixin(object):

    def __init__(self, *args, **kargs):
        self.__num_mini_batch = kargs.pop('num_mini_batch', constant.NUM_MINI_BATCH)
        super(MiniBatchMixin, self).__init__(*args, **kargs)

    def do_shuffle(self):
        if not (self.__num_mini_batch > 1) or len(set(self.mini_batch_index)) == 1:
            return super(MiniBatchMixin, self).do_shuffle()

        _index = np.arange(self.num_data)
        assert self.batch_size % self.__num_mini_batch == 0
        mini_batch_size = self.batch_size // self.__num_mini_batch

        logger.info('AdaBN shuffle with a mini batch size of {}', mini_batch_size)
        self.random_state.shuffle(_index)
        subject_shuffled = self.mini_batch_index[_index]
        index_batch = []
        for i in sorted(set(self.mini_batch_index)):
            index = _index[subject_shuffled == i]
            index = index[:len(index) // mini_batch_size * mini_batch_size]
            index_batch.append(index.reshape(-1, mini_batch_size))
        index_batch = np.vstack(index_batch)
        index = np.arange(len(index_batch))
        self.random_state.shuffle(index)
        _index = index_batch[index, :].ravel()

        for i in range(0, len(self.mini_batch_index), mini_batch_size):
            # Make sure that the samples in one batch are from the same subject
            assert np.all(self.mini_batch_index[_index[i:i + mini_batch_size - 1]] ==
                          self.mini_batch_index[_index[i + 1:i + mini_batch_size]])

        if mini_batch_size != self.batch_size:
            assert self.batch_size % mini_batch_size == 0
            _index = _index[:len(_index) // self.batch_size * self.batch_size].reshape(
                -1, self.__num_mini_batch, mini_batch_size).transpose(0, 2, 1).ravel()

        self.select(_index)


class BalanceGestureMixin(object):

    def __init__(self, *args, **kargs):
        dataset = kargs.get('dataset')
        self.__balance_gesture = utils.get(kargs, 'balance_gesture')(dataset).default(False)
        self.__balance_gesture_ignore_ratio = self._parse_number(
            utils.get(kargs, 'balance_gesture_ignore_ratio')(dataset).default(None))
        self.__balance_gesture_ignore_downsample = self._parse_number(
            utils.get(kargs, 'balance_gesture_ignore_downsample')(dataset).default(1))
        super(BalanceGestureMixin, self).__init__(*args, **kargs)

    def _parse_number(self, x):
        if isinstance(x, six.string_types):
            if x.endswith('%'):
                return float(x[:-1]) / 100
            elif '/' in x:
                up, down = x.split('/')
                return float(up) / float(down)
            else:
                return None
        return x

    @property
    def __num_data_balanced(self):
        ratio = self._parse_number(self.__balance_gesture)
        if ratio <= 1:
            return int(round(ratio * self.num_data))
        n = int(ratio)
        nt.assert_equal(n, ratio)
        return n

    def do_balance_gesture(self):
        gestures = sorted(set(self.gesture))
        num_gesture = len(gestures)

        n = self.__num_data_balanced
        if gestures[0] >= 0:
            num_sample_per_gesture = [int(round(n / num_gesture))] * num_gesture
        else:
            ignore_ratio = self.__balance_gesture_ignore_ratio or np.sum(self.gesture < 0) / self.num_data
            num_sample_per_gesture = [int(round(n * ignore_ratio * self.__balance_gesture_ignore_downsample))]
            assert gestures[1] >= 0
            num_sample_per_gesture += [int(round((1 - ignore_ratio) * n / (num_gesture - 1)))] * (num_gesture - 1)

        logger.debug('Balance gesture to {} samples', sum(num_sample_per_gesture))
        logger.debug('Balanced gestures: {}', num_sample_per_gesture)

        choice = []
        for i, gesture in enumerate(gestures):
            mask = self.gesture == gesture
            choice.append(self.random_state.choice(np.where(mask)[0],
                                                   num_sample_per_gesture[i]))
        choice = np.sort(np.hstack(choice))
        self.select(choice)

    def reset(self):
        with utils.restore_field(self, shuffle=False):
            super(BalanceGestureMixin, self).reset()
            if self.__balance_gesture:
                logger.debug('Balance gesture')
                self.do_balance_gesture()

        if self.shuffle:
            self.do_shuffle()


class IndexMixin(object):

    def __init__(self, *args, **kargs):
        self.__index = np.asarray(kargs.pop('index'))
        self.__index_orig = self.__index.copy()
        super(IndexMixin, self).__init__(*args, **kargs)

    @property
    def num_data(self):
        return len(self.__index)

    @num_data.setter
    def num_data(self, value):
        if value != len(self.__index):
            logger.warn('num_data ({}) cannot be set with {}',
                        len(self.__index), value)
        return len(self.__index)

    def get_inner_index(self, index):
        return super(IndexMixin, self).get_inner_index(self.__index[index])

    def select(self, index):
        self.__index = self.__index[index]
        logger.debug('Index selected: {}...', list(self.__index[:10]))

    def reset(self):
        self.__index = self.__index_orig.copy()
        super(IndexMixin, self).reset()

    def get_input(self, name):
        return super(IndexMixin, self).get_input(name)[self.__index]


class DownsampleMixin(object):

    def __init__(self, *args, **kargs):
        self.__downsample = kargs.pop('downsample', None)
        self.__keep = kargs.pop('keep', False)
        super(DownsampleMixin, self).__init__(*args, **kargs)

    def do_downsample(self):
        if callable(self.__downsample):
            self.__downsample(self)
        else:
            if self.__downsample < 1:
                samples = np.arange(self.num_data)
                np.random.RandomState(667).shuffle(samples)
                assert self.__downsample > 0 and self.__downsample <= 1
                samples = samples[:int(round(len(samples) * self.__downsample))]
                assert len(samples) > 0
                self.select(samples)
            else:
                logger.debug('Determnistic downsample')
                assert hasattr(self, 'segment')
                step = int(self.__downsample)
                nt.assert_equal(step, self.__downsample)
                from .downsample import downsample_even_segment as des
                segment = self.segment
                n = len(segment)
                index = des(step, segment)
                if self.__keep:
                    logger.debug('Upsample to original size')
                    index = self.random_state.choice(index, n)
                self.select(index)

    def reset(self):
        shuffle = self.shuffle
        self.shuffle = False
        try:
            super(DownsampleMixin, self).reset()
            if self.__downsample:
                logger.debug('Downsample')
                self.do_downsample()
        finally:
            self.shuffle = shuffle
        if self.shuffle:
            self.do_shuffle()


class WindowMixin(object):

    def __init__(self, *args, **kargs):
        dataset = kargs.get('dataset')
        self.window = utils.get(kargs, 'window')(dataset).default(1)
        self.skip = utils.get(kargs, 'skip')(dataset).default(0)

        if self.window > 1:
            nt.assert_is_instance(self, IndexMixin)
            index = kargs.pop('index')
            segment = WindowMixin._get_segment(kargs['data'], kargs['label'])
            index = np.array([begin for begin in index
                              if WindowMixin._good_window(begin, self.window, self.skip, segment)])
            kargs['index'] = index

        super(WindowMixin, self).__init__(*args, **kargs)

    @property
    def provide_data(self):
        return [(k, (self.window * v[0],) + v[1:]) for k, v
                in super(WindowMixin, self).provide_data]

    @property
    def provide_label(self):
        return [(k, (self.window * v[0],) + v[1:]) for k, v
                in super(WindowMixin, self).provide_label]

    def get_inner_index(self, index):
        inner_index = super(WindowMixin, self).get_inner_index(index)
        return self._expand_index(inner_index) if self.window > 1 else inner_index

    def _expand_index(self, index):
        return np.hstack([np.arange(i, i + self.window * (self.skip + 1), self.skip + 1) for i in index])

    @staticmethod
    def _get_segment(*args):
        data = {}
        for arg in args:
            if isinstance(arg, list):
                arg = dict(arg)
            data.update(arg)
        return data['segment']

    @staticmethod
    def _good_window(begin, window, skip, segment):
        return begin + (window - 1) * (skip + 1) < len(segment) and \
            np.all(segment[begin:begin + window * (skip + 1):skip + 1] == segment[begin])


class ProxyDataIter(object):

    def __init__(self, base):
        self.base = base

    def __iter__(self):
        return self

    def __getattr__(self, name):
        if name in self.__class__.__dict__:
            return self.__getattribute__(name)
        return getattr(self.base, name)

    def __setattr__(self, name, value):
        if name not in ['base'] and hasattr(self.base, name):
            return setattr(self.base, name, value)
        return super(ProxyDataIter, self).__setattr__(name, value)

    def __next__(self):
        return self.next()

    def next(self):
        if self.iter_next():
            return mx.io.DataBatch(data=self.getdata(), label=self.getlabel(),
                                   pad=self.getpad(), index=self.getindex())
        else:
            raise StopIteration


class IgnoreInputDataIter(ProxyDataIter):

    def __init__(self, base, ignore_data=[], ignore_label=[]):
        super(IgnoreInputDataIter, self).__init__(base)

        self.ignore_data = []
        for i, (name, _) in enumerate(base.provide_data):
            if name in ignore_data:
                self.ignore_data.append(i)
        self.ignore_label = []
        for i, (name, _) in enumerate(base.provide_label):
            if name in ignore_label:
                self.ignore_label.append(i)

    @property
    def provide_data(self):
        return [item for i, item in enumerate(self.base.provide_data)
                if i not in self.ignore_data]

    @property
    def provide_label(self):
        return [item for i, item in enumerate(self.base.provide_label)
                if i not in self.ignore_label]

    def getdata(self):
        return [item for i, item in enumerate(self.base.getdata())
                if i not in self.ignore_data]

    def getlabel(self):
        return [item for i, item in enumerate(self.base.getlabel())
                if i not in self.ignore_label]


def _as_ordered_dict(d):
    if not isinstance(d, OrderedDict):
        d = OrderedDict(d)
    return d


def InitMixin(data_shapes, label_shapes, indices):
    data_shapes = _as_ordered_dict(data_shapes)
    label_shapes = _as_ordered_dict(label_shapes)

    shapes = OrderedDict(list(data_shapes.items()) + list(label_shapes.items()))
    main_field = list(data_shapes)[0]

    class cls(object):

        def __init__(self, combos, dataset, preprocess=None, **kargs):
            get_trial_cls = kargs.pop('get_trial_cls', dataset.get_trial_cls)
            get_trial = get_trial_cls(dataset=dataset, preprocess=preprocess)
            combos = list(combos)

            fields = utils.Bunch(segment=[])
            for key in shapes:
                fields[key] = []

            for combo in combos:
                trial = get_trial(combo=combo)
                for key in shapes:
                    fields[key].append(getattr(trial, key))
                fields.segment.append(np.repeat(len(fields.segment),
                                                len(fields[main_field][-1])))

            logger.debug('Combos loaded')
            assert fields[main_field], 'Empty data'

            index = []
            n = 0
            for seg in fields[main_field]:
                index.append(np.arange(n, n + len(seg)))
                n += len(seg)
            index = np.hstack(index)
            logger.debug('Index made')

            logger.debug('Segments: {}', len(fields[main_field]))
            logger.debug('First segment shape: {}', fields[main_field][0].shape)

            for key in shapes:
                fields[key] = np.concatenate(fields[key], axis=0).reshape(shapes[key])
            fields.segment = np.hstack(fields.segment)
            logger.debug('Data stacked')

            for key in indices:
                fields[key] = utils.get_index(fields[key], ignores=[-1])
                setattr(self, 'num_' + key, fields[key].max() + 1)
                logger.debug('Re-index {}, num {}, valid ratio {}',
                             key, getattr(self, 'num_' + key),
                             (fields[key] >= 0).sum() / len(fields[key]))

            super(cls, self).__init__(
                data=OrderedDict([(key, fields[key]) for key in data_shapes]),
                label=OrderedDict([(key, fields[key]) for key in list(label_shapes) + ['segment']]),
                index=index,
                dataset=dataset,
                **kargs
            )

    return cls


class PrevMixin(object):

    @property
    def provide_label(self):
        shapes = super(PrevMixin, self).provide_label
        if self.dataset.num_step > 1:
            shapes[1] = (shapes[1][0], self.provide_data[0][1])
        return shapes

    def reset(self):
        if hasattr(self, 'prev_orig'):
            self.label[1] = (self.label[1][0], self.prev_orig)
        else:
            self.prev_orig = self.label[1][1].copy()

        super(PrevMixin, self).reset()

        if self.dataset.num_step > 1:
            prev = self.label[1][1]
            select = self.random_state.randint(prev.shape[1], size=prev.shape[0])
            mask = np.zeros(prev.shape, dtype=np.bool)
            for i in range(mask.shape[0]):
                mask[i, select[i], :, :] = True
            prev = prev[mask].reshape(self.data[0][1].shape)
            self.label[1] = (self.label[1][0], prev)


def MuteDataIter(fields):
    def mute(source, shapes):
        for (key, _), value in zip(shapes, source):
            if key in fields:
                value[:] = -1
        return source

    class cls(ProxyDataIter):

        def getdata(self):
            return mute(self.base.getdata(), self.provide_data)

        def getlabel(self):
            return mute(self.base.getlabel(), self.provide_label)

    return cls
