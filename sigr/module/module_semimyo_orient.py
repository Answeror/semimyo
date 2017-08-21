from __future__ import division
from logbook import Logger
import numpy as np
import nose.tools as nt
from .base_module import BaseModule, Accuracy, RuntimeMixin, TransformDataIterMixin
from .adabn import AdaBNMixin
from ..dataset.data_iter import ProxyDataIter


logger = Logger('module')


class Module(TransformDataIterMixin, AdaBNMixin, BaseModule):

    def __init__(self, **kargs):
        self.kargs = kargs.copy()
        self.for_training = kargs.pop('for_training')
        self.snapshot_period = kargs.pop('snapshot_period', 1)
        symbol_kargs = kargs.pop('symbol_kargs', {}).copy()
        symbol_kargs.update(
            for_training=self.for_training,
            network=kargs.pop('network')
        )
        kargs['symbol_kargs'] = symbol_kargs
        super(Module, self).__init__(
            data_names=['semg'],
            label_names=['gesture', 'orient', 'orient_mask'],
            **kargs
        )

    def get_eval_metric(self):
        eval_metric = [Accuracy(0, 'gesture'),
                       Accuracy(1, 'orient')]
        return eval_metric

    def transform_data_iter(self, data):
        return DataIter(data)


class RuntimeModule(RuntimeMixin, Module):
    pass


class DataIter(ProxyDataIter):

    def __init__(self, *args, **kargs):
        super(DataIter, self).__init__(*args, **kargs)

        self._row_radius = max(1, self.num_semg_row // 3)
        self._col_radius = max(1, self.num_semg_col // 3)
        self._orients = []
        self._orient_labels = []
        for row in range(-self._row_radius, self._row_radius + 1):
            for col in range(-self._col_radius, self._col_radius + 1):
                if row or col:
                    self._orients.append((row, col))
                    if row < 0 and col <= 0:
                        label = 0
                    elif row >= 0 and col < 0:
                        label = 1
                    elif row > 0 and col >= 0:
                        label = 2
                    elif row <= 0 and col > 0:
                        label = 3
                    else:
                        assert False
                    self._orient_labels.append(label)
        self._num_orient = len(self._orients)
        nt.assert_equal(self._num_orient, (self._row_radius * 2 + 1) * (self._col_radius * 2 + 1) - 1)

        self._good_orients = []
        for row in range(self.num_semg_row):
            self._good_orients.append([])
            for col in range(self.num_semg_col):
                self._good_orients[-1].append(self._get_good_orient(row, col))

        self._target = np.empty((self.num_semg_row, self.num_semg_col, self._num_orient), dtype=np.object)
        for row in range(self.num_semg_row):
            for col in range(self.num_semg_col):
                for ori in range(self._num_orient):
                    rt = row + self._orients[ori][0]
                    ct = col + self._orients[ori][1]
                    if self.circular == 'row':
                        rt = (rt + self.num_semg_row) % self.num_semg_row
                    if self.circular == 'col':
                        ct = (ct + self.num_semg_col) % self.num_semg_col
                    self._target[row, col, ori] = (rt, ct)

    @property
    def provide_label(self):
        shapes = self.base.provide_label
        return shapes[:1] + [('orient', (self.batch_size,)),
                             ('orient_mask', (self.batch_size, self.num_semg_pixel, 2))]

    def _get_good_orient(self, row, col):
        good = []
        for i, (ro, co) in enumerate(self._orients):
            rt = row + ro
            ct = col + co
            if (self.circular == 'row' or rt >= 0 and rt < self.num_semg_row) \
                    and (self.circular == 'col' or ct >= 0 and ct < self.num_semg_col):
                good.append(i)
        nt.assert_greater(len(good), 0)
        return good

    def getlabel(self):
        label = self.base.getlabel()

        row = self.random_state.randint(self.num_semg_row, size=self.batch_size)
        col = self.random_state.randint(self.num_semg_col, size=self.batch_size)
        orient = self.random_state.randint(self._num_orient, size=self.batch_size)
        for i in range(self.batch_size):
            good = self._good_orients[row[i]][col[i]]
            orient[i] = self._orient_labels[good[orient[i] % len(good)]]

        orient_mask = np.zeros((self.batch_size, self.num_semg_row, self.num_semg_col, 2), dtype=np.float32)
        for i in range(self.batch_size):
            orient_mask[i, row[i], col[i], 0] = 1
            rt, ct = self._target[row[i], col[i], orient[i]]
            orient_mask[i, rt, ct, 1] = 1
        orient_mask = orient_mask.reshape(self.batch_size, self.num_semg_pixel, 2)

        return label[:1] + [self.asmxnd(orient), self.asmxnd(orient_mask)]
