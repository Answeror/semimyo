from __future__ import division
from logbook import Logger
import numpy as np
import nose.tools as nt
from .base_module import BaseModule, Accuracy, Loss, RuntimeMixin, TransformDataIterMixin
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
            data_names=['semg', 'dist_mask', 'dist'],
            label_names=['gesture'],
            **kargs
        )

    def get_eval_metric(self):
        eval_metric = [Accuracy(0, 'gesture'),
                       Loss(1, 'dist')]
        return eval_metric

    def transform_data_iter(self, data):
        return DataIter(data)


class RuntimeModule(RuntimeMixin, Module):
    pass


class DataIter(ProxyDataIter):

    @property
    def provide_data(self):
        shapes = self.base.provide_data
        return shapes + [('dist_mask', (self.batch_size, self.num_semg_pixel, 2)),
                         ('dist', (self.batch_size, 2))]

    def getdata(self):
        data = self.base.getdata()

        row = self.random_state.randint(self.num_semg_row, size=(self.batch_size, 2))
        col = self.random_state.randint(self.num_semg_col, size=(self.batch_size, 2))

        #  row_dist = np.abs(row[:, 0] - row[:, 1])
        #  if self.circular:
            #  row_dist = np.minimum(row_dist, self.num_semg_row - row_dist)
        #  dist = np.sqrt(np.square(row_dist) + np.square(col[:, 0] - col[:, 1]))
        #  if self.circular:
            #  dist /= ((self.num_semg_row / 2) ** 2 + self.num_semg_col ** 2) ** 0.5
        #  else:
            #  dist /= (self.num_semg_row ** 2 + self.num_semg_col ** 2) ** 0.5

        def get_dist_axis(axis, num_semg_axis):
            nt.assert_greater_equal(num_semg_axis, 4)
            dist_axis = axis[:, 1] % (num_semg_axis // 2) - num_semg_axis // 4
            axis[:, 1] = (dist_axis + num_semg_axis + axis[:, 0]) % num_semg_axis
            dist_axis = dist_axis.astype(np.float32) / (num_semg_axis // 4)
            return dist_axis

        if not self.circular:
            dist_row = (row[:, 0] - row[:, 1]) / self.num_semg_row
            dist_col = (col[:, 0] - col[:, 1]) / self.num_semg_col
        elif self.circular == 'row':
            dist_row = get_dist_axis(row, self.num_semg_row)
            dist_col = (col[:, 0] - col[:, 1]) / self.num_semg_col
        elif self.circular == 'col':
            dist_col = get_dist_axis(col, self.num_semg_col)
            dist_row = (row[:, 0] - row[:, 1]) / self.num_semg_row
        else:
            assert False

        dist = np.vstack([dist_row, dist_col]).T

        dist_mask = np.zeros((self.batch_size, self.num_semg_row, self.num_semg_col, 2),
                             dtype=np.float32)
        for i in range(self.batch_size):
            for j in range(2):
                dist_mask[i, row[i, j], col[i, j], j] = 1
        dist_mask = dist_mask.reshape(self.batch_size, self.num_semg_pixel, 2)

        return data + [self.asmxnd(dist_mask), self.asmxnd(dist)]
