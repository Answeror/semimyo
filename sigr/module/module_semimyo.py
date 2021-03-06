from __future__ import division
from logbook import Logger
from .base_module import BaseModule, Accuracy, RuntimeMixin
from .adabn import AdaBNMixin


logger = Logger('module')


class Module(AdaBNMixin, BaseModule):

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
            label_names=['gesture'],
            **kargs
        )

    def get_eval_metric(self):
        return [Accuracy(0, 'gesture')]


class RuntimeModule(RuntimeMixin, Module):
    pass
