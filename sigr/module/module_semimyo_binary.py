from __future__ import division
from logbook import Logger
from .base_module import BaseModule, Accuracy, RuntimeMixin
from .ignore_input import IgnoreInputMixin
from .adabn import AdaBNMixin


logger = Logger('module')


class Module(IgnoreInputMixin, AdaBNMixin, BaseModule):

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

        self.ignore_label = []
        if not symbol_kargs.get('gesture_loss_weight'):
            self.ignore_label.append('gesture')
        if not symbol_kargs.get('binary_loss_weight'):
            self.ignore_label.append('binary')

        super(Module, self).__init__(
            data_names=['semg'],
            label_names=['gesture', 'binary'],
            ignore_label=self.ignore_label,
            **kargs
        )

    @property
    def has_gesture_branch(self):
        return 'gesture' not in self.ignore_label

    @property
    def has_binary_branch(self):
        return 'binary' not in self.ignore_label

    def get_eval_metric(self):
        eval_metric = []
        if self.has_gesture_branch:
            eval_metric.append(Accuracy(len(eval_metric), 'gesture'))
        if self.has_binary_branch:
            eval_metric.append(Accuracy(len(eval_metric), 'binary'))
        return eval_metric


class RuntimeModule(RuntimeMixin, Module):
    pass
