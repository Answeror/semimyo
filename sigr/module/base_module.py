from __future__ import division
from functools import partial
import numpy as np
import mxnet as mx
from .. import utils
from .module_common import BaseModule, Accuracy, RMSE, TransformDataIterMixin


class RuntimeMixin(object):

    def __init__(self, **kargs):
        args = []
        backup = kargs.copy()
        self.params = kargs.pop('params')
        super(RuntimeMixin, self).__init__(**kargs)
        self.args = args
        self.kargs = backup

    def predict_proba(self, eval_data, *args, **kargs):
        if not self.binded:
            self.bind(data_shapes=eval_data.provide_data, for_training=False)
        if not self.params_initialized:
            self.init_params(self.get_loader())

        out = super(RuntimeMixin, self).predict(eval_data, *args, **kargs).asnumpy()
        return utils.Bunch(pred=out, **self.get_predict_meta(eval_data))

    def predict(self, eval_data, *args, **kargs):
        out = self.predict_proba(eval_data, *args, **kargs)
        out.pred = out.pred.argmax(axis=1)
        #  import scipy.io as sio
        #  sio.savemat('out.mat', out)
        return out

    def get_predict_meta(self, eval_data):
        if hasattr(eval_data, 'get_input_from_batch_all'):
            return dict(true=eval_data.get_input_from_batch_all('gesture'),
                        segment=eval_data.get_input_from_batch_all('segment'))
        else:
            eval_data.reset()
            true = eval_data.gesture.copy()
            segment = eval_data.segment.copy()
            eval_data.reset()
            assert np.all(true == eval_data.gesture.copy())
            assert np.all(segment == eval_data.segment.copy())
            return dict(true=true, segment=segment)

    @property
    def Clone(self):
        return partial(type(self), *self.args, **self.kargs)


class Loss(mx.metric.EvalMetric):

    def __init__(self, index, name):
        super(Loss, self).__init__('loss[{}]'.format(name))
        if not isinstance(index, list):
            index = [index]
        self.index = index

    def update(self, _, preds):
        for index in self.index:
            pred = preds[index].asnumpy()
            self.sum_metric += pred.mean()
            self.num_inst += 1


__all__ = ['BaseModule', 'RuntimeMixin', 'Accuracy', 'RMSE', 'Loss', 'TransformDataIterMixin']
