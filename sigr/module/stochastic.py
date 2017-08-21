from __future__ import division
import mxnet as mx
import numpy as np
from nose.tools import assert_equal
from .. import utils


class StochasticBugfixMixin(object):

    def predict_proba(self, eval_data, num_batch=None, merge_batches=True,
                      reset=True, always_output_list=False):
        if not self.binded:
            self.bind(data_shapes=eval_data.provide_data, for_training=True)
        if not self.params_initialized:
            self.init_params(self.get_loader())

        if reset:
            eval_data.reset()

        output_list = []

        for nbatch, eval_batch in enumerate(eval_data):
            if num_batch is not None and nbatch == num_batch:
                break
            self.forward(eval_batch, is_train=True)
            pad = eval_batch.pad
            outputs = [out[0:out.shape[0]-pad].copy() for out in self.get_outputs()]

            output_list.append(outputs)

        if len(output_list) == 0:
            return output_list

        if merge_batches:
            num_outputs = len(output_list[0])
            for out in output_list:
                assert len(out) == num_outputs, \
                    'Cannot merge batches, as num of outputs is not the same ' + \
                    'in mini-batches. Maybe bucketing is used?'
            output_list2 = [mx.nd.concatenate([out[i] for out in output_list])
                            for i in range(num_outputs)]

            if num_outputs == 1 and not always_output_list:
                return output_list2[0]
            return output_list2

        return output_list


class StochasticPredictionMixin(StochasticBugfixMixin):

    def predict_proba(self, eval_data, *args, **kargs):
        '''
        One sampling of MC Dropout
        '''
        out = super(StochasticPredictionMixin, self).predict_proba(
            eval_data, *args, **kargs).asnumpy()
        assert_equal(out.ndim, 2)
        return utils.Bunch(pred=out, **self.get_predict_meta(eval_data))


class MonteCarloPredictionMixin(StochasticPredictionMixin):

    def predict_proba(self, eval_data, *args, **kargs):
        '''
        MC Dropout

        Gal, Yarin, and Zoubin Ghahramani. "Dropout as a Bayesian
        approximation: Representing model uncertainty in deep learning."
        ICML (2016).
        '''
        num_sample = kargs.pop('num_sample', None)
        if num_sample is None:
            num_sample = self.num_mc_sample
        func = lambda: super(MonteCarloPredictionMixin, self).predict_proba(
            eval_data, *args, **kargs)
        out = func()
        out.pred = np.mean([out.pred] + [func().pred for i in
                                         range(1, num_sample)], axis=0)
        return out
