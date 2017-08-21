from __future__ import division
import os
import numpy as np
from functools import partial
from logbook import Logger
import mxnet as mx
from .. import utils, constant
from ..get_data_iter import get_data_iter


logger = Logger(__name__)
Exp = utils.Bunch


def _transform(Mod, func, args_not_cached, *args, **kargs):
    return getattr(Mod(**args_not_cached), func)(*args, **kargs)


class BaseEvaluation(object):

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def transform(self, Mod, func, *args, **kargs):
        return getattr(Mod(), func)(*args, **kargs)

    def predict(self, Mod, *args, **kargs):
        ret = self.transform(Mod, self.predict_func_name, *args, **kargs)
        if any(ret.true < 0):
            mask = ret.true >= 0
            ret = utils.Bunch({key: value[mask] for key, value in ret.items()})
        return ret

    @property
    def predict_func_name(self):
        return 'predict'

    @property
    def args_not_cached(self):
        return {}


class StochasticPredictionMixin(object):

    @property
    def predict_func_name(self):
        return 'predict_stochastic'


class CacheMixin(object):
    '''Cache predict and transform etc.'''

    def transform(self, Mod, func, *args, **kargs):
        return utils.cached_call(partial(_transform, Mod, func),
                                 self.args_not_cached,
                                 *args, **kargs)


class MxNetMixin(object):

    def __init__(self, context=[mx.gpu(0)], **kargs):
        super(MxNetMixin, self).__init__(**kargs)
        self.context = context

    @property
    def args_not_cached(self):
        args = super(MxNetMixin, self).args_not_cached.copy()
        args.update(context=self.context)
        return args


class CrossvalMixin(object):

    def __init__(self, crossval, **kargs):
        super(CrossvalMixin, self).__init__(**kargs)
        self.crossval = crossval.replace('-', '_')

    def _bind_fold(self, func, fold):
        # Convert np.int64 and othre types to int, for cache
        fold = int(fold)
        return partial(func, fold=fold)

    def _get_data(self, dataset, fold, **kargs):
        func = partial(
            get_data_iter,
            dataset,
            self.crossval + '_val',
            batch_size=self.batch_size,
            **kargs
        )
        func = self._bind_fold(func, fold)
        return utils.LazyProxy(func)

    def calc_accuracy(self, Mod, dataset, fold,
                      vote=False,
                      dataset_args=None,
                      balance=constant.BALANCE_LABEL):
        Mod = self._bind_fold(Mod, fold)
        data = self._get_data(dataset, fold, **dataset_args)
        ret = self.predict(Mod, data)
        if vote or balance:
            from ..vote import vote as do
            window = vote or 1
            return do(ret.true, ret.pred, ret.segment, window, balance)
        return (ret.true == ret.pred).sum() / ret.true.size

    def _accuracy(self, exp, fold):
        if hasattr(exp, 'Mod') and hasattr(exp, 'dataset'):
            return self.calc_accuracy(Mod=exp.Mod,
                                      dataset=exp.dataset,
                                      fold=fold,
                                      vote=exp.get('vote', False),
                                      dataset_args=exp.get('dataset_args'),
                                      balance=exp.get('balance', constant.BALANCE_LABEL))
        else:
            try:
                from ..parse_log import parse_log
                return parse_log(os.path.join(exp.root % fold, 'log')).val.iloc[-1]
            except:
                return np.nan

    def accuracy(self, **kargs):
        if 'exp' in kargs:
            return self._accuracy(**kargs)
        elif 'Mod' in kargs:
            return self.calc_accuracy(**kargs)
        else:
            raise Exception('Neither exp nor Mod is given')

    def accuracies(self, exps, folds):
        acc = []
        for exp in exps:
            for i, fold in enumerate(folds):
                logger.debug('Fold {}/{}', i + 1, len(folds))
                acc.append(self.accuracy(exp=exp, fold=fold))
        return np.array(acc).reshape(len(exps), len(folds))

    def vote_accuracy_curves(self, exps, folds, windows, balance=constant.BALANCE_LABEL):
        acc = []
        for exp in exps:
            for i, fold in enumerate(folds):
                logger.debug('Fold {}/{}', i + 1, len(folds))
                acc.append(self.vote_accuracy_curve(
                    Mod=exp.Mod,
                    dataset=exp.dataset,
                    fold=int(fold),
                    windows=windows,
                    dataset_args=exp.get('dataset_args'),
                    balance=balance))
        return np.array(acc).reshape(len(exps), len(folds), len(windows))

    def vote_accuracy_curve(self, Mod, dataset, fold, windows,
                            dataset_args=None,
                            balance=constant.BALANCE_LABEL):
        Mod = self._bind_fold(Mod, fold)
        data = self._get_data(dataset, fold, **dataset_args)
        ret = self.predict(Mod, data)
        from ..vote import get_vote_accuracy_curve as do
        return do(ret.true, ret.pred, ret.segment, windows, balance)[1]
