from __future__ import division
import re
import os
import time
import logging
#  from contextlib import contextmanager
from logbook import Logger
import mxnet as mx
import numpy as np
import nose.tools as nt
from nose.tools import assert_equal
from .. import constant, ROOT, utils
from ..get_symbol import get_symbol


logger = Logger('module')


class TransformDataIterMixin(object):

    def transform_data_iter(self, data):
        return data

    def fit(self, **kargs):
        train_data = kargs.pop('train_data')
        eval_data = kargs.pop('eval_data')
        return super(TransformDataIterMixin, self).fit(
            train_data=self.transform_data_iter(train_data),
            eval_data=self.transform_data_iter(eval_data),
            **kargs
        )


class IgnoreInputMixin(object):

    def __init__(self, **kargs):
        data_names = kargs.pop('data_names', [])
        label_names = kargs.pop('label_names', [])
        ignore_data = kargs.pop('ignore_data', [])
        ignore_label = kargs.pop('ignore_label', [])
        if not ignore_data and not ignore_label:
            self._wrap_dataiter = lambda x: x
        else:
            from ..data import IgnoreInputDataIter
            data_names = [name for name in data_names if name not in ignore_data]
            label_names = [name for name in label_names if name not in ignore_label]
            self._wrap_dataiter = lambda x: IgnoreInputDataIter(
                x,
                ignore_data=ignore_data,
                ignore_label=ignore_label
            )

        super(IgnoreInputMixin, self).__init__(
            data_names=data_names,
            label_names=label_names,
            **kargs
        )

    def fit(self, **kargs):
        train_data = kargs.pop('train_data')
        eval_data = kargs.pop('eval_data')
        return super(IgnoreInputMixin, self).fit(
            train_data=self._wrap_dataiter(train_data),
            eval_data=self._wrap_dataiter(eval_data),
            **kargs
        )


class SymbolMixin(object):

    def __init__(self, **kargs):
        symbol = kargs.pop('symbol', None)
        symbol_kargs = kargs.pop('symbol_kargs', {}).copy()
        if symbol is None:
            symbol = get_symbol(symbol_kargs.pop('network'), **symbol_kargs)
        super(SymbolMixin, self).__init__(symbol=symbol, **kargs)


class AdaBNMixin(object):

    @property
    def adabn(self):
        return utils.g.get(utils.hash(self.symbol), {}).get('adabn', 0)

    @property
    def num_adabn_epoch(self):
        return utils.g.get(utils.hash(self.symbol), {}).get('num_adabn_epoch', constant.NUM_ADABN_EPOCH)

    #  def __init__(self, **kargs):
        #  parent = kargs.get('parent', {})
        #  self.downsample = kargs.pop('downsample', getattr(parent, 'downsample', False))
        #  self.adabn = kargs.pop('adabn', getattr(parent, 'adabn', False))
        #  self.num_adabn_epoch = kargs.pop(
            #  'num_adabn_epoch',
            #  getattr(parent, 'num_adabn_epoch', constant.NUM_ADABN_EPOCH)
        #  )
        #  super(AdaBNMixin, self).__init__(**kargs)

    #  @contextmanager
    #  def _restore_eval_data(self, eval_data):
        #  shuffle = eval_data.shuffle
        #  eval_data.shuffle = True
        #  if hasattr(eval_data, 'base'):
            #  assert eval_data.base.shuffle
        #  downsample = eval_data.downsample
        #  eval_data.downsample = self.downsample
        #  last_batch_handle = eval_data.last_batch_handle
        #  eval_data.last_batch_handle = 'roll_over'
        #  try:
            #  yield
        #  finally:
            #  eval_data.shuffle = shuffle
            #  eval_data.downsample = downsample
            #  eval_data.last_batch_handle = last_batch_handle

    def _update_adabn(self, eval_data):
        '''Update moving mean and moving var with eval data'''
        start = time.time()
        with utils.restore_field(eval_data, shuffle=True, last_batch_handle='roll_over'):
            for _ in range(self.num_adabn_epoch):
                eval_data.reset()
                for nbatch, eval_batch in enumerate(eval_data):
                    self.forward(eval_batch, is_train=True)
                    for out in self.get_outputs():
                        #  Cause memory leak (though not increase after this _update_adabn) without this wait
                        #  TODO: fixme
                        out.wait_to_read()
                    #  for name, block in zip(self._exec_group.aux_names, self._exec_group.aux_arrays):
                        #  if 'moving' in name:
                            #  for a in block:
                                #  a.wait_to_read()
        logger.info(
            'AdaBN with {} epochs takes {} seconds',
            self.num_adabn_epoch,
            time.time() - start
        )

    def _try_update_adabn(self, eval_data, reset):
        assert self.binded and self.params_initialized
        if self.adabn:
            self._update_adabn(eval_data)
        if not reset and self.adabn:
            eval_data.reset()

    def score(
        self,
        eval_data,
        eval_metric,
        num_batch=None,
        batch_end_callback=None,
        reset=True,
        epoch=0
    ):
        self._try_update_adabn(eval_data, reset)
        return super(AdaBNMixin, self).score(
            eval_data=eval_data,
            eval_metric=eval_metric,
            num_batch=num_batch,
            batch_end_callback=batch_end_callback,
            reset=reset,
            epoch=epoch
        )

    def predict(
        self,
        eval_data,
        num_batch=None,
        merge_batches=True,
        reset=True,
        always_output_list=False
    ):
        self._try_update_adabn(eval_data, reset)
        return super(AdaBNMixin, self).predict(
            eval_data=eval_data,
            num_batch=num_batch,
            merge_batches=merge_batches,
            reset=reset,
            always_output_list=always_output_list
        )


class EvalMixin(object):
    '''Used to specify num_eval_epoch'''

    def __init__(self, **kargs):
        self.num_eval_epoch = kargs.pop('num_eval_epoch', 1)
        super(EvalMixin, self).__init__(**kargs)

    def fit(self, train_data, eval_data=None, eval_metric='acc',
            epoch_end_callback=None, batch_end_callback=None, kvstore='local',
            optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
            eval_batch_end_callback=None, initializer=mx.init.Uniform(0.01),
            arg_params=None, aux_params=None, allow_missing=False,
            force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
            validation_metric=None, monitor=None):
        assert num_epoch is not None, 'please specify number of epochs'

        self.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label,
                  for_training=True, force_rebind=force_rebind)
        if monitor is not None:
            self.install_monitor(monitor)
        self.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                         allow_missing=allow_missing, force_init=force_init)
        self.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                            optimizer_params=optimizer_params)

        if validation_metric is None:
            validation_metric = eval_metric
        if not isinstance(eval_metric, mx.metric.EvalMetric):
            eval_metric = mx.metric.create(eval_metric)

        ################################################################################
        # training loop
        ################################################################################
        for epoch in range(begin_epoch, num_epoch):
            tic = time.time()
            need_eval = any((epoch + 1) % n == 0 for n in self.num_eval_epoch)
            need_eval_train = True
            if need_eval_train:
                eval_metric.reset()
            for nbatch, data_batch in enumerate(train_data):
                if monitor is not None:
                    monitor.tic()
                self.forward_backward(data_batch)
                self.update()
                if need_eval_train:
                    self.update_metric(eval_metric, data_batch.label)

                if monitor is not None:
                    monitor.toc_print()

                if batch_end_callback is not None and need_eval_train:
                    batch_end_params = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                              eval_metric=eval_metric,
                                                              locals=locals())
                    for callback in mx.module.base_module._as_list(batch_end_callback):
                        callback(batch_end_params)

            # one epoch of training is finished
            if need_eval_train:
                for name, val in eval_metric.get_name_value():
                    self.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

            # sync aux params across devices
            arg_params, aux_params = self.get_params()
            self.set_params(arg_params, aux_params)

            if epoch_end_callback is not None:
                arg_params, aux_params = self.get_params()
                for callback in mx.module.base_module._as_list(epoch_end_callback):
                    callback(epoch, self.symbol, arg_params, aux_params)

            # ----------------------------------------
            # evaluation on validation set
            if eval_data and need_eval:
                res = self.score(eval_data, validation_metric,
                                 batch_end_callback=eval_batch_end_callback, epoch=epoch)
                for name, val in res:
                    self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)

            # end of 1 epoch, reset the data-iter for another epoch
            train_data.reset()


class MxNetModule(mx.mod.Module):
    '''Used to ignore irrelevant kargs'''

    def __init__(self, **kargs):
        super(MxNetModule, self).__init__(
            **{k: kargs[k] for k in kargs
               if k in ['symbol', 'data_names', 'label_names', 'logger',
                        'context', 'work_load_list']})


class BaseModule(EvalMixin, SymbolMixin, MxNetModule):

    @property
    def symbol_props(self):
        return utils.g.get(utils.hash(self.symbol), {})

    @property
    def loss_normalized(self):
        return self.symbol_props.get('loss_normalization', 'null') != 'null'

    @property
    def num_semg_row(self):
        return self.symbol.num_semg_row

    @property
    def num_semg_col(self):
        return self.symbol.num_semg_col

    @property
    def data_shape_1(self):
        return self.symbol.data_shape_1

    def __init__(self, **kargs):
        parent = kargs.pop('parent', {})
        kargs['data_names'] = kargs.get('data_names', getattr(parent, '_data_names', ('data',)))
        kargs['label_names'] = kargs.get('label_names', getattr(parent, '_label_names', ('softmax_label',)))
        super(BaseModule, self).__init__(**kargs)

    def get_loader(self, **kargs):
        return Load(self.params, default_init=self._get_init(), mod=self)

    def init_params(self, initializer=mx.init.Uniform(0.01), arg_params=None, aux_params=None,
                    allow_missing=False, force_init=False):
        if self.params_initialized and not force_init:
            return
        assert self.binded, 'call bind before initializing the parameters'

        if self._arg_params is None:
            param_arrays = [mx.nd.zeros(x[0].shape) for x in self._exec_group.param_arrays]
            self._arg_params = {name: arr for name, arr in zip(self._param_names, param_arrays)}

        if self._aux_params is None:
            aux_arrays = [mx.nd.zeros(x[0].shape) for x in self._exec_group.aux_arrays]
            self._aux_params = {name: arr for name, arr in zip(self._aux_names, aux_arrays)}

        def _impl(name, arr, cache):
            """Internal helper for parameter initialization"""
            if cache is not None:
                if name in cache:
                    cache_arr = cache[name]

                    # just in case the cached array is just the target itself
                    if cache_arr is not arr:
                        assert cache_arr.shape == arr.shape, '{} {} {}'.format(name, cache_arr.shape, arr.shape)
                        cache_arr.copyto(arr)
                else:
                    assert allow_missing, name
                    initializer(name, arr)
            else:
                initializer(name, arr)

        for name, arr in self._arg_params.items():
            _impl(name, arr, arg_params)

        for name, arr in self._aux_params.items():
            _impl(name, arr, aux_params)

        self.params_initialized = True
        self._params_dirty = False

        # copy the initialized parameters to devices
        self._exec_group.set_params(self._arg_params, self._aux_params)

    def init_optimizer(self, kvstore='local', optimizer='sgd',
                       optimizer_params=(('learning_rate', 0.01),), force_init=False):
        """Install and initialize optimizers.

        Parameters
        ----------
        kvstore : str or KVStore
            Default `'local'`.
        optimizer : str or Optimizer
            Default `'sgd'`
        optimizer_params : dict
            Default `(('learning_rate', 0.01),)`. The default value is not a dictionary,
            just to avoid pylint warning of dangerous default values.
        force_init : bool
            Default `False`, indicating whether we should force re-initializing the
            optimizer in the case an optimizer is already installed.
        """
        assert self.binded and self.params_initialized
        opt = mx.optimizer

        if self.optimizer_initialized and not force_init:
            self.logger.warning('optimizer already initialized, ignoring...')
            return

        (kvstore, update_on_kvstore) = mx.model._create_kvstore(kvstore, len(self._context), self._arg_params)

        if isinstance(optimizer, str):
            batch_size = self._exec_group.batch_size
            if kvstore and kvstore.type == 'dist_sync':
                batch_size *= kvstore.num_workers
            idx2name = {}
            # Fix bug in mxnet for multi-gpu training
            if update_on_kvstore:
                idx2name.update(enumerate(self._exec_group.param_names))
            else:
                for k in range(len(self._context)):
                    idx2name.update({i*len(self._context)+k: n
                                    for i, n in enumerate(self._exec_group.param_names)})
            optimizer_params = dict(optimizer_params)
            if 'rescale_grad' not in optimizer_params and not self.loss_normalized:
                optimizer_params['rescale_grad'] = 1.0/batch_size
            optimizer = opt.create(optimizer,
                                   sym=self.symbol, param_idx2name=idx2name,
                                   **optimizer_params)
        else:
            assert isinstance(optimizer, opt.Optimizer)

        self._optimizer = optimizer
        self._kvstore = kvstore
        self._update_on_kvstore = update_on_kvstore
        self._updater = None

        if not update_on_kvstore:
            self._updater = opt.get_updater(optimizer)
        if kvstore:
            # copy initialized local parameters to kvstore
            mx.model._initialize_kvstore(kvstore=kvstore,
                                         param_arrays=self._exec_group.param_arrays,
                                         arg_params=self._arg_params,
                                         param_names=self._param_names,
                                         update_on_kvstore=update_on_kvstore)
        if update_on_kvstore:
            kvstore.set_optimizer(self._optimizer)

        self.optimizer_initialized = True

    def get_eval_metric(self):
        return 'acc'

    def fit(self, **kargs):
        num_epoch = kargs.pop('num_epoch')
        num_train = kargs.pop('num_train')
        batch_size = kargs.pop('batch_size')
        epoch_size = num_train / batch_size
        lr_step = kargs.pop('lr_step')
        lr = kargs.pop('lr', constant.LR)
        wd = kargs.pop('wd', constant.WD)
        snapshot = kargs.pop('snapshot')
        #  for debug
        self._debug_snapshot = snapshot
        params = kargs.pop('params', None)
        ignore_params = kargs.pop('ignore_params', [])
        fix_params = kargs.pop('fix_params', [])
        decay_all = kargs.pop('decay_all', False)
        lr_factor = kargs.pop('lr_factor', ()) or constant.LR_FACTOR

        checkpoint = []
        if snapshot:
            def do_checkpoint(prefix, period=1):
                def _callback(iter_no, sym, arg, aux):
                    #  Always save the first epoch
                    if iter_no == 0 or (iter_no + 1) % period == 0:
                        mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
                period = int(max(1, period))
                return _callback
            checkpoint.append(do_checkpoint(snapshot, period=self.snapshot_period))

        if params:
            logger.info('Load params from {}', params)
            init = Load(
                params,
                default_init=self._get_init(),
                ignore=ignore_params,
                mod=self
            )
        else:
            init = self._get_init()

        monitor_pattern = kargs.pop('monitor_pattern', None)
        monitor_interval = kargs.pop('monitor_interval', 1000)
        if monitor_pattern:
            self.monitor = mx.mon.Monitor(
                monitor_interval,
                lambda d: d if d.size == 1 else mx.nd.norm(d) / np.sqrt(d.size),
                #  pattern='subject_.*fc\d*_(?:output|weight)',
                #  pattern='.*(gesture_last_fc|subject_last_fc|bottleneck)_(backward_data|backward_weight|_scale).*',
                #  pattern='.*_last_fc_backward_data',
                #  pattern='glove.*',
                #  pattern='glove_last_fc_.*',
                #  pattern=r'.*_branch\d*_backward_data',
                pattern=monitor_pattern,
                sort=True
            )
            self.monitor_installed = False

        def nbatch(param):
            nbatch.value = param.nbatch

        batch_end_callback = [mx.callback.Speedometer(batch_size, 50), nbatch]

        #  def debug(params):
            #  data = params.locals['data_batch'].data[0].asnumpy()
            #  label = params.locals['data_batch'].label[0].asnumpy()
            #  import joblib as jb
            #  jb.dump(dict(data=data, label=label), 'dump')
            #  import sys
            #  sys.exit(0)

        #  batch_end_callback.append(debug)

        eval_metric = kargs.pop('eval_metric', None)
        if eval_metric is None:
            eval_metric = self.get_eval_metric()

        return super(BaseModule, self).fit(
            eval_metric=eval_metric,
            optimizer='NAGex',
            optimizer_params=dict(
                learning_rate=lr,
                momentum=0.9,
                wd=wd,
                lr_scheduler=Scheduler(
                    lr_step=lr_step,
                    factor=lr_factor,
                    epoch_size=epoch_size
                ),
                get_nbatch=lambda: getattr(nbatch, 'value', -1),
                clip_gradient=1,
                fix_params=fix_params,
                decay_all=decay_all,
                mod=self
            ),
            initializer=init,
            num_epoch=num_epoch,
            batch_end_callback=batch_end_callback,
            epoch_end_callback=checkpoint,
            **kargs
        )

    @property
    def mc(self):
        return utils.g.get(utils.hash(self.symbol), {}).get('mc', 0)

    def forward(self, data_batch, is_train=None):
        if is_train or (is_train is None and self.for_training) or not self.mc:
            return super(BaseModule, self).forward(data_batch, is_train)
        super(BaseModule, self).forward(data_batch, is_train)
        if self.mc > 1:
            out = [y.copy() for y in self.get_outputs()]
            for i in range(self.mc - 1):
                super(BaseModule, self).forward(data_batch, is_train)
                for x, y in zip(out, self.get_outputs()):
                    x += y
            for y in out:
                y /= self.mc
            for x, y in zip(out, self.get_outputs()):
                x.copyto(y)

    def forward_backward(self, data_batch):
        if hasattr(self, 'monitor'):
            if not self.monitor_installed:
                for exe in self._exec_group.execs:
                    self.monitor.install(exe)
                self.monitor_installed = True
            self.monitor.tic()
        super(BaseModule, self).forward_backward(data_batch)

    def update(self):
        super(BaseModule, self).update()
        if hasattr(self, 'monitor'):
            self.monitor.toc_print()

    def _get_init(self):
        return Init(factor_type='in', magnitude=2, mod=self)


class Init(mx.init.Xavier):

    def __init__(self, *args, **kargs):
        self.mod = kargs.pop('mod')
        super(Init, self).__init__(*args, **kargs)

    def __call__(self, name, arr):
        """Override () function to do Initialization

        Parameters
        ----------
        name : str
            name of corrosponding ndarray

        arr : NDArray
            ndarray to be Initialized
        """
        if not isinstance(name, mx.base.string_types):
            raise TypeError('name must be string')
        if not isinstance(arr, mx.ndarray.NDArray):
            raise TypeError('arr must be NDArray')
        if name.endswith('lambda'):
            self._init_zero(name, arr)
        elif 'im2col' in name and name.endswith('weight'):
            self._init_im2col(name, arr)
        elif 'lc' in name and name.endswith('weight'):
            self._init_lc_weight(name, arr)
        elif 'pll' in name:
            self._init_pll(name, arr)
        elif 'stochastic_transform_loss_W' in name:
            self._init_stochastic_transform_loss_W(name, arr)
        elif 'rnn_init_state' in name:
            self._init_zero(name, arr)
        elif 'rnn_parameters' in name:
            self._init_rnn_parameters(name, arr)
        elif name.endswith('_zero'):
            self._init_zero(name, arr)
        elif name.endswith('_one'):
            self._init_one(name, arr)
        elif name.startswith('sum'):
            self._init_gamma(name, arr)
        elif name.endswith('upsampling'):
            self._init_bilinear(name, arr)
        elif name.endswith('bias'):
            self._init_bias(name, arr)
        elif name.endswith('gamma'):
            self._init_gamma(name, arr)
        elif name.endswith('beta'):
            self._init_beta(name, arr)
        elif name.endswith('weight'):
            self._init_weight(name, arr)
        elif name.endswith("moving_mean"):
            self._init_zero(name, arr)
        elif name.endswith("moving_var"):
            self._init_zero(name, arr)
        elif name.endswith("moving_inv_var"):
            self._init_zero(name, arr)
        elif name.endswith("moving_avg"):
            self._init_zero(name, arr)
        else:
            self._init_default(name, arr)

    def _init_lc_weight(self, name, arr):
        #  if 'foo' in getattr(self.mod, '_debug_snapshot', ''):
            #  logger.debug('new lc weight init: {}', name)
        #  print(arr.asnumpy().ravel()[:3])
        #  print(arr[0].shape)
        for i in range(arr.shape[0]):
            self._init_weight(name, arr[i])
        #  print(arr.asnumpy().ravel()[:3])
        #  assert False
        #  else:
            #  logger.debug('old lc weight init: {}', name)
            #  self._init_weight(name, arr)

    def _init_pll(self, name, arr):
        pll = float(name.split('_')[-1])
        arr[:] = pll

    def _init_rnn_parameters(self, name, arr):
        std = 1 / int(name.split('_')[-1]) ** 0.5
        mx.random.uniform(-std, std, out=arr)

    def _init_stochastic_transform_loss_W(self, _, arr):
        w = np.zeros(arr.shape, np.float32)
        row = 0
        n = w.shape[1]
        for lhs in range(n):
            for rhs in range(lhs):
                w[row, lhs] = 1
                w[row, rhs] = -1
                row += 1
        nt.assert_equal(row, w.shape[0])
        arr[:] = w

    def _init_one(self, _, arr):
        arr[:] = 1

    def _init_proj(self, _, arr):
        '''Initialization of shortcut of kenel (2, 2)'''
        w = np.zeros(arr.shape, np.float32)
        for i in range(w.shape[1]):
            w[i, i, ...] = 0.25
        arr[:] = w

    def _init_im2col(self, _, arr):
        assert_equal(len(arr.shape), 4)
        assert_equal(arr.shape[0], arr.shape[1] * arr.shape[2] * arr.shape[3])
        arr[:] = np.eye(arr.shape[0]).reshape(arr.shape)


@mx.optimizer.register
class NAGex(mx.optimizer.NAG):

    def __init__(self, get_nbatch, fix_params, decay_all, **kargs):
        self.decay_all = decay_all
        self.mod = kargs.pop('mod')
        super(NAGex, self).__init__(**kargs)
        self.get_nbatch = get_nbatch
        self.confuse = 0
        self.fix_params = fix_params

    @property
    def nbatch(self):
        return self.get_nbatch()

    def update(self, index, weight, grad, state):
        """Update the parameters.
        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters
        weight : NDArray
            weight ndarray
        grad : NDArray
            grad ndarray
        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
        """
        assert(isinstance(weight, mx.nd.NDArray))
        assert(isinstance(grad, mx.nd.NDArray))
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            #  if 'debug' in getattr(self.mod, '_debug_snapshot', ''):
                #  scale = mx.nd.maximum(self.clip_gradient, mx.nd.max(mx.nd.abs(grad)))
                #  scale = scale.reshape([1] * len(grad.shape))
                #  grad = mx.nd.broadcast_div(grad, scale)
            #  else:
            grad = mx.nd.clip(grad, -self.clip_gradient, self.clip_gradient)

        if lr:
            if state:
                mom = state
                mom *= self.momentum
                if wd:
                    # L1 reg
                    # grad += wd * mx.nd.sign(weight)
                    grad += wd * weight
                mom += grad
                grad += self.momentum * mom
                weight += -lr * grad
            else:
                assert self.momentum == 0.0
                if wd:
                    # weight += -lr * (grad + wd * mx.nd.sign(weight))
                    weight += -lr * (grad + wd * weight)
                else:
                    weight += -lr * grad

    def set_wd_mult(self, args_wd_mult):
        """Set individual weight decay multipler for parameters.
        By default wd multipler is 0 for all params whose name doesn't
        end with _weight, if param_idx2name is provided.

        Parameters
        ----------
        args_wd_mult : dict of string/int to float
            set the wd multipler for name/index to float.
            setting multipler by index is supported for backward compatibility,
            but we recommend using name and symbol.
        """
        assert self.decay_all

        self.wd_mult = {}
        for n in self.idx2name.values():
            if not (n.endswith('_weight') or
                    n.endswith('_gamma') or
                    n.endswith('_bias') or
                    n.endswith('_beta')) or 'zscore' in n:
                self.wd_mult[n] = 0.0
        if self.sym is not None:
            attr = self.sym.attr_dict()
            for name in self.sym.list_arguments():
                if name in attr and '__wd_mult__' in attr[name]:
                    self.wd_mult[name] = float(attr[name]['__wd_mult__'])
        self.wd_mult.update(args_wd_mult)

    def _get_lr(self, index):
        """get learning rate for index.

        Parameters
        ----------
        index : int
            The index for weight

        Returns
        -------
        lr : float
            learning rate for this index
        """
        lr = 1

        if index not in self.lr_mult and index in self.idx2name:
            index = self.idx2name[index]
        assert isinstance(index, str)

        if self.fix_params:
            for pattern in self.fix_params:
                if re.match(pattern, index):
                    return 0

        return super(NAGex, self)._get_lr(index) if lr else 0


class Load(mx.init.Load):

    def __init__(self, params, *args, **kargs):
        self.ignore = kargs.pop('ignore', [])
        kargs.pop('mod')
        if not os.path.exists(params) and os.path.exists(os.path.join(ROOT, params)):
            params = os.path.join(ROOT, params)
        super(Load, self).__init__(params, *args, **kargs)
        for name in list(self.param):
            for ignore in self.ignore:
                if re.match(ignore, name):
                    logger.info('Ignore param {}', name)
                    del self.param[name]
                    break
            #  if getattr(mod, 'adabn', False):
                #  if name.endswith('moving_mean') or name.endswith('moving_var'):
                    #  del self.param[name]

    def __call__(self, name, arr):
        if name in self.param and ('gamma' in name or 'beta' in name):
            self.param[name] = self.param[name].reshape(arr.shape)
        return super(Load, self).__call__(name, arr)


class Accuracy(mx.metric.EvalMetric):

    def __init__(self, index, name):
        super(Accuracy, self).__init__('accuracy[{}]'.format(name))
        if not isinstance(index, list):
            index = [index]
        self.index = index

    def update(self, labels, preds):
        # mx.metric.check_label_shapes(labels, preds)

        for index in self.index:
            label = labels[index].astype('int32')
            ndim = len(label.shape)
            assert ndim in (1, 2)
            if ndim == 1:
                pred_label = mx.nd.argmax_channel(preds[index]).astype('int32')
            else:
                pred_label = mx.nd.greater(preds[index], 0.5).astype('int32')

            # mx.metric.check_label_shapes(label, pred_label)

            if ndim == 1:
                total = mx.nd.sum(mx.nd.greater_equal(label, 0).astype('float32'))
            else:
                total = pred_label.size

            # mx.metric.check_label_shapes(label, pred_label)

            self.sum_metric += mx.nd.sum(mx.nd.equal(pred_label, label).astype('float32'))
            self.num_inst += total

    def get(self):
        try:
            value = mx.nd.true_divide(self.sum_metric, self.num_inst).asscalar()
        except ZeroDivisionError:
            value = float('nan')
        return (self.name, value)


class RMSE(mx.metric.EvalMetric):
    """Calculate Root Mean Squred Error loss"""

    def __init__(self, index, name):
        super(RMSE, self).__init__('rmse[{}]'.format(name))
        if not isinstance(index, list):
            index = [index]
        self.index = index

    def update(self, labels, preds):
        #  mx.metric.check_label_shapes(labels, preds)

        for index in self.index:
            label = labels[index].asnumpy()
            pred = preds[index].asnumpy()

            if len(label.shape) == 1:
                label = label.reshape(label.shape[0], 1)

            self.sum_metric += np.sqrt(((label - pred)**2.0).mean())
            self.num_inst += 1


class Scheduler(mx.lr_scheduler.LRScheduler):

    def __init__(self, lr_step, factor, epoch_size):
        if not isinstance(lr_step, (tuple, list)):
            lr_step = list(range(lr_step, 1000, lr_step))
        step = [epoch_size * s for s in lr_step]
        super(Scheduler, self).__init__()
        assert isinstance(step, list) and len(step) >= 1
        for i, _step in enumerate(step):
            if i != 0 and step[i] <= step[i-1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if _step < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        self.step = step
        self.cur_step_ind = 0
        self.factor = factor
        self.count = 0

    def __call__(self, num_update):
        """
        Call to schedule current learning rate

        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """

        if self.cur_step_ind <= len(self.step)-1:
            if num_update > self.step[self.cur_step_ind]:
                self.count = self.step[self.cur_step_ind]
                self.cur_step_ind += 1
                self.base_lr *= (
                    self.factor[self.cur_step_ind - 1]
                    if isinstance(self.factor, (tuple, list)) else self.factor
                )
                logging.info("Update[%d]: Change learning rate to %0.5e",
                             num_update, self.base_lr)
        return self.base_lr
