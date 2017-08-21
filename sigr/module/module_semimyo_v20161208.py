from __future__ import division
from logbook import Logger
import mxnet as mx
import numpy as np
from functools import partial
from nose.tools import assert_equal
from .. import constant, utils
from ..data import ProxyDataIter
from .module_common import (
    BaseModule,
    AdaBNMixin,
    Load,
    Scheduler,
    Accuracy,
    TransformDataIterMixin
)


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
            label_names=['gesture_label', 'static', 'dynamic', 'spot'],
            **kargs
        )

    def transform_data_iter(self, data):
        return DataIter(data)

    def _get_init(self):
        return Init(factor_type='in', magnitude=2, mod=self)

    def fit(self, **kargs):
        num_epoch = kargs.pop('num_epoch')
        num_train = kargs.pop('num_train')
        batch_size = kargs.pop('batch_size')
        epoch_size = num_train / batch_size
        lr_step = kargs.pop('lr_step')
        lr = kargs.pop('lr', constant.LR)
        wd = kargs.pop('wd', constant.WD)
        snapshot = kargs.pop('snapshot')
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

        self.monitor = mx.mon.Monitor(
            1000,
            lambda d: mx.nd.norm(d) / np.sqrt(d.size),
            #  pattern='subject_.*fc\d*_(?:output|weight)',
            #  pattern='.*(gesture_last_fc|subject_last_fc|bottleneck)_(backward_data|backward_weight|_scale).*',
            #  pattern='.*_last_fc_backward_data',
            #  pattern='glove.*',
            #  pattern='glove_last_fc_.*',
            pattern='.*_branch_backward_data',
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
            eval_metric = [Accuracy(0, 'gesture'),
                           Accuracy(1, 'static'),
                           Accuracy(2, 'dynamic'),
                           Accuracy(3, 'spot')]

        return super(Module, self).fit(
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
                decay_all=decay_all
            ),
            initializer=init,
            num_epoch=num_epoch,
            batch_end_callback=batch_end_callback,
            epoch_end_callback=checkpoint,
            **kargs
        )

    def forward_backward(self, data_batch):
        if hasattr(self, 'monitor'):
            if not self.monitor_installed:
                for exe in self._exec_group.execs:
                    self.monitor.install(exe)
                self.monitor_installed = True
            self.monitor.tic()
        super(Module, self).forward_backward(data_batch)

    def update(self):
        super(Module, self).update()
        if hasattr(self, 'monitor'):
            self.monitor.toc_print()


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
        elif name.endswith('_zero'):
            self._init_zero(name, arr)
        elif name.endswith('_one'):
            self._init_one(name, arr)
        elif name.startswith('sum'):
            self._init_gamma(name, arr)
        elif 'im2col' in name and name.endswith('weight'):
            self._init_im2col(name, arr)
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


class RuntimeMixin(object):

    def __init__(self, **kargs):
        args = []
        backup = kargs.copy()
        self.params = kargs.pop('params')
        super(RuntimeMixin, self).__init__(**kargs)
        self.args = args
        self.kargs = backup

    def predict(self, eval_data, *args, **kargs):
        if not self.binded:
            self.bind(data_shapes=eval_data.provide_data, for_training=False)
        if not self.params_initialized:
            self.init_params(Load(self.params, default_init=self._get_init(), mod=self))

        eval_data.reset()
        true = eval_data.gesture.copy()
        segment = eval_data.segment.copy()
        eval_data.reset()
        assert np.all(true == eval_data.gesture.copy())
        assert np.all(segment == eval_data.segment.copy())

        out = super(RuntimeMixin, self).predict(eval_data, *args, **kargs).asnumpy()
        assert_equal(out.ndim, 2)
        pred = out.argmax(axis=1)
        assert_equal(true.shape, pred.shape)
        return utils.Bunch(pred=pred, true=true, segment=segment)

    @property
    def Clone(self):
        return partial(type(self), *self.args, **self.kargs)


class RuntimeModule(RuntimeMixin, Module):
    pass


class DataIter(ProxyDataIter):

    @property
    def provide_label(self):
        return [('gesture_label', (self.batch_size,)),
                ('static', (self.batch_size,)),
                ('dynamic', (self.batch_size,)),
                ('spot', (self.batch_size,))]

    def getlabel(self):
        gesture, spot = self.base.getlabel()
        static = (gesture + 1) * spot - 1
        dynamic = (gesture + 1) * (1 - spot) - 1
        return [gesture, static, dynamic, spot]
