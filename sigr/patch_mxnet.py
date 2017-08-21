from __future__ import division
import mxnet as mx
from functools import partial
from types import MethodType


_mxnet_operator_registered = {}


def _mxnet_operator_register(orig, name):
    ret = _mxnet_operator_registered.get(name)
    if ret is None:
        ret = orig(name)
        _mxnet_operator_registered[name] = ret
    return ret


def _mxnet_symbol_infer_type(orig, self, *args, **kargs):
    names = self.list_arguments()
    for name in names:
        t = self.get_internals()[name].attr('__type__')
        if t and t not in kargs:
            if t == 'float32':
                t = mx.symbol.mx_real_t
            else:
                assert 'Unsupported type: ' + t
            kargs[name] = t
    for name in list(kargs):
        if name not in names:
            del kargs[name]
    return orig(self, *args, **kargs)


def _mxnet_symbol_infer_shape(orig, self, *args, **kargs):
    names = self.list_arguments()
    for name in list(kargs):
        if name not in names:
            del kargs[name]
    return orig(self, *args, **kargs)


def _mxnet_executor_group_bind_exec(orig, self, data_shapes, label_shapes, shared_group):
    if data_shapes:
        data_shapes = [(name, _) for name, _ in data_shapes if name in self.arg_names]
    if label_shapes:
        label_shapes = [(name, _) for name, _ in label_shapes if name in self.arg_names]
    return orig(self, data_shapes, label_shapes, shared_group)


def patch_mxnet():
    mx.operator.register = partial(_mxnet_operator_register,
                                   mx.operator.register)
    mx.symbol.Symbol.infer_type = MethodType(
        partial(_mxnet_symbol_infer_type, mx.symbol.Symbol.infer_type),
        None,
        mx.symbol.Symbol,
    )
    mx.symbol.Symbol.infer_shape = MethodType(
        partial(_mxnet_symbol_infer_shape, mx.symbol.Symbol.infer_shape),
        None,
        mx.symbol.Symbol,
    )
    mx.module.executor_group.DataParallelExecutorGroup.bind_exec = MethodType(
        partial(_mxnet_executor_group_bind_exec,
                mx.module.executor_group.DataParallelExecutorGroup.bind_exec),
        None,
        mx.module.executor_group.DataParallelExecutorGroup,
    )
