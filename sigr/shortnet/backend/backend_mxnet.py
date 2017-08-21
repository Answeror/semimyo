from __future__ import division
import nose.tools as nt
import mxnet as mx
import numpy as np
from logbook import Logger
from functools import partial
from contextlib import contextmanager
from ..context import ctx, BaseContext
from ..utils import get


logger = Logger(__name__)
_mxnet_operator_registered = {}


def _patch(func):
    func()


def _mxnet_operator_register(orig, name):
    ret = _mxnet_operator_registered.get(name)
    if ret is None:
        ret = orig(name)
        _mxnet_operator_registered[name] = ret
    return ret


@_patch
def _patch_mxnet():
    mx.operator.register = partial(_mxnet_operator_register,
                                   mx.operator.register)


class Prop(object):

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls):
        return self.func(obj)


class ParentProp(Prop):

    def __init__(self, name, default=None):
        super(ParentProp, self).__init__(lambda self: self.parent.get(name, default))


class Backend(BaseContext):

    batch_norm_momentum = ParentProp('batch_norm_momentum', 0.9)
    batch_norm_use_global_stats = ParentProp('batch_norm_use_global_stats', False)
    cudnn_tune = ParentProp('cudnn_tune', 'fastest')
    fix_batch_norm = ParentProp('fix_batch_norm', False)
    lc_no_bias = ParentProp('lc_no_bias', True)
    lc_loss_weight = ParentProp('lc_loss_weight', 1)
    lc_loss_type = ParentProp('lc_loss_type', 'l2')
    lc_loss_bias = ParentProp('lc_loss_bias', True)
    lc_share_weight = ParentProp('lc_share_weight', False)
    lc_bottleneck_scale = ParentProp('lc_bottleneck_scale', None)
    lc_bottleneck_sum = ParentProp('lc_bottleneck_sum', True)
    lc_bottleneck_bias = ParentProp('lc_bottleneck_bias', False)
    sd_p = ParentProp('sd_p', 0.5)
    loss = ParentProp('loss', [])
    resnet_bottleneck = ParentProp('resnet_bottleneck', False)
    lsd_bottleneck_mid_conv = ParentProp('lsd_bottleneck_mid_conv', False)
    lc_loss_8 = ParentProp('lc_loss_8', True)
    in_sd = ParentProp('in_sd', False)
    sd_soft = ParentProp('sd_soft', False)
    preact = ParentProp('preact', False)
    sd_dropout = ParentProp('sd_dropout', 0)
    mc = ParentProp('mc', False)

    @property
    def __with_lc_loss(self):
        return 'lc_loss' in ctx and (isinstance(ctx.lc_loss, list) or ctx.lc_loss)

    def get_label_mask(self, label, target):
        return 1 - mx.symbol.abs(mx.symbol.sign(label - target))

    def get_softmax(self, data, name, label, grad_scale, **kargs):
        if ctx.for_training:
            return mx.symbol.SoftmaxOutput(
                name=name,
                data=data,
                label=label,
                grad_scale=grad_scale,
                **kargs
            )
        else:
            return mx.symbol.SoftmaxActivation(name=name, data=data)

    def get_bn(self, name, data):
        net = data
        shape = ctx.infer_shape(net)
        bar = 1024

        if shape[0] <= bar:
            return ctx._get_bn(name, net)

        #  https://github.com/dmlc/mxnet/issues/3184
        shape_orig = shape

        for num_sample in range(bar, 0, -1):
            if shape_orig[0] % num_sample == 0:
                break
        nt.assert_greater(num_sample, 0)
        num_block = shape_orig[0] // num_sample

        if len(shape_orig) == 2:
            net = mx.symbol.Reshape(net, shape=(num_sample, num_block, shape_orig[1]))
            net = mx.symbol.SwapAxis(net, dim1=1, dim2=2)
            net = ctx._get_bn(name, net)
            net = mx.symbol.SwapAxis(net, dim1=1, dim2=2)
        elif len(shape_orig) == 4:
            net = mx.symbol.Reshape(net, shape=(num_sample, num_block, shape_orig[1], shape_orig[2] * shape_orig[3]))
            net = mx.symbol.SwapAxis(net, dim1=1, dim2=2)
            net = ctx._get_bn(name, net)
            net = mx.symbol.SwapAxis(net, dim1=1, dim2=2)
        else:
            assert False

        net = mx.symbol.Reshape(net, shape=shape_orig)
        return net

    def _get_bn(self, name, data):
        #  Get BatchNorm or AdaBN
        attr = {}
        if 'lr_mult' in ctx:
            attr['__lr_mult__'] = str(ctx.lr_mult)
        if 'wd_mult' in ctx:
            attr['__wd_mult__'] = str(ctx.wd_mult)

        bn_kargs = dict(
            momentum=ctx.batch_norm_momentum,
            use_global_stats=ctx.batch_norm_use_global_stats
        )
        if ctx.get('fix_batch_norm', False):
            bn_kargs.update(momentum=1, use_global_stats=True)

        if ctx.get('num_mini_batch', 0) > 1:
            net = data
            shape = ctx.infer_shape(net)
            net = mx.symbol.Reshape(net, shape=(-1, ctx.num_mini_batch * shape[1]) + shape[2:])
            net = mx.symbol.BatchNorm(
                name=ctx.concat_name(name, '_norm'),
                data=net,
                fix_gamma=True,
                attr={'__wd_mult__': '0', '__lr_mult__': '0'},
                **bn_kargs
            )
            net = mx.symbol.Reshape(data=net, shape=(-1,) + shape[1:])
            if len(shape) == 4:
                #  Convolution
                gamma = mx.symbol.Variable(ctx.concat_name(name, '_gamma'), shape=(1, shape[1], 1, 1), attr=attr)
                beta = mx.symbol.Variable(ctx.concat_name(name, '_beta'), shape=(1, shape[1], 1, 1), attr=attr)
            else:
                #  Fully connected
                gamma = mx.symbol.Variable(ctx.concat_name(name, '_gamma'), shape=(1, shape[1]), attr=attr)
                beta = mx.symbol.Variable(ctx.concat_name(name, '_beta'), shape=(1, shape[1]), attr=attr)
            net = mx.symbol.broadcast_mul(net, gamma)
            net = mx.symbol.broadcast_plus(net, beta, name=ctx.concat_name(name, '_last'))
        else:
            net = mx.symbol.BatchNorm(
                name=name,
                data=data,
                fix_gamma=False,
                attr=attr,
                **bn_kargs
            )
        return net

    def get_bn_relu(self, name, data):
        net = ctx.get_bn(ctx.concat_name(name, '_bn'), data)
        return mx.symbol.Activation(name=ctx.concat_name(name, '_relu'), data=net, act_type='relu')

    def get_relu(self, name, data):
        return mx.symbol.Activation(name=name, data=data, act_type='relu')

    def get_bn_prelu(self, name, data):
        net = ctx.get_bn(ctx.concat_name(name, '_bn'), data)
        return mx.symbol.LeakyReLU(name=ctx.concat_name(name, '_prelu'), data=net, act_type='prelu')

    def concat_name(self, name, suffix):
        return name if name is None else name + suffix

    def im2col(self, data, name, kernel, pad=(0, 0), stride=(1, 1)):
        shape = ctx.infer_shape(data)
        return ctx.get_conv(
            name=name,
            data=data,
            num_filter=shape[1] * kernel[0] * kernel[1],
            kernel=kernel,
            stride=stride,
            pad=pad,
            no_bias=True,
            attr={'__lr_mult__': '0'}
        )

    def concat(self, *data, **kargs):
        axis = kargs.pop('axis')
        return mx.sym.Concat(*data, dim=axis)

    def flatten(self, data):
        return mx.sym.Flatten(data)

    def get_avg(self, data, name=None, kernel=(0, 0), stride=(1, 1), pad=(0, 0)):
        return mx.sym.Pooling(data=data,
                              name=name,
                              kernel=kernel,
                              stride=stride,
                              pad=pad,
                              pool_type='avg',
                              global_pool=kernel == (0, 0))

    def get_lc(self, name, data, num_filter, no_bias, kernel=(1, 1), stride=(1, 1), pad=(0, 0)):
        net = data

        if kernel == 1 or kernel == (1, 1):
            assert stride == 1 or stride == (1, 1)
            assert pad == 0 or pad == (0, 0)
            return ctx.get_lc_1x1(net, name, num_filter, no_bias)

        if ctx.lc_bottleneck_scale is not None:
            nt.assert_less_equal(ctx.lc_bottleneck_scale, 1)
            net = ctx.get_conv(data=net,
                               name=name + '_head',
                               num_filter=num_filter,
                               kernel=kernel,
                               stride=stride,
                               pad=pad,
                               no_bias=True)
            if ctx.lc_bottleneck_bias:
                if ctx.lc_bottleneck_scale:
                    tail = ctx.lc_bottleneck_scale * ctx.get_lc_1x1_bias(net, name + '_tail', num_filter)
                    net = (1 - ctx.lc_bottleneck_scale) * net + tail
            else:
                tail = ctx.lc_bottleneck_scale * ctx.get_lc_1x1(net, name + '_tail', num_filter, no_bias)
                if ctx.lc_bottleneck_sum:
                    if ctx.lc_bottleneck_scale:
                        net = (1 - ctx.lc_bottleneck_scale) * net + tail
                else:
                    net = tail
        else:
            net = ctx.im2col(name=name + '_im2col',
                             data=net,
                             kernel=kernel,
                             pad=pad,
                             stride=stride)
            net = ctx.get_lc_1x1(net, name, num_filter, no_bias)

        return net

    def get_lc_1x1_bias(self, data, name, num_filter):
        assert 'lc_loss' not in ctx

        attr = {}
        if 'lr_mult' in ctx:
            attr['__lr_mult__'] = str(ctx.lr_mult)
        if 'wd_mult' in ctx:
            attr['__wd_mult__'] = str(ctx.wd_mult)

        net = data
        _, channels, rows, cols = ctx.infer_shape(net)

        b_attr = attr.copy()
        b_attr.update({'__shape__': str((1, num_filter, rows, cols))})
        b = mx.sym.Variable(name=name + '_bias', attr=b_attr)
        net = mx.sym.broadcast_plus(net, b)
        return net

    def get_lc_1x1(self, data, name, num_filter, no_bias):
        attr = {}
        if 'lr_mult' in ctx:
            attr['__lr_mult__'] = str(ctx.lr_mult)
        if 'wd_mult' in ctx:
            attr['__wd_mult__'] = str(ctx.wd_mult)

        net = data
        _, channels, rows, cols = ctx.infer_shape(net)
        net = mx.symbol.Reshape(net, shape=(0, 0, -1))
        net = mx.symbol.SwapAxis(net, dim1=0, dim2=2)
        W_attr = attr.copy()
        W_attr.update({'__shape__': str((rows * cols, num_filter, channels))})
        W = mx.symbol.Variable(name=name + '_weight', attr=W_attr)
        net = mx.symbol.batch_dot(W, net)

        if not no_bias:
            b_attr = attr.copy()
            b_attr.update({'__shape__': str((rows * cols, num_filter))})
            b = mx.sym.Variable(name=name + '_bias', attr=b_attr)
            net = mx.sym.broadcast_plus(net, mx.sym.expand_dims(b, axis=2))

        if self.__with_lc_loss:
            self.__push_lc_loss_fast(name, W, b, rows, cols, no_bias)

        net = mx.symbol.SwapAxis(net, dim1=0, dim2=2)
        net = mx.symbol.Reshape(net, shape=(0, 0, rows, cols))
        return net

    def __push_lc_loss(self, name, W, b, rows, cols, no_bias):
        nt.assert_in(ctx.lc_loss_type, ['l1', 'l2'])
        get_norm = {'l1': mx.sym.abs, 'l2': mx.sym.square}[ctx.lc_loss_type]
        get_dist = lambda lhs, rhs: get_norm(lhs - rhs)
        get_sum = partial(reduce, lambda x, y: x + y)
        get_grid = lambda data: mx.sym.Reshape(data, shape=(rows, cols, -1))
        loss = get_sum(self.__gen_lc_loss(get_grid(W), get_dist))
        if not no_bias and ctx.lc_loss_bias:
            loss += get_sum(self.__gen_lc_loss(get_grid(b), get_dist))
        loss = ctx.get_loss(data=loss,
                            name=name + '_loss',
                            grad_scale=ctx.lc_loss_weight)
        ctx.loss.append(loss)

    def __gen_lc_loss(self, data, get_dist):
        rows, cols, _ = data.infer_shape()[1][0]
        s = mx.sym.slice_axis
        s2 = lambda data, b0, e0, b1, e1: s(s(data, axis=0, begin=b0, end=e0),
                                            axis=1, begin=b1, end=e1)
        if rows > 1:
            lhs = s(data, axis=0, begin=1, end=rows)
            rhs = s(data, axis=0, begin=0, end=rows - 1)
            yield mx.sym.sum(get_dist(lhs, rhs))
        if cols > 1:
            lhs = s(data, axis=1, begin=1, end=cols)
            rhs = s(data, axis=1, begin=0, end=cols - 1)
            yield mx.sym.sum(get_dist(lhs, rhs))
        if rows and cols and ctx.lc_loss_8:
            lhs = s2(data, b0=1, e0=rows, b1=1, e1=cols)
            rhs = s2(data, b0=0, e0=rows - 1, b1=0, e1=cols - 1)
            yield mx.sym.sum(get_dist(lhs, rhs))
            lhs = s2(data, b0=1, e0=rows, b1=0, e1=cols - 1)
            rhs = s2(data, b0=0, e0=rows - 1, b1=1, e1=cols)
            yield mx.sym.sum(get_dist(lhs, rhs))

    def __get_lc_loss_dist_func(self):
        get_norm = {'l1': mx.sym.abs, 'l2': mx.sym.square}[ctx.lc_loss_type]
        get_dist = lambda lhs, rhs: get_norm(lhs - rhs)
        return get_dist

    def __push_lc_loss_fast(self, name, W, b, rows, cols, no_bias):
        nt.assert_in(ctx.lc_loss_type, ['l1', 'l2'])
        get_sum = list
        get_grid = lambda data: mx.sym.Reshape(data, shape=(rows, cols, -1))
        loss = get_sum(self.__gen_lc_loss_fast(get_grid(W)))
        if not no_bias and ctx.lc_loss_bias:
            loss += get_sum(self.__gen_lc_loss_fast(get_grid(b)))

        if ctx.in_sd:
            ctx.loss.extend(loss)
        else:
            lhs, rhs = zip(*loss)
            cat = lambda data: mx.sym.Concat(*[mx.sym.Reshape(x, shape=(-1,)) for x in data], dim=0)
            lhs = cat(lhs)
            rhs = cat(rhs)
            get_dist = self.__get_lc_loss_dist_func()
            loss = mx.sym.sum(get_dist(lhs, rhs))
            loss = ctx.get_loss(data=loss,
                                name=name + '_loss',
                                grad_scale=ctx.lc_loss_weight)
            ctx.loss.append(loss)

    def __gen_lc_loss_fast(self, data):
        rows, cols, _ = data.infer_shape()[1][0]
        s = mx.sym.slice_axis
        s2 = lambda data, b0, e0, b1, e1: s(s(data, axis=0, begin=b0, end=e0),
                                            axis=1, begin=b1, end=e1)
        if rows > 1:
            lhs = s(data, axis=0, begin=1, end=rows)
            rhs = s(data, axis=0, begin=0, end=rows - 1)
            yield lhs, rhs
        if cols > 1:
            lhs = s(data, axis=1, begin=1, end=cols)
            rhs = s(data, axis=1, begin=0, end=cols - 1)
            yield lhs, rhs
        if rows and cols and ctx.lc_loss_8:
            lhs = s2(data, b0=1, e0=rows, b1=1, e1=cols)
            rhs = s2(data, b0=0, e0=rows - 1, b1=0, e1=cols - 1)
            yield lhs, rhs
            lhs = s2(data, b0=1, e0=rows, b1=0, e1=cols - 1)
            rhs = s2(data, b0=0, e0=rows - 1, b1=1, e1=cols)
            yield lhs, rhs

    @contextmanager
    def __fuse_lc_loss(self, name):
        grad_scale = ctx.lc_loss_weight
        if not self.__with_lc_loss or grad_scale == 0:
            yield
            return

        loss = []
        with ctx.push(in_sd=True, loss=loss):
            yield

        lhs, rhs = zip(*loss)
        cat = lambda data: mx.sym.Concat(*[mx.sym.Reshape(x, shape=(-1,)) for x in data], dim=0)
        lhs = cat(lhs)
        rhs = cat(rhs)
        get_dist = self.__get_lc_loss_dist_func()
        loss = mx.sym.sum(get_dist(lhs, rhs))
        loss = ctx.get_loss(data=loss,
                            name=name + '_loss',
                            grad_scale=grad_scale)
        ctx.loss.append(loss)

    def get_loss(self, data, name=None, grad_scale=1):
        if isinstance(grad_scale, mx.sym.Symbol):
            if name:
                logger.debug('Grad scale of {}: Symbol', name)
            loss = mx.sym.MakeLoss(data * grad_scale)
        else:
            if name:
                logger.debug('Grad scale of {}: {}', name, grad_scale)
            loss = mx.sym.MakeLoss(data, grad_scale=grad_scale)
        return loss

    def get_fc_bn_relu(self, name, data, num_hidden):
        net = ctx.get_fc(name=name, data=data, num_hidden=num_hidden, no_bias=True)
        net = ctx.get_bn_relu(name, net)
        return net

    def get_fsd(self, data, name, num_block, num_hidden, p, dropout):
        P = p
        net = data
        for i in range(num_block):
            net = ctx.get_fsd_block(
                data=net,
                name=name + '_block%d' % i,
                num_hidden=num_hidden,
                #  p=1 - i / (num_block - 1) * (1 - P)
                p=P,
                dropout=dropout
            )
        return net

    def get_fsd_block(self, data, name, num_hidden, p, dropout):
        net = data
        shortcut = net
        gate = ctx.get_gate(p=p, shape=(ctx.infer_shape(net)[0], 1), name=name + '_gate')
        net = ctx.get_fc(data=net, name=name + '_fc1', num_hidden=num_hidden, no_bias=True)
        net = ctx.get_bn(data=net, name=name + '_fc1_bn')
        net = ctx.get_relu(data=net, name=name + '_fc1_relu')
        net = ctx.get_fc(data=net, name=name + '_fc2', num_hidden=num_hidden, no_bias=True)
        net = ctx.get_bn(data=net, name=name + '_fc2_bn')
        if dropout:
            net = ctx.get_dropout(data=net, name=name + '_drop', p=dropout)
        net = shortcut + mx.sym.broadcast_mul(gate, net)
        net = ctx.get_relu(data=net, name=name + '_relu')
        return net

    def get_lsd(self, data, name, num_block, num_filter, kernel, pad, p, decay, dropout, scale, no_bias):
        P = p
        net = data
        for i in range(num_block):
            block_name = name + '_block%d' % i
            with self.__fuse_lc_loss(block_name):
                net = ctx.get_lsd_block(
                    data=net,
                    name=block_name,
                    num_filter=num_filter,
                    kernel=kernel,
                    pad=pad,
                    p=1 - i / (num_block - 1) * (1 - P) if decay else P,
                    dropout=dropout,
                    scale=scale,
                    no_bias=no_bias
                )
        return net

    def get_lsd_block(self, data, name, num_filter, kernel, pad, p, dropout, scale, no_bias):
        logger.debug('sd_p of {}: {}', name, p)

        net = data
        shortcut = net
        get_gate = ctx.get_gate if not ctx.sd_soft else lambda p, **kargs: ctx.get_uniform(high=p, **kargs)
        gate = get_gate(p=p, shape=(ctx.infer_shape(net)[0], 1, 1, 1), name=name + '_gate')

        if not ctx.resnet_bottleneck:
            if ctx.preact:
                net = ctx.get_bn(data=net, name=name + '_lc1_bn')
                net = ctx.get_relu(data=net, name=name + '_lc1_relu')
                net = ctx.get_lc(data=net, name=name + '_lc1', num_filter=num_filter, no_bias=no_bias, kernel=kernel, pad=pad)
                net = ctx.get_bn(data=net, name=name + '_lc2_bn')
                net = ctx.get_relu(data=net, name=name + '_lc2_relu')
                net = ctx.get_lc(data=net, name=name + '_lc2', num_filter=num_filter, no_bias=no_bias, kernel=kernel, pad=pad)
            else:
                net = ctx.get_lc(data=net, name=name + '_lc1', num_filter=num_filter, no_bias=no_bias, kernel=kernel, pad=pad)
                net = ctx.get_bn(data=net, name=name + '_lc1_bn')
                net = ctx.get_relu(data=net, name=name + '_lc1_relu')
                net = ctx.get_lc(data=net, name=name + '_lc2', num_filter=num_filter, no_bias=no_bias, kernel=kernel, pad=pad)
                net = ctx.get_bn(data=net, name=name + '_lc2_bn')
        else:
            if ctx.preact:
                net = ctx.get_bn(data=net, name=name + '_lc1_bn')
                net = ctx.get_relu(data=net, name=name + '_lc1_relu')
                net = ctx.get_lc(data=net, name=name + '_lc1', num_filter=num_filter // 4, no_bias=no_bias, kernel=(1, 1), pad=(0, 0))
                net = ctx.get_bn(data=net, name=name + '_lc2_bn')
                net = ctx.get_relu(data=net, name=name + '_lc2_relu')
                get_mid = ctx.get_conv if ctx.lsd_bottleneck_mid_conv else ctx.get_lc
                net = get_mid(data=net, name=name + '_lc2', num_filter=num_filter // 4, no_bias=no_bias, kernel=kernel, pad=pad)
                net = ctx.get_bn(data=net, name=name + '_lc3_bn')
                net = ctx.get_relu(data=net, name=name + '_lc3_relu')
                net = ctx.get_lc(data=net, name=name + '_lc3', num_filter=num_filter, no_bias=no_bias, kernel=(1, 1), pad=(0, 0))
            else:
                net = ctx.get_lc(data=net, name=name + '_lc1', num_filter=num_filter // 4, no_bias=no_bias, kernel=(1, 1), pad=(0, 0))
                net = ctx.get_bn(data=net, name=name + '_lc1_bn')
                net = ctx.get_relu(data=net, name=name + '_lc1_relu')
                get_mid = ctx.get_conv if ctx.lsd_bottleneck_mid_conv else ctx.get_lc
                net = get_mid(data=net, name=name + '_lc2', num_filter=num_filter // 4, no_bias=no_bias, kernel=kernel, pad=pad)
                net = ctx.get_bn(data=net, name=name + '_lc2_bn')
                net = ctx.get_relu(data=net, name=name + '_lc2_relu')
                net = ctx.get_lc(data=net, name=name + '_lc3', num_filter=num_filter, no_bias=no_bias, kernel=(1, 1), pad=(0, 0))
                net = ctx.get_bn(data=net, name=name + '_lc3_bn')

        if dropout or ctx.sd_dropout:
            net = ctx.get_dropout(data=net, name=name + '_drop', p=dropout or ctx.sd_dropout)
        if scale is not None and scale != 1:
            net *= scale
        net = shortcut + mx.sym.broadcast_mul(gate, net)

        if not ctx.preact:
            net = ctx.get_relu(data=net, name=name + '_relu')
        return net

    def get_sd(self, data, name, num_block, num_filter, p):
        P = p
        net = data
        for i in range(num_block):
            net = ctx.get_sd_block(
                data=net,
                name=name + '_block%d' % i,
                num_filter=num_filter,
                #  p=1 - i / (num_block - 1) * (1 - P)
                p=P
            )
        return net

    def get_sd_block(self, data, name, num_filter, p, kernel=(3, 3), stride=(1, 1), pad=(1, 1)):
        net = data
        shortcut = net
        gate = ctx.get_gate(p=p, shape=(ctx.infer_shape(net)[0], 1, 1, 1), name=name + '_gate')
        net = ctx.get_conv(data=net, name=name + '_conv1', num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True)
        net = ctx.get_bn(data=net, name=name + '_conv1_bn')
        net = ctx.get_relu(data=net, name=name + '_conv1_relu')
        net = ctx.get_conv(data=net, name=name + '_conv2', num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True)
        net = ctx.get_bn(data=net, name=name + '_conv2_bn')
        net = shortcut + mx.sym.broadcast_mul(gate, net)
        net = ctx.get_relu(data=net, name=name + '_relu')
        return net

    def get_gate(self, p, shape, name=None):
        return mx.sym.sign(
            name=name,
            data=mx.sym.sign(mx.sym.uniform(shape=shape) - (1 - p)) + 1
        )

    def get_uniform(self, shape, high=1, name=None):
        return mx.sym.uniform(high=high, shape=shape)

    def reshape(self, data, shape):
        return mx.sym.Reshape(data, shape=shape)

    def get_fc(self, name, data, num_hidden, no_bias, **kargs):
        attr = kargs.get('attr')
        if attr is None:
            attr = kargs['attr'] = {}

        if 'lr_mult' in ctx:
            attr['__lr_mult__'] = str(ctx.lr_mult)
        if 'wd_mult' in ctx:
            attr['__wd_mult__'] = str(ctx.wd_mult)
        if 'weight' not in kargs and 'fc_weight' in ctx:
            fc_weight = ctx.fc_weight
            weight_name = name + '_weight'
            if weight_name in fc_weight:
                logger.debug('get_weight: ' + weight_name)
                kargs['weight'] = fc_weight[weight_name]
            else:
                logger.debug('set_weight: ' + weight_name)
                shape = ctx.infer_shape(data)
                kargs['weight'] = fc_weight[weight_name] = \
                    mx.symbol.Variable(
                        weight_name,
                        attr={'__shape__': str((num_hidden, np.prod(shape[1:])))}
                    )
        if 'bias' not in kargs and not no_bias and 'fc_bias' in ctx:
            fc_bias = ctx.fc_bias
            bias_name = name + '_bias'
            if bias_name in fc_bias:
                logger.debug('get_bias: ' + bias_name)
                kargs['bias'] = fc_bias[bias_name]
            else:
                logger.debug('set_bias: ' + bias_name)
                kargs['bias'] = fc_bias[bias_name] = \
                    mx.symbol.Variable(
                        bias_name,
                        attr={'__shape__': str((num_hidden,))}
                    )

        return mx.symbol.FullyConnected(
            name=name,
            data=data,
            num_hidden=num_hidden,
            no_bias=no_bias,
            **kargs
        )

    def get_conv(self, *args, **kargs):
        attr = {}
        attr.update(kargs.pop('attr', {}))
        if 'lr_mult' in ctx:
            attr['__lr_mult__'] = str(ctx.lr_mult)
        if 'wd_mult' in ctx:
            attr['__wd_mult__'] = str(ctx.wd_mult)
        kargs['attr'] = attr
        kargs['cudnn_tune'] = ctx.cudnn_tune
        return mx.symbol.Convolution(*args, **kargs)

    def get_dropout(self, data, name, **kargs):
        p = get(kargs, 'p')(ctx, 'dropout')()
        return mx.symbol.Dropout(data=data, name=name, p=p, mc=bool(ctx.mc))

    def get_block_grad(self, data):
        return mx.symbol.BlockGrad(data=data)

    def get_scale_grad(self, data, scale):
        opname = self._try_register_scale_grad(scale)
        return mx.symbol.Custom(data=data, op_type=opname)

    def get_clip_grad(self, data, clip):
        opname = self._try_register_clip_grad(clip)
        return mx.symbol.Custom(data=data, op_type=opname)

    def get_identity(self, data, name):
        #  opname = self._try_register_identity()
        #  return mx.symbol.Custom(data=data, name=name, op_type=opname)
        return mx.sym.identity(data=data, name=name)

    def _try_register_scale_grad(self, scale):
        opname = 'ScaleGrad({})'.format(scale)

        class ScaleGrad(mx.operator.CustomOp):

            def forward(self, is_train, req, in_data, out_data, aux):
                self.assign(out_data[0], req[0], 0 + in_data[0])

            def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
                self.assign(
                    in_grad[0],
                    req[0],
                    scale * out_grad[0]
                )

        @mx.operator.register(opname)
        class ScaleGradProp(mx.operator.CustomOpProp):

            def __init__(self):
                super(ScaleGradProp, self).__init__(need_top_grad=True)

            def infer_shape(self, in_shape):
                data_shape = in_shape[0]
                output_shape = in_shape[0]
                return [data_shape], [output_shape], []

            def create_operator(self, ctx, shapes, dtypes):
                return ScaleGrad()

        return opname

    def _try_register_clip_grad(self, clip):
        opname = 'ClipGrad({})'.format(clip)

        class ClipGrad(mx.operator.CustomOp):

            def forward(self, is_train, req, in_data, out_data, aux):
                self.assign(out_data[0], req[0], 0 + in_data[0])

            def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
                self.assign(
                    in_grad[0],
                    req[0],
                    mx.nd.maximum(-clip, mx.nd.minimum(clip, out_grad[0]))
                )

        @mx.operator.register(opname)
        class ClipGradProp(mx.operator.CustomOpProp):

            def __init__(self):
                super(ClipGradProp, self).__init__(need_top_grad=True)

            def infer_shape(self, in_shape):
                data_shape = in_shape[0]
                output_shape = in_shape[0]
                return [data_shape], [output_shape], []

            def create_operator(self, ctx, shapes, dtypes):
                return ClipGrad()

        return opname

    def _try_register_identity(self):
        class Identity(mx.operator.CustomOp):

            def forward(self, is_train, req, in_data, out_data, aux):
                self.assign(out_data[0], req[0], 0 + in_data[0])

            def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
                self.assign(in_grad[0], req[0], 0 + out_grad[0])

        @mx.operator.register('Identity')
        class IdentityProp(mx.operator.CustomOpProp):

            def __init__(self):
                super(IdentityProp, self).__init__(need_top_grad=True)

            def infer_shape(self, in_shape):
                data_shape = in_shape[0]
                output_shape = in_shape[0]
                return [data_shape], [output_shape], []

            def create_operator(self, ctx, shapes, dtypes):
                return Identity()

        return 'Identity'
