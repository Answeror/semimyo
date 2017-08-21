from __future__ import division
import mxnet as mx
from .. import utils
utils.deprecated(__name__)


class FastLocallyConnectedMixin(object):

    def get_lc_1x1(self, data, name, num_filter, no_bias, context=None):
        attr = {}
        if context and 'lr_mult' in context:
            attr['lr_mult'] = str(context['lr_mult'])
        if context and 'wd_mult' in context:
            attr['wd_mult'] = str(context['wd_mult'])

        net = data
        _, channels, rows, cols = self.infer_shape(net)
        net = mx.symbol.Reshape(net, shape=(0, 0, -1))
        net = mx.symbol.SwapAxis(net, dim1=0, dim2=2)
        W_attr = {'__shape__': str((rows * cols, num_filter, channels))}
        W = mx.symbol.Variable(name=name + '_weight', attr=W_attr)
        net = mx.symbol.batch_dot(W, net)
        assert no_bias
        net = mx.symbol.SwapAxis(net, dim1=0, dim2=2)
        net = mx.symbol.Reshape(net, shape=(0, 0, rows, cols))
        return net


class FastLocallyConnectedMixin2(object):

    def get_lc_1x1(self, data, name, num_filter, no_bias, context=None):
        attr = {}
        if context and 'lr_mult' in context:
            attr['lr_mult'] = str(context['lr_mult'])
        if context and 'wd_mult' in context:
            attr['wd_mult'] = str(context['wd_mult'])

        net = data
        _, channels, rows, cols = self.infer_shape(net)
        net = mx.symbol.Reshape(net, shape=(0, 0, -1))
        net = mx.symbol.SwapAxis(net, dim1=0, dim2=2)
        W_attr = attr.copy()
        W_attr.update({'__shape__': str((rows * cols, num_filter, channels))})
        W = mx.symbol.Variable(name=name + '_weight', attr=W_attr)
        net = mx.symbol.batch_dot(W, net)
        assert no_bias
        net = mx.symbol.SwapAxis(net, dim1=0, dim2=2)
        net = mx.symbol.Reshape(net, shape=(0, 0, rows, cols))
        return net
