from __future__ import division
import mxnet as mx
from logbook import Logger
from pprint import pformat
from .base_symbol import BaseSymbol


logger = Logger(__name__)


def get_symbol(**kargs):
    return Symbol(**kargs).net


class Symbol(BaseSymbol):

    def get_shortnet(self, text, data, **_kargs):
        kargs = self.shortnet_args.copy()
        kargs.update(**_kargs)
        return super(Symbol, self).get_shortnet(text, data, **kargs)

    def __init__(self, **kargs):
        if kargs:
            logger.debug('kargs not used in get_symbol:\n{}', pformat(kargs))

        kargs.setdefault('act', 'bn_relu')
        kargs.setdefault('dropout', 0.5)
        kargs.setdefault('num_semg_channel', 1)
        super(Symbol, self).__init__(**kargs)

        data = mx.symbol.Variable('semg')
        shared_net = self.get_shortnet(self.shared_net, data)

        gesture_branch = self.get_gesture_branch(self.get_shortnet('id:gesture_branch', shared_net))

        if self.for_training:
            loss = [gesture_branch]
            loss.append(self.get_orient_branch(self.get_shortnet('id:orient_branch', shared_net)))
            self.net = mx.symbol.Group(loss)
        else:
            self.net = gesture_branch

        self.net.num_semg_row = self.num_semg_row
        self.net.num_semg_col = self.num_semg_col
        self.net.num_semg_channel = self.num_semg_channel
        self.net.data_shape_1 = self.num_semg_channel

    def infer_shape(self, data):
        net = data
        shape = dict(
            semg=(self.batch_size,
                  self.num_semg_channel,
                  self.num_semg_row, self.num_semg_col),
            orient_mask=(self.batch_size, self.num_semg_row * self.num_semg_col, 2),
            orient=(self.batch_size,),
        )
        return tuple(int(s) for s in net.infer_shape(**shape)[1][0])

    def get_gesture_branch(self, data):
        net = self.get_shortnet(self.gesture_net, data, prefix='gesture')
        net = self.get_softmax(
            data=net,
            name='gesture_softmax',
            label=mx.symbol.Variable('gesture'),
            grad_scale=self.gesture_loss_weight,
            use_ignore=True,
            ignore_label=-1
        )
        return net

    def get_double(self, net):
        net = mx.sym.expand_dims(net, axis=1)
        net = mx.sym.broadcast_axis(net, axis=1, size=2)
        net = mx.sym.Flatten(net)
        return net

    def get_orient(self, net):
        return mx.sym.sign(mx.sym.sign(net) + 1)

    def get_orient_branch(self, data):
        net = data
        orient = mx.sym.Variable('orient')
        orient_mask = mx.sym.Variable('orient_mask')
        net = mx.sym.Reshape(net, shape=(0, 0, -1))
        net = mx.sym.batch_dot(net, orient_mask)
        net = self.get_shortnet(self.orient_net, net, prefix='orient')
        net = self.get_softmax(
            data=net,
            name='orient_softmax',
            label=orient,
            grad_scale=self.orient_loss_weight
        )
        return net
