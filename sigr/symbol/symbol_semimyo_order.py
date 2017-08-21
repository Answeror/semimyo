from __future__ import division
import mxnet as mx
from logbook import Logger
from pprint import pformat
from .base_symbol import BaseSymbol
from .. import utils


logger = Logger(__name__)


def get_symbol(**kargs):
    return Symbol(**kargs).net


class Symbol(BaseSymbol):

    def get_shortnet(self, text, data, **_kargs):
        kargs = self.shortnet_args.copy()
        kargs.update(**_kargs)
        return super(Symbol, self).get_shortnet(text, data, **kargs)

    def __init__(self, num_semg_channel=1, **kargs):
        if kargs:
            logger.debug('kargs not used in get_symbol:\n{}', pformat(kargs))

        super(Symbol, self).__init__(**kargs)
        self.num_semg_channel = num_semg_channel
        self.shortnet_args.setdefault('act', 'bn_relu')
        self.shortnet_args.setdefault('dropout', 0.5)

        semg = mx.symbol.Variable('semg')
        prev = mx.symbol.Variable('prev')
        data = mx.sym.Concat(semg, prev, dim=0)
        shared_net = self.get_shortnet(self.shared_net, data)
        shared_net = mx.sym.Flatten(shared_net)

        has_gesture_branch = self.gesture_loss_weight > 0
        assert has_gesture_branch
        has_order_branch = self.order_loss_weight > 0

        shared_nets = mx.sym.SliceChannel(shared_net, axis=0, num_outputs=2)
        shared_net_semg = shared_nets[0]
        shared_net_prev = shared_nets[1]

        gesture_branch = self.get_gesture_branch(self.get_shortnet('id:gesture_branch', shared_net_semg))

        if self.for_training:
            loss = [gesture_branch]
            if has_order_branch:
                loss.extend(self.get_order_branch(self.get_shortnet('id:semg_branch', shared_net_semg),
                                                  self.get_shortnet('id:prev_branch', shared_net_prev)))
            self.net = mx.symbol.Group(loss)
        else:
            self.net = gesture_branch

        #  self.net.num_semg_row = self.num_semg_row
        #  self.net.num_semg_col = self.num_semg_col
        #  self.net.num_semg_channel = num_semg_channel
        #  self.net.data_shape_1 = num_semg_channel

        utils.g_set(self.net, kargs.copy())

    def infer_shape(self, data):
        net = data
        shape = dict(
            semg=(self.batch_size,
                  self.num_semg_channel,
                  self.num_semg_row, self.num_semg_col),
            prev=(self.batch_size,
                  self.num_semg_channel,
                  self.num_semg_row, self.num_semg_col),
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
        #  net = mx.sym.broadcast_axis(net, axis=1, size=2)
        net = mx.sym.Concat(net, net, dim=1)
        net = mx.sym.Flatten(net)
        return net

    def get_argmax(self, net):
        net = mx.sym.SliceChannel(net, axis=1, num_outputs=2, squeeze_axis=True)
        return mx.sym.sign(mx.sym.sign(net[1] - net[0]) + 1)

    def get_order_branch(self, semg, prev):
        with self.push_shortnet_context(**self.shortnet_args) as ctx:
            order = ctx.get_gate(p=0.5, shape=(self.batch_size,))
        gate = mx.sym.expand_dims(order, axis=1)
        gate = mx.sym.broadcast_axis(gate, axis=1, size=self.infer_shape(semg)[1])
        gate = mx.sym.Concat(gate, 1 - gate, dim=1)
        net = self.get_double(semg) * gate + self.get_double(prev) * (1 - gate)
        net = self.get_shortnet(self.order_net, net, prefix='order')
        loss = 1 - mx.sym.abs(self.get_argmax(net) - order)
        loss = mx.sym.MakeLoss(loss, grad_scale=0)
        net = self.get_softmax(
            data=net,
            name='order_softmax',
            label=order,
            grad_scale=self.order_loss_weight
        )
        return net, loss

        #  net = self.get_shortnet(self.diff_net, data, prefix='diff')
        #  diff = mx.sym.Variable('diff')
        #  diff = self.get_shortnet(
            #  getattr(self, 'diff_label_net', None),
            #  diff,
            #  prefix='diff_label'
        #  )
        #  net = mx.sym.sum(mx.sym.square(net - diff), axis=1) * (1 / self.num_diff)
        #  net = mx.sym.MakeLoss(net, grad_scale=self.diff_loss_weight)
        #  return net
