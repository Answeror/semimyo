from __future__ import division
import mxnet as mx
from logbook import Logger
from pprint import pformat
from .base_symbol import BaseSymbol
from ..context import ctx
from .. import utils


logger = Logger(__name__)


def get_symbol(**kargs):
    return Symbol(**kargs).net


class Symbol(BaseSymbol):

    def get_shortnet(self, text, data, **_kargs):
        #  logger.debug('shortnet_args:\n{}', pformat(self.shortnet_args))
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

        loss = []
        with ctx.push(loss=loss):
            data = mx.symbol.Variable('semg')
            loss.insert(0, self.get_gesture_branch(data))
        self.net = loss[0] if len(loss) == 1 else mx.sym.Group(loss)
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
                  self.num_semg_row, self.num_semg_col)
        )
        return tuple(int(s) for s in net.infer_shape(**shape)[1][0])

    def get_gesture_branch(self, data):
        net = data
        if hasattr(self, 'shortnet'):
            net = self.get_shortnet(self.shortnet, net)
        else:
            net = self.get_shortnet(self.shared_net, net)
            net = self.get_shortnet(self.gesture_net, net, prefix='gesture')
        net = self.get_softmax(
            data=net,
            name='softmax',
            label=mx.symbol.Variable('gesture')
        )
        return net
