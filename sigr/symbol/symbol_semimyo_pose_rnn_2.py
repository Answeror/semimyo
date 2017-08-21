from __future__ import division
import mxnet as mx
from logbook import Logger
from pprint import pformat
import nose.tools as nt
from .base_symbol import BaseSymbol


logger = Logger(__name__)


def get_symbol(**kargs):
    return Symbol(**kargs).net


class Symbol(BaseSymbol):

    def get_shortnet(self, text, data, **kargs):
        return super(Symbol, self).get_shortnet(
            text,
            data,
            dropout=self.dropout,
            infer_shape=self.infer_shape,
            act='bn_relu',
            **kargs
        )

    def __init__(
        self,
        shared_net=None,
        gesture_net=None,
        gesture_loss_weight=None,
        pose_loss_weight=None,
        **kargs
    ):
        if kargs:
            logger.debug('kargs not used in get_symbol:\n{}', pformat(kargs))

        super(Symbol, self).__init__(**kargs)

        data = mx.symbol.Variable('semg')
        shared_net = self.get_shortnet(shared_net, data)

        has_gesture_branch = gesture_loss_weight > 0
        assert has_gesture_branch
        has_pose_branch = pose_loss_weight > 0

        gesture_branch = self.get_gesture_branch(
            self.get_shortnet('id:gesture_branch', shared_net),
            gesture_net,
            gesture_loss_weight
        )

        if self.for_training and has_pose_branch:
            pose_branch = self.get_pose_branch(
                self.get_shortnet('id:pose_branch', shared_net),
                pose_loss_weight
            )
            self.net = mx.symbol.Group([gesture_branch, pose_branch])
        else:
            self.net = gesture_branch

        self.net.num_semg_row = self.num_semg_row
        self.net.num_semg_col = self.num_semg_col
        self.net.num_semg_channel = self.num_semg_channel
        self.net.data_shape_1 = self.num_semg_channel

    def infer_shape(self, data):
        net = data
        data_shape = (self.batch_size * self.window,
                      self.num_semg_channel,
                      self.num_semg_row, self.num_semg_col)
        return tuple(int(s) for s in net.infer_shape(semg=data_shape)[1][0])

    def get_gesture_branch(self, data, net, gesture_loss_weight):
        net = self.get_shortnet(net, data, prefix='gesture')
        net = self.get_softmax(
            data=net,
            name='gesture_softmax',
            label=mx.symbol.Variable('gesture'),
            grad_scale=gesture_loss_weight
        )
        return net

    def get_pose_branch(self, data, pose_loss_weight):
        net = data
        net = self.get_shortnet(self.pose_head_net, net, prefix='pose_head')
        net = mx.symbol.Reshape(
            net,
            shape=(self.batch_size, self.window, -1)
        )
        #  net = mx.symbol.SliceChannel(net, num_outputs=self.window, axis=1)
        #  net = [self.get_shortnet('id:pose_branch%d' % i, net[i]) for i in range(self.window)]
        #  net = mx.symbol.Concat(*net, dim=1)
        net = mx.symbol.SwapAxis(net, dim1=0, dim2=1)
        net = mx.symbol.RNN(
            data=net,
            name='pose_rnn',
            parameters=mx.symbol.Variable('rnn_parameters_%d' % self.num_rnn_hidden),
            state=mx.symbol.Variable('rnn_init_state'),
            mode='gru',
            state_outputs=False,
            num_layers=self.num_rnn_layer,
            state_size=self.num_rnn_hidden,
            p=0,
            bidirectional=False
        )
        shape = self.infer_shape(net)
        nt.assert_equal(len(shape), 3)
        nt.assert_equal(shape[0], self.window)
        nt.assert_equal(shape[1], self.batch_size)
        net = mx.symbol.SwapAxis(net, dim1=0, dim2=1)
        net = mx.symbol.Reshape(
            net,
            shape=(self.batch_size * self.window, self.num_rnn_hidden)
        )
        net = self.get_shortnet(self.pose_tail_net, net, prefix='pose_tail')
        net = self.get_softmax(
            data=net,
            name='pose_softmax',
            label=mx.symbol.Variable('pose'),
            grad_scale=pose_loss_weight
        )
        return net
