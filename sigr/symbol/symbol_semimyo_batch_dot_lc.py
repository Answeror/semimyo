from __future__ import division
import mxnet as mx
from logbook import Logger
from pprint import pformat
from .. import constant, _mxnet_operator_registered
from . import symbol_base


logger = Logger(__name__)


def get_symbol(**kargs):
    return Symbol(**kargs).net


class Symbol(symbol_base.Symbol):

    def __init__(
        self,
        for_training,
        num_pose,
        num_semg_row,
        num_semg_col,
        num_semg_channel=1,
        shared_net=None,
        num_gesture=None,
        gesture_net=None,
        pose_net=None,
        gesture_loss_weight=None,
        pose_loss_weight=None,
        num_mini_batch=constant.NUM_MINI_BATCH,
        dropout=constant.DROPOUT,
        cudnn_tune='fastest',
        **kargs
    ):
        if kargs:
            logger.debug('kargs not used in get_symbol:\n{}', pformat(kargs))

        super(Symbol, self).__init__(**kargs)

        self.cudnn_tune = cudnn_tune
        self.for_training = for_training
        self.num_mini_batch = num_mini_batch
        self.num_semg_row = num_semg_row
        self.num_semg_col = num_semg_col
        self.num_semg_channel = num_semg_channel
        self.dropout = dropout
        self.context = dict(
            dropout=self.dropout,
            get_act=self.get_bn_relu,
            get_bn=self.get_bn,
            get_fc=self.get_fc,
            get_lc=self.get_lc,
            get_conv=self.get_conv,
            operator_registered=lambda name: name in _mxnet_operator_registered
        )

        data = mx.symbol.Variable('semg')
        shared_net = self.get_one_line_net(self.context, shared_net, data)

        has_gesture_branch = gesture_loss_weight > 0
        has_pose_branch = pose_loss_weight > 0

        if has_gesture_branch:
            gesture_branch = self.get_gesture_branch(
                self.get_one_line_net(self.context, 'id:gesture_branch', shared_net),
                gesture_net,
                num_gesture,
                gesture_loss_weight
            )

        if self.for_training and has_pose_branch:
            pose_branch = self.get_pose_branch(
                self.get_one_line_net(self.context, 'id:pose_branch', shared_net),
                pose_net,
                num_pose,
                pose_loss_weight
            )
            if has_gesture_branch:
                self.net = mx.symbol.Group([gesture_branch, pose_branch])
            else:
                self.net = pose_branch
        else:
            assert has_gesture_branch
            self.net = gesture_branch

        self.net.num_semg_row = num_semg_row
        self.net.num_semg_col = num_semg_col
        self.net.num_semg_channel = num_semg_channel
        self.net.data_shape_1 = num_semg_channel

    def infer_shape(self, data):
        net = data
        data_shape = (self.num_mini_batch,
                      self.num_semg_channel,
                      self.num_semg_row, self.num_semg_col)
        return tuple(int(s) for s in net.infer_shape(semg=data_shape)[1][0])

    def get_branch(self, data, net, num_class, kind):
        if net:
            net = self.get_one_line_net(self.context, net, data, prefix=kind)
        else:
            net = data

        net = self.get_fc(
            name=kind + '_last_fc',
            data=net,
            num_hidden=num_class,
            no_bias=False
        )
        return net

    def get_gesture_branch(self, data, net, num_gesture, gesture_loss_weight):
        net = self.get_branch(data, net, num_gesture, 'gesture')
        if self.for_training:
            net = mx.symbol.SoftmaxOutput(
                name='gesture_softmax',
                data=net,
                label=mx.symbol.Variable('gesture'),
                grad_scale=gesture_loss_weight
            )
        else:
            net = mx.symbol.SoftmaxActivation(name='gesture_softmax', data=net)
        return net

    def get_pose_branch(self, data, net, num_pose, pose_loss_weight):
        net = self.get_branch(data, net, num_pose, 'pose')
        if self.for_training:
            net = mx.symbol.SoftmaxOutput(
                name='pose_softmax',
                data=net,
                label=mx.symbol.Variable('pose'),
                grad_scale=pose_loss_weight
            )
        else:
            net = mx.symbol.SoftmaxActivation(name='pose_softmax', data=net)
        return net

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
