from __future__ import division
import mxnet as mx
from logbook import Logger
from pprint import pformat
from .. import constant, _mxnet_operator_registered
from .base_symbol import BaseSymbol
from .locally_connected import FastLocallyConnectedMixin2


logger = Logger(__name__)


def get_symbol(**kargs):
    return Symbol(**kargs).net


class Symbol(FastLocallyConnectedMixin2, BaseSymbol):

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
        bias_net=None,
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

        pose_branch = self.get_pose_branch(
            self.get_one_line_net(self.context, 'id:pose_branch', shared_net),
            pose_net,
            num_pose,
            pose_loss_weight
        )
        gesture_branch = self.get_gesture_branch(
            self.get_one_line_net(self.context, 'id:gesture_branch', shared_net),
            gesture_net,
            num_gesture,
            gesture_loss_weight,
            #  pose_branch
            pose_branch.get_internals()['pose_fc1_relu_output'],
            bias_net
        )

        if self.for_training:
            self.net = mx.symbol.Group([gesture_branch, pose_branch])
        else:
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

    def get_branch(self, data, net, num_class, kind, bias=None):
        if net:
            net = self.get_one_line_net(self.context, net, data, prefix=kind)
        else:
            net = data

        net = self.get_fc(
            name=kind + '_last_fc',
            data=net,
            num_hidden=num_class,
            no_bias=bias is not None
        )
        if bias:
            net += bias
        return net

    def get_gesture_branch(self, data, text, num_gesture, gesture_loss_weight, pose_net, bias_net):
        bias = self.get_one_line_net(
            self.context,
            bias_net,
            pose_net,
            prefix='pose_conditioned_gesture_bias'
        )
        net = data
        net = self.get_one_line_net(self.context, text, net, prefix='gesture')
        net += bias
        net = self.get_softmax(data=net,
                               name='gesture_softmax',
                               label=mx.symbol.Variable('gesture'),
                               grad_scale=gesture_loss_weight)
        return net

    def get_pose_branch(self, data, text, num_pose, pose_loss_weight):
        net = data
        net = self.get_one_line_net(self.context, text, net, prefix='pose')
        net = self.get_softmax(data=net,
                               name='pose_softmax',
                               label=mx.symbol.Variable('pose'),
                               grad_scale=pose_loss_weight)
        return net
