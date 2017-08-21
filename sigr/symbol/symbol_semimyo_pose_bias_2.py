from __future__ import division
from ..utils import deprecated
deprecated(__name__, 'bug in get_pose_branch')

import mxnet as mx
from logbook import Logger
from pprint import pformat
from .. import constant, _mxnet_operator_registered
from .locally_connected import FastLocallyConnectedMixin
from .base_symbol import BaseSymbol


logger = Logger(__name__)


def get_symbol(**kargs):
    return Symbol(**kargs).net


class Symbol(FastLocallyConnectedMixin, BaseSymbol):

    def __init__(
        self,
        for_training,
        batch_size,
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

        self.batch_size = batch_size
        self.num_pose = num_pose
        self.num_gesture = num_gesture
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

        assert gesture_loss_weight > 0
        assert pose_loss_weight > 0

        pose_branch = self.get_pose_branch(
            self.get_one_line_net(self.context, 'id:pose_branch', shared_net),
            pose_net,
            pose_loss_weight
        )
        gesture_branch = self.get_gesture_branch(
            pose_branch,
            self.get_one_line_net(self.context, 'id:gesture_branch', shared_net),
            gesture_net,
            gesture_loss_weight
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
        data_shape = (self.batch_size,
                      self.num_semg_channel,
                      self.num_semg_row, self.num_semg_col)
        return tuple(int(s) for s in net.infer_shape(semg=data_shape)[1][0])

    def get_branch(self, data, text, num_class, kind, no_bias=False):
        if text:
            net = self.get_one_line_net(self.context, text, data, prefix=kind)
        else:
            net = data

        net = self.get_fc(
            name=kind + '_last_fc',
            data=net,
            num_hidden=num_class,
            no_bias=no_bias
        )
        return net

    def get_gesture_branch(self, pose_net, data, text, gesture_loss_weight):
        net = self.get_branch(data, text, self.num_gesture, 'gesture', no_bias=True)
        #  net = mx.symbol.Concat(*[net for i in range(self.num_pose)], dim=0)
        #  bias = mx.symbol.Variable('gesture_bias', attr={'__shape__': str((1, self.num_gesture))})
        #  net = mx.symbol.broadcast_plus(net, bias)
        net = mx.symbol.Reshape(net, shape=(1, -1, self.num_gesture))
        net = mx.symbol.broadcast_axis(net, axis=0, size=self.num_pose)
        bias = mx.symbol.Variable('gesture_bias', attr={'__shape__': str((self.num_pose, 1, self.num_gesture))})
        net = mx.symbol.broadcast_plus(net, bias)
        net = mx.symbol.Reshape(net, shape=(-1, self.num_gesture))

        gesture = mx.symbol.Variable('gesture')
        if self.for_training:
            pose = pose_net.get_internals()['pose']
        else:
            pose = mx.symbol.Variable('pose')
        gesture = mx.symbol.Concat(*[(gesture + 1) * self.get_label_mask(pose, i) - 1
                                     for i in range(self.num_pose)], dim=0)
        shape = self.infer_shape(net)
        #  mshadow/mshadow/cuda/tensor_gpu-inl.cuh
        bar = 65535
        if shape[0] <= bar:
            net = self.get_softmax(data=net,
                                   name='gesture',
                                   label=gesture,
                                   grad_scale=gesture_loss_weight,
                                   use_ignore=True,
                                   ignore_label=-1)
        else:
            branches = []
            for begin in range(0, shape[0], bar):
                end = min(shape[0], begin + bar)
                branch = self.get_softmax(
                    data=mx.symbol.slice_axis(net, axis=0, begin=begin, end=end),
                    name='gesture',
                    label=mx.symbol.slice_axis(gesture, axis=0, begin=begin, end=end),
                    grad_scale=gesture_loss_weight,
                    use_ignore=True,
                    ignore_label=-1
                )
                branches.append(branch)
            net = mx.symbol.Concat(*branches, dim=0)
        return self.get_gesture_output(net, pose_net)

    def get_pose_branch(self, data, text, pose_loss_weight):
        net = self.get_branch(data, text, self.num_pose, 'pose')
        net = self.get_softmax(data=data,
                               name='pose',
                               label=mx.symbol.Variable('pose'),
                               grad_scale=pose_loss_weight)
        return net

    def get_gesture_output(self, gesture_net, latent_net):
        gesture_net = mx.symbol.Reshape(gesture_net, shape=(self.num_pose, -1, self.num_gesture))
        gesture_net = mx.symbol.transpose(gesture_net, axes=(1, 2, 0))
        W = mx.symbol.Reshape(latent_net, shape=(0, self.num_pose, 1))
        gesture_net = mx.symbol.batch_dot(gesture_net, W)
        gesture_net = mx.symbol.Reshape(gesture_net, shape=(0, self.num_gesture))
        return gesture_net
