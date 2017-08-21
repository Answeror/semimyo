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

        self.num_pose = num_pose
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
            num_pose,
            pose_loss_weight
        )
        gesture_branch = self.get_gesture_branch(
            pose_branch,
            self.get_one_line_net(self.context, 'id:gesture_branch', shared_net),
            gesture_net,
            num_gesture,
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
        data_shape = (self.num_mini_batch,
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

    def get_gesture_branch(self, pose_net, data, text, num_gesture, gesture_loss_weight):
        net = self.get_branch(data, text, num_gesture, 'gesture', no_bias=True)
        branches = [net + mx.symbol.Variable('gesture%d_last_fc_bias' % i)
                    for i in range(self.num_pose)]
        gesture = mx.symbol.Variable('gesture')
        pose = pose_net.get_internals()['pose']
        branches = [self.get_softmax(data=branches[i],
                                     name='gesture%d' % i,
                                     label=(gesture + 1) * self.get_label_mask(pose, i) - 1,
                                     grad_scale=gesture_loss_weight,
                                     use_ignore=True,
                                     ignore_label=-1)
                    for i in range(self.num_pose)]
        return self.get_gesture_output(branches, pose_net)

    def get_pose_branch(self, data, text, num_pose, pose_loss_weight):
        net = self.get_branch(data, text, num_pose, 'pose')
        net = self.get_softmax(data=data,
                               name='pose',
                               label=mx.symbol.Variable('pose'),
                               grad_scale=pose_loss_weight)
        return net

    def get_gesture_output(self, gesture_nets, latent_net):
        gesture_nets = [mx.symbol.Reshape(net, shape=(0, -1, 1)) for net in gesture_nets]
        gesture_net = mx.symbol.Concat(*gesture_nets, dim=2)
        W = mx.symbol.Reshape(latent_net, shape=(0, len(gesture_nets), 1))
        gesture_net = mx.symbol.batch_dot(gesture_net, W)
        gesture_net = mx.symbol.Reshape(gesture_net, shape=(0, -1))
        return gesture_net
