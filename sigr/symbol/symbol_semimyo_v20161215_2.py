from __future__ import division
import mxnet as mx
import numpy as np
from logbook import Logger
from pprint import pformat
from functools import partial
from .. import constant
from ..patch_mxnet import _mxnet_operator_registered
from .base_symbol import BaseSymbol
from .locally_connected import FastLocallyConnectedMixin
from .grid_loss import GridLossMixin


logger = Logger(__name__)


def get_symbol(**kargs):
    return Symbol(**kargs).net


class Symbol(FastLocallyConnectedMixin, GridLossMixin, BaseSymbol):

    def __init__(
        self,
        for_training,
        batch_size,
        lam,
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
        self.lam = lam
        self.cudnn_tune = cudnn_tune
        self.for_training = for_training
        self.num_mini_batch = num_mini_batch
        self.num_semg_row = num_semg_row
        self.num_semg_col = num_semg_col
        self.num_semg_channel = num_semg_channel
        self.dropout = dropout
        self.context = dict(
            pack=self.pack,
            get_grid_fc=self.get_grid_fc,
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
            gesture_branch = self.get_nets(
                shared_net,
                gesture_net,
                num_gesture,
                gesture_loss_weight,
                'gesture'
            )

        if self.for_training:
            if has_pose_branch:
                pose_branch = self.get_nets(
                    shared_net,
                    pose_net,
                    num_pose,
                    pose_loss_weight,
                    'pose'
                )
                if has_gesture_branch:
                    self.net = mx.symbol.Group(
                        [gesture_branch[0], pose_branch[0]] + gesture_branch[1:] + pose_branch[1:])
                else:
                    self.net = mx.symbol.Group(pose_branch)
            else:
                assert has_gesture_branch
                self.net = mx.symbol.Group(gesture_branch)
        else:
            # Because the weight and bias parameters major branch depend on
            # that of the minor branches', we can't discard minor branches
            # here.
            # But we only want the output of the major branch, so we just use
            # minimum operations to suppress the output of the minor branches.
            assert has_gesture_branch
            net = 0.
            for minor in gesture_branch[1:]:
                net = mx.symbol.minimum(net, minor)
            self.net = gesture_branch[0] + net

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

    def get_branch(self, data, text, num_class, name, context):
        if text:
            net = self.get_one_line_net(context, text, data, prefix=name)
        else:
            net = data
        return net

    def get_softmax(self, data, name, label, grad_scale):
        if self.for_training:
            return mx.symbol.SoftmaxOutput(
                name=name,
                data=data,
                label=label,
                grad_scale=grad_scale
            )
        else:
            return mx.symbol.SoftmaxActivation(name=name, data=data)

    def get_nets(self, data, text, num_class, loss_weight, tag):
        net = data
        label = mx.symbol.Variable(tag)
        net = self.get_branch(net, text, num_class, tag, self.context)
        func = partial(self.get_softmax, label=label)
        major = func(data=net[0],
                     name=tag + '_major_softmax',
                     grad_scale=loss_weight)
        minor = [func(data=net[i],
                      name=tag + '_minor%d_softmax' % i,
                      grad_scale=self.lam * loss_weight)
                 for i in range(1, len(net))]
        return [major] + minor

    def pack(self, data):
        net = data
        net = mx.symbol.Concat(*data, dim=0)
        return net, partial(self.unpack, data)

    def unpack(self, orig, data):
        num_sample = [self.infer_shape(net)[0] for net in orig]
        breaks = np.cumsum([0] + num_sample)
        return [mx.symbol.slice_axis(data, axis=0, begin=begin, end=end)
                for begin, end in zip(breaks[:-1], breaks[1:])]
