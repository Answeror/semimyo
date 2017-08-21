from __future__ import division
import mxnet as mx
from logbook import Logger
from pprint import pformat
from .. import constant, _mxnet_operator_registered
from .base_symbol import BaseSymbol
from .grid_loss import GridLossMixin


logger = Logger(__name__)


def get_symbol(**kargs):
    return Symbol(**kargs).net


class Symbol(GridLossMixin, BaseSymbol):

    def __init__(
        self,
        for_training,
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

        self.lam = lam
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
            gesture_branch = self.get_nets(
                self.get_one_line_net(self.context, 'id:gesture_branch', shared_net),
                gesture_net,
                num_gesture,
                gesture_loss_weight,
                'gesture'
            )

        if self.for_training:
            if has_pose_branch:
                pose_branch = self.get_nets(
                    self.get_one_line_net(self.context, 'id:pose_branch', shared_net),
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
        data_shape = (self.num_mini_batch,
                      self.num_semg_channel,
                      self.num_semg_row, self.num_semg_col)
        return tuple(int(s) for s in net.infer_shape(semg=data_shape)[1][0])

    def get_branch(self, data, net, num_class, name, context):
        if net:
            net = self.get_one_line_net(context, net, data, prefix=name)
        else:
            net = data

        net = self.get_fc(
            name=name + '_last_fc',
            data=net,
            num_hidden=num_class,
            no_bias=False,
            context=context
        )
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

    def get_loss(self, data, name, text, num_class, label, grad_scale, context):
        net = data
        net = self.get_branch(net, text, num_class, name, context)
        net = self.get_softmax(net, name + '_softmax', label, grad_scale)
        return net

    def get_nets(self, data, text, num_class, loss_weight, tag):
        label = mx.symbol.Variable(tag)
        return self.get_grid_loss(
            data=data,
            name=tag,
            lam=self.lam,
            context=self.context,
            get_loss=lambda grad_scale=1, **kargs: self.get_loss(
                text=text,
                num_class=num_class,
                label=label,
                grad_scale=grad_scale * loss_weight,
                **kargs
            )
        )
