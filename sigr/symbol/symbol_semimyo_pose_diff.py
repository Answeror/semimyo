from __future__ import division
import mxnet as mx
from logbook import Logger
from pprint import pformat
from .base_symbol import BaseSymbol


logger = Logger(__name__)


def get_symbol(**kargs):
    return Symbol(**kargs).net


class Symbol(BaseSymbol):

    @property
    def num_diff(self):
        return self.num_semg_channel * self.num_semg_row * self.num_semg_col

    def get_shortnet(self, text, data, **_kargs):
        kargs = self.shortnet_args.copy()
        kargs.update(**_kargs)
        return super(Symbol, self).get_shortnet(
            text,
            data,
            dropout=self.dropout,
            infer_shape=self.infer_shape,
            act='bn_relu',
            **kargs
        )

    def __init__(self, num_semg_channel=1, **kargs):
        if kargs:
            logger.debug('kargs not used in get_symbol:\n{}', pformat(kargs))

        super(Symbol, self).__init__(**kargs)
        self.num_semg_channel = num_semg_channel

        data = mx.symbol.Variable('semg')
        shared_net = self.get_shortnet(self.shared_net, data)

        has_gesture_branch = self.gesture_loss_weight > 0
        assert has_gesture_branch
        has_pose_branch = self.pose_loss_weight > 0
        has_diff_branch = self.diff_loss_weight > 0

        gesture_branch = self.get_gesture_branch(self.get_shortnet('id:gesture_branch', shared_net))

        if self.for_training:
            loss = [gesture_branch]
            if has_pose_branch:
                loss.append(self.get_pose_branch(self.get_shortnet('id:pose_branch', shared_net)))
            if has_diff_branch:
                loss.append(self.get_diff_branch(self.get_shortnet('id:diff_branch', shared_net)))
            self.net = mx.symbol.Group(loss)
        else:
            self.net = gesture_branch

        self.net.num_semg_row = self.num_semg_row
        self.net.num_semg_col = self.num_semg_col
        self.net.num_semg_channel = num_semg_channel
        self.net.data_shape_1 = num_semg_channel

    def infer_shape(self, data):
        net = data
        shape = dict(
            semg=(self.batch_size,
                  self.num_semg_channel,
                  self.num_semg_row, self.num_semg_col),
            diff=(self.batch_size, self.num_diff)
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

    def get_pose_branch(self, data):
        net = self.get_shortnet(self.pose_net, data, prefix='pose')
        net = self.get_softmax(
            data=net,
            name='pose_softmax',
            label=mx.symbol.Variable('pose'),
            grad_scale=self.pose_loss_weight
        )
        return net

    def get_diff_branch(self, data):
        net = self.get_shortnet(self.diff_net, data, prefix='diff')
        diff = mx.sym.Variable('diff')
        diff = self.get_shortnet(
            getattr(self, 'diff_label_net', None),
            diff,
            prefix='diff_label'
        )
        net = mx.sym.sum(mx.sym.square(net - diff), axis=1) * (1 / self.num_diff)
        net = mx.sym.MakeLoss(net, grad_scale=self.diff_loss_weight)
        return net
