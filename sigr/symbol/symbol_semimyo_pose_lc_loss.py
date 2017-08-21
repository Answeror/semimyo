from __future__ import division
import mxnet as mx
from logbook import Logger
from pprint import pformat
from .. import constant
from .base_symbol import BaseSymbol


logger = Logger(__name__)


def get_symbol(**kargs):
    return Symbol(**kargs).net


class Symbol(BaseSymbol):

    def get_shortnet(self, text, data, **_kargs):
        kargs = self.shortnet_args.copy()
        kargs.update(**_kargs)
        return super(Symbol, self).get_shortnet(
            text,
            data,
            dropout=self.dropout,
            infer_shape=self.infer_shape,
            act='bn_relu',
            lc_loss=self.lc_loss,
            lc_loss_weight=self.lc_loss_weight,
            **kargs
        )

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

        self.lc_loss = []
        self.cudnn_tune = cudnn_tune
        self.for_training = for_training
        self.num_mini_batch = num_mini_batch
        self.num_semg_row = num_semg_row
        self.num_semg_col = num_semg_col
        self.num_semg_channel = num_semg_channel
        self.dropout = dropout

        data = mx.symbol.Variable('semg')
        shared_net = self.get_shortnet(shared_net, data)

        has_gesture_branch = gesture_loss_weight > 0
        assert has_gesture_branch
        has_pose_branch = pose_loss_weight > 0

        gesture_branch = self.get_gesture_branch(
            self.get_shortnet('id:gesture_branch', shared_net),
            gesture_net,
            num_gesture,
            gesture_loss_weight
        )

        if self.for_training:
            loss = [gesture_branch]
            if has_pose_branch:
                pose_branch = self.get_pose_branch(
                    self.get_shortnet('id:pose_branch', shared_net),
                    pose_net,
                    num_pose,
                    pose_loss_weight
                )
                loss.append(pose_branch)
            loss.extend(self.lc_loss)
            self.net = mx.symbol.Group(loss)
        else:
            assert has_gesture_branch
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

    def get_gesture_branch(self, data, net, num_gesture, gesture_loss_weight):
        net = self.get_shortnet(net, data, prefix='gesture')
        net = self.get_softmax(
            data=net,
            name='gesture_softmax',
            label=mx.symbol.Variable('gesture'),
            grad_scale=gesture_loss_weight
        )
        return net

    def get_pose_branch(self, data, net, num_pose, pose_loss_weight):
        net = self.get_shortnet(net, data, prefix='pose')
        net = self.get_softmax(
            data=net,
            name='pose_softmax',
            label=mx.symbol.Variable('pose'),
            grad_scale=pose_loss_weight
        )
        return net
