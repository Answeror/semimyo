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

    def __init__(
        self,
        for_training,
        num_semg_row,
        num_semg_col,
        num_semg_channel=1,
        shared_net=None,
        num_gesture=None,
        gesture_net=None,
        diff_net=None,
        gesture_loss_weight=None,
        diff_loss_weight=None,
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

        data = mx.symbol.Variable('semg')
        shared_net = self.get_shortnet(shared_net, data)

        has_gesture_branch = gesture_loss_weight > 0
        has_diff_branch = diff_loss_weight > 0

        if has_gesture_branch:
            gesture_branch = self.get_gesture_branch(
                self.get_shortnet('id:gesture_branch', shared_net),
                gesture_net,
                num_gesture,
                gesture_loss_weight
            )

        if self.for_training and has_diff_branch:
            diff_branch = self.get_diff_branch(
                self.get_shortnet('id:diff_branch', shared_net),
                diff_net,
                diff_loss_weight
            )
            if has_gesture_branch:
                self.net = mx.symbol.Group([gesture_branch, diff_branch])
            else:
                self.net = diff_branch
        else:
            assert has_gesture_branch
            self.net = gesture_branch

        self.net.num_semg_row = num_semg_row
        self.net.num_semg_col = num_semg_col
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

    def get_gesture_branch(self, data, net, num_gesture, gesture_loss_weight):
        net = self.get_shortnet(net, data, prefix='gesture')
        net = self.get_softmax(
            data=net,
            name='gesture_softmax',
            label=mx.symbol.Variable('gesture'),
            grad_scale=gesture_loss_weight,
            use_ignore=True,
            ignore_label=-1
        )
        return net

    def get_diff_branch(self, data, net, diff_loss_weight):
        net = self.get_shortnet(net, data, prefix='diff')
        diff = mx.sym.Variable('diff')
        diff = self.get_shortnet(
            getattr(self, 'diff_label_net', None),
            diff,
            prefix='diff_label'
        )
        net = mx.sym.sum(mx.sym.square(net - diff), axis=1) * (1 / self.num_diff)
        net = mx.sym.MakeLoss(net, grad_scale=diff_loss_weight)
        return net
