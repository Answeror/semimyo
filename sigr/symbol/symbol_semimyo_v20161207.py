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
        num_semg_row,
        num_semg_col,
        num_semg_channel=1,
        shared_net=None,
        num_gesture=None,
        gesture_net=None,
        num_glove=None,
        glove_net=None,
        gesture_loss_weight=None,
        glove_loss_weight=None,
        num_mini_batch=constant.NUM_MINI_BATCH,
        dropout=constant.DROPOUT,
        cudnn_tune='fastest',
        batch_norm_momentum=constant.BATCH_NORM_MOMENTUM,
        batch_norm_use_global_stats=constant.BATCH_NORM_USE_GLOBAL_STATS,
        **kargs
    ):
        if kargs:
            logger.debug('kargs not used in get_symbol:\n{}', pformat(kargs))

        super(Symbol, self).__init__(**kargs)

        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_use_global_stats = batch_norm_use_global_stats
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

        has_gesture_branch = gesture_loss_weight != 0
        has_glove_branch = glove_loss_weight and num_glove

        if has_gesture_branch:
            gesture_branch = self.get_gesture_branch(
                shared_net,
                gesture_net,
                num_gesture=num_gesture,
                gesture_loss_weight=gesture_loss_weight
            )

        if self.for_training and has_glove_branch:
            glove_branch = self.get_glove_branch(
                shared_net,
                glove_net,
                num_glove=num_glove,
                glove_loss_weight=glove_loss_weight
            )
            if has_gesture_branch:
                self.net = mx.symbol.Group([gesture_branch, glove_branch])
            else:
                self.net = glove_branch
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

    def get_gesture_branch(
        self,
        data,
        net,
        num_gesture,
        gesture_loss_weight
    ):
        if net:
            net = self.get_one_line_net(self.context, net, data, prefix='gesture')
        else:
            net = data

        net = self.get_fc(
            name='gesture_last_fc',
            data=net,
            num_hidden=num_gesture,
            no_bias=False
        )
        if self.for_training:
            kargs = {}
            if gesture_loss_weight is not None:
                kargs.update(grad_scale=gesture_loss_weight)
            net = mx.symbol.SoftmaxOutput(
                name='gesture_softmax',
                data=net,
                label=mx.symbol.Variable('gesture'),
                use_ignore=True,
                ignore_label=-1,
                **kargs
            )
        else:
            net = mx.symbol.SoftmaxActivation(name='gesture_softmax', data=net)
        return net

    def get_glove_branch(
        self,
        data,
        net,
        num_glove,
        glove_loss_weight
    ):
        if net:
            net = self.get_one_line_net(self.context, net, data, prefix='glove')
        else:
            net = data

        net = self.get_fc(
            name='glove_last_fc',
            data=net,
            num_hidden=num_glove,
            no_bias=False
        )
        if self.for_training:
            net = mx.sym.LinearRegressionOutput(
                data=net,
                label=mx.symbol.Variable('glove'),
                grad_scale=glove_loss_weight
            )
        return net
