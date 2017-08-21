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
        static_net=None,
        dynamic_net=None,
        spot_net=None,
        static_loss_weight=None,
        dynamic_loss_weight=None,
        spot_loss_weight=None,
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
        static_branch = self.get_static_branch(shared_net, static_net, num_gesture)
        dynamic_branch = self.get_dynamic_branch(shared_net, dynamic_net, num_gesture)
        spot_branch = self.get_spot_branch(shared_net, spot_net)
        self.net = self.fuse(
            static_branch,
            dynamic_branch,
            spot_branch,
            static_loss_weight=static_loss_weight,
            dynamic_loss_weight=dynamic_loss_weight,
            spot_loss_weight=spot_loss_weight,
        )
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

    def get_static_branch(self, data, net, num_gesture):
        return self.get_branch(data, net, num_gesture, 'static')

    def get_dynamic_branch(self, data, net, num_gesture):
        return self.get_branch(data, net, num_gesture, 'dynamic')

    def get_spot_branch(self, data, net):
        return self.get_branch(data, net, 2, 'spot')

    def fuse(
        self,
        static_net,
        dynamic_net,
        spot_net,
        static_loss_weight,
        dynamic_loss_weight,
        spot_loss_weight,
    ):
        spot_label = mx.symbol.Variable('spot')
        spot_net = mx.symbol.SoftmaxOutput(
            name='spot_softmax',
            data=spot_net,
            label=spot_label,
            grad_scale=spot_loss_weight
        )

        static_net = mx.symbol.SoftmaxOutput(
            name='static_softmax',
            data=static_net,
            label=mx.symbol.Variable('static'),
            grad_scale=static_loss_weight,
            use_ignore=True,
            ignore_label=-1
        )

        dynamic_net = mx.symbol.SoftmaxOutput(
            name='dynamic_softmax',
            data=dynamic_net,
            label=mx.symbol.Variable('dynamic'),
            grad_scale=dynamic_loss_weight,
            use_ignore=True,
            ignore_label=-1
        )

        gesture_net = self.get_gesture_output(static_net, dynamic_net, spot_net)

        if not self.for_training:
            return gesture_net
        else:
            return mx.symbol.Group([gesture_net,
                                    static_net,
                                    dynamic_net,
                                    spot_net])

    def get_gesture_output(self, static_net, dynamic_net, spot_net):
        static_net = mx.symbol.Reshape(static_net, shape=(0, -1, 1))
        dynamic_net = mx.symbol.Reshape(dynamic_net, shape=(0, -1, 1))
        gesture_net = mx.symbol.Concat(dynamic_net, static_net, dim=2)
        W = mx.symbol.Reshape(spot_net, shape=(0, 2, 1))
        gesture_net = mx.symbol.batch_dot(gesture_net, W)
        gesture_net = mx.symbol.Reshape(gesture_net, shape=(0, -1))
        gesture_net = mx.symbol.Custom(data=gesture_net,
                                       name='gesture',
                                       op_type='DummyOutput')
        return gesture_net


class DummyOutput(mx.operator.CustomOp):

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], 0 + in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0 + out_grad[0])


@mx.operator.register('DummyOutput')
class DummyOutputProp(mx.operator.CustomOpProp):

    def __init__(self):
        super(DummyOutputProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return DummyOutput()
