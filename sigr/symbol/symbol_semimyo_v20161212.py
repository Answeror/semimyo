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
        num_latent,
        num_semg_row,
        num_semg_col,
        num_semg_channel=1,
        shared_net=None,
        num_gesture=None,
        gesture_net=None,
        latent_net=None,
        gesture_loss_weight=None,
        latent_loss_weight=None,
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
        gesture_branches = self.get_gesture_branches(
            self.get_one_line_net(self.context, 'id:gesture_branch', shared_net),
            gesture_net, num_gesture, num_latent)
        latent_branch = self.get_latent_branch(
            self.get_one_line_net(self.context, 'id:latent_branch', shared_net),
            latent_net, num_latent)
        self.net = self.fuse(
            gesture_branches,
            latent_branch,
            gesture_loss_weight=gesture_loss_weight,
            latent_loss_weight=latent_loss_weight,
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

    def get_gesture_branches(self, data, net, num_gesture, num_latent):
        return [self.get_branch(data, net, num_gesture, 'gesture%d' % i)
                for i in range(num_latent)]

    def get_latent_branch(self, data, net, num_latent):
        return self.get_branch(data, net, num_latent, 'latent')

    def fuse(
        self,
        gesture_nets,
        latent_net,
        gesture_loss_weight,
        latent_loss_weight,
    ):
        if self.for_training:
            latent = mx.symbol.Variable('latent')
            latent_net = mx.symbol.SoftmaxOutput(
                name='latent_softmax',
                data=latent_net,
                label=latent,
                grad_scale=latent_loss_weight
            )
        else:
            latent_net = mx.symbol.SoftmaxActivation(
                name='latent_softmax',
                data=latent_net
            )

        get_mask = lambda label, target: 1 - mx.symbol.abs(mx.symbol.sign(label - target))

        if self.for_training:
            gesture = mx.symbol.Variable('gesture')
            gesture_nets = [mx.symbol.SoftmaxOutput(
                            name='gesture%d_softmax' % i,
                            data=net,
                            label=(gesture + 1) * get_mask(latent, i) - 1,
                            grad_scale=gesture_loss_weight,
                            use_ignore=True,
                            ignore_label=-1) for i, net in enumerate(gesture_nets)]
        else:
            gesture_nets = [mx.symbol.SoftmaxActivation(
                            name='gesture%d_softmax' % i,
                            data=net) for i, net in enumerate(gesture_nets)]

        gesture_net = self.get_gesture_output(gesture_nets, latent_net)

        if not self.for_training:
            return gesture_net
        else:
            return mx.symbol.Group([gesture_net, latent_net])

    def get_gesture_output(self, gesture_nets, latent_net):
        gesture_nets = [mx.symbol.Reshape(net, shape=(0, -1, 1)) for net in gesture_nets]
        gesture_net = mx.symbol.Concat(*gesture_nets, dim=2)
        W = mx.symbol.Reshape(latent_net, shape=(0, len(gesture_nets), 1))
        gesture_net = mx.symbol.batch_dot(gesture_net, W)
        gesture_net = mx.symbol.Reshape(gesture_net, shape=(0, -1))
        return gesture_net
