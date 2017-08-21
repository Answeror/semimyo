from __future__ import division
import mxnet as mx
from logbook import Logger
from pprint import pformat
from symbol_common import parse
from .. import constant


logger = Logger(__name__)


def get_symbol(**kargs):
    return Symbol(**kargs).net


class Symbol(object):

    def __init__(
        self,
        for_training,
        num_gesture,
        num_semg_row,
        num_semg_col,
        num_semg_channel=1,
        glove_net=None,
        num_glove=None,
        num_gesture_layer=None,
        num_gesture_hidden=None,
        gesture_loss_weight=None,
        glove_loss_weight=None,
        num_glove_layer=None,
        num_glove_hidden=None,
        num_mini_batch=constant.NUM_MINI_BATCH,
        num_conv_layer=constant.NUM_CONV_LAYER,
        num_conv_filter=constant.NUM_CONV_FILTER,
        num_lc_layer=constant.NUM_LC_LAYER,
        num_lc_hidden=constant.NUM_LC_HIDDEN,
        lc_kernel=constant.LC_KERNEL,
        lc_stride=constant.LC_STRIDE,
        lc_pad=constant.LC_PAD,
        num_fc_layer=constant.NUM_FC_LAYER,
        num_fc_hidden=constant.NUM_FC_HIDDEN,
        num_bottleneck=constant.NUM_BOTTLENECK,
        dropout=constant.DROPOUT,
        glove_branch_at=constant.GLOVE_BRANCH_AT,
        **kargs
    ):
        if kargs:
            logger.debug('kargs not used in get_symbol:\n{}', pformat(kargs))

        self.for_training = for_training
        self.num_mini_batch = num_mini_batch
        self.num_semg_row = num_semg_row
        self.num_semg_col = num_semg_col
        self.num_semg_channel = num_semg_channel
        self.dropout = dropout

        data = mx.symbol.Variable('semg')
        data = self.get_bn('zscore', data)

        bottleneck = self.get_bottleneck(
            data,
            num_conv_layer=num_conv_layer,
            num_conv_filter=num_conv_filter,
            num_lc_layer=num_lc_layer,
            num_lc_hidden=num_lc_hidden,
            lc_kernel=lc_kernel,
            lc_stride=lc_stride,
            lc_pad=lc_pad,
            num_fc_layer=num_fc_layer,
            num_fc_hidden=num_fc_hidden,
            num_bottleneck=num_bottleneck,
            dropout=dropout
        )

        has_gesture_branch = gesture_loss_weight != 0
        has_glove_branch = glove_loss_weight and num_glove

        if has_gesture_branch:
            gesture_branch = self.get_gesture_branch(
                bottleneck,
                num_gesture=num_gesture,
                num_gesture_layer=num_gesture_layer,
                num_gesture_hidden=num_gesture_hidden,
                gesture_loss_weight=gesture_loss_weight
            )

        if self.for_training and has_glove_branch:
            glove_branch = self.get_glove_branch(
                self.get_glove_branch_root(bottleneck, glove_branch_at),
                glove_net=glove_net,
                num_glove=num_glove,
                num_glove_layer=num_glove_layer,
                num_glove_hidden=num_glove_hidden,
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

    def get_glove_branch_root(self, bottleneck, name):
        if name == 'bottleneck':
            return bottleneck
        internals = bottleneck.get_internals()
        if name == 'lc2':
            if self.dropout:
                return internals['lc_drop_output']
            else:
                return internals['lc2_relu_output']
        if name == 'fc2':
            if self.dropout:
                return internals['fc2_drop_output']
            else:
                return internals['fc2_relu_output']
        assert False, 'Unknown branch {}'.format(name)

    def infer_shape(self, data):
        net = data
        data_shape = (self.num_mini_batch,
                      self.num_semg_channel,
                      self.num_semg_row, self.num_semg_col)
        return tuple(int(s) for s in net.infer_shape(semg=data_shape)[1][0])

    def get_bn(self, name, data):
        #  Get BatchNorm or AdaBN
        if self.num_mini_batch > 1:
            net = data
            shape = self.infer_shape(net)
            net = mx.symbol.Reshape(net, shape=(-1, self.num_mini_batch * shape[1]) + shape[2:])
            net = mx.symbol.BatchNorm(
                name=self.concat_name(name, '_norm'),
                data=net,
                fix_gamma=True,
                momentum=0.9,
                attr={'wd_mult': '0', 'lr_mult': '0'}
            )
            net = mx.symbol.Reshape(data=net, shape=(-1,) + shape[1:])
            if len(shape) == 4:
                #  Convolution
                gamma = mx.symbol.Variable(self.concat_name(name, '_gamma'), shape=(1, shape[1], 1, 1))
                beta = mx.symbol.Variable(self.concat_name(name, '_beta'), shape=(1, shape[1], 1, 1))
            else:
                #  Fully connected
                gamma = mx.symbol.Variable(self.concat_name(name, '_gamma'), shape=(1, shape[1]))
                beta = mx.symbol.Variable(self.concat_name(name, '_beta'), shape=(1, shape[1]))
            net = mx.symbol.broadcast_mul(net, gamma)
            net = mx.symbol.broadcast_plus(net, beta, name=self.concat_name(name, '_last'))
        else:
            net = mx.symbol.BatchNorm(
                name=name,
                data=data,
                fix_gamma=False,
                momentum=0.9
            )
        return net

    def get_bn_relu(self, name, data):
        net = self.get_bn(self.concat_name(name, '_bn'), data)
        return mx.symbol.Activation(name=self.concat_name(name, '_relu'), data=net, act_type='relu')

    def concat_name(self, name, suffix):
        return name if name is None else name + suffix

    def im2col(self, data, name, kernel, pad=(0, 0), stride=(1, 1)):
        shape = self.infer_shape(data)
        return mx.symbol.Convolution(
            name=name,
            data=data,
            num_filter=shape[1] * kernel[0] * kernel[1],
            kernel=kernel,
            stride=stride,
            pad=pad,
            no_bias=True,
            attr={'lr_mult': '0'}
        )

    def get_lc(self, name, data, num_filter, no_bias, rows, cols, kernel=1, stride=1, pad=0):
        net = data

        if kernel != 1:
            net = self.im2col(name=name + '_im2col', data=net,
                              kernel=(kernel, kernel),
                              pad=(pad, pad),
                              stride=(stride, stride))
            return self.get_lc(name, net, num_filter, no_bias, rows, cols)

        net = mx.symbol.Reshape(net, shape=(0, 0, -1))

        nets = mx.symbol.SliceChannel(net, num_outputs=rows * cols, axis=2)
        W = [[mx.symbol.Variable(name=name + '_fc%d_weight' % (row * cols + col))
              for col in range(cols)] for row in range(rows)]
        nets = [mx.symbol.FullyConnected(name=name + '_fc%d' % i,
                                         data=nets[i],
                                         num_hidden=num_filter,
                                         no_bias=no_bias,
                                         weight=W[i // cols][i % cols])
                for i in range(rows * cols)]
        nets = [mx.symbol.Reshape(p, shape=(0, 0, 1)) for p in nets]
        net = mx.symbol.Concat(*nets, dim=2)
        return mx.symbol.Reshape(net, shape=(0, 0, rows, cols))

    def get_lc_v2(self, name, data, num_filter, no_bias, kernel=1, stride=1, pad=0):
        net = data

        if kernel != 1:
            net = self.im2col(name=name + '_im2col', data=net,
                              kernel=(kernel, kernel),
                              pad=(pad, pad),
                              stride=(stride, stride))
            return self.get_lc_v2(name, net, num_filter, no_bias)

        _, _, rows, cols = self.infer_shape(net)
        net = mx.symbol.Reshape(net, shape=(0, 0, -1))

        nets = mx.symbol.SliceChannel(net, num_outputs=rows * cols, axis=2)
        W = [[mx.symbol.Variable(name=name + '_fc%d_weight' % (row * cols + col))
              for col in range(cols)] for row in range(rows)]
        nets = [mx.symbol.FullyConnected(name=name + '_fc%d' % i,
                                         data=nets[i],
                                         num_hidden=num_filter,
                                         no_bias=no_bias,
                                         weight=W[i // cols][i % cols])
                for i in range(rows * cols)]
        nets = [mx.symbol.Reshape(p, shape=(0, 0, 1)) for p in nets]
        net = mx.symbol.Concat(*nets, dim=2)
        return mx.symbol.Reshape(net, shape=(0, 0, rows, cols))

    def get_bottleneck(
        self,
        data,
        num_conv_layer,
        num_conv_filter,
        num_lc_layer,
        num_lc_hidden,
        lc_kernel,
        lc_stride,
        lc_pad,
        num_fc_layer,
        num_fc_hidden,
        num_bottleneck,
        dropout
    ):
        net = data

        # Convolution
        for i in range(num_conv_layer):
            name = 'conv%d' % (i + 1)
            net = Convolution(
                name=name,
                data=net,
                num_filter=num_conv_filter,
                kernel=(3, 3),
                stride=(1, 1),
                pad=(1, 1),
                no_bias=True
            )
            net = self.get_bn_relu(name, net)

        # Locally connected
        rows = self.num_semg_row
        cols = self.num_semg_col
        for i in range(num_lc_layer):
            name = 'lc%d' % (i + 1) if num_lc_layer > 1 else 'lc'
            rows //= lc_stride
            cols //= lc_stride
            net = self.get_lc(name, net,
                              num_lc_hidden,
                              no_bias=True,
                              rows=rows, cols=cols,
                              kernel=lc_kernel,
                              stride=lc_stride,
                              pad=lc_pad)
            net = self.get_bn_relu(name, net)
        net = Dropout(name='lc_drop', data=net, p=dropout)

        # Fully connected
        for i in range(num_fc_layer):
            name = 'fc%d' % (i + 1)
            net = self.get_fc_bn_relu(name=name, data=net, num_hidden=num_fc_hidden)
            net = Dropout(name=name + '_drop', data=net, p=dropout)

        # Bottleneck
        return self.get_fc_bn_relu(name='bottleneck', data=net, num_hidden=num_bottleneck)

    def get_gesture_branch(
        self,
        bottleneck,
        num_gesture,
        num_gesture_layer,
        num_gesture_hidden,
        gesture_loss_weight
    ):
        net = bottleneck
        if num_gesture_layer:
            for i in range(num_gesture_layer):
                net = self.get_fc_bn_relu(name='gesture_fc%d' % (i + 1),
                                          data=net,
                                          num_hidden=num_gesture_hidden)
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
        bottleneck,
        glove_net,
        num_glove,
        num_glove_layer,
        num_glove_hidden,
        glove_loss_weight
    ):
        net = bottleneck

        if glove_net:
            context = dict(
                dropout=self.dropout,
                get_act=lambda name, data: self.get_bn_relu(name, data),
                prefix='glove'
            )
            net = parse(context, glove_net).compile(context, net)
        else:
            if num_glove_layer:
                for i in range(num_glove_layer):
                    net = self.get_fc_bn_relu(name='glove_fc%d' % (i + 1),
                                              data=net,
                                              num_hidden=num_glove_hidden)

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

    def get_fc_bn_relu(self, name, data, num_hidden):
        net = self.get_fc(name=name, data=data, num_hidden=num_hidden, no_bias=True)
        net = self.get_bn_relu(name, net)
        return net

    def get_fc(self, name, data, num_hidden, no_bias):
        return mx.symbol.FullyConnected(
            name=name,
            data=data,
            num_hidden=num_hidden,
            no_bias=no_bias
        )


def Dropout(**kargs):
    p = kargs.pop('p')
    return kargs.pop('data') if p == 0 else mx.symbol.Dropout(p=p, **kargs)


def Convolution(*args, **kargs):
    kargs['cudnn_tune'] = 'fastest'
    return mx.symbol.Convolution(*args, **kargs)
