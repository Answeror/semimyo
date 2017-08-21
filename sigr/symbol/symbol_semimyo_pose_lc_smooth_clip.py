from __future__ import division
import mxnet as mx
from logbook import Logger
from pprint import pformat
import nose.tools as nt
from .base_symbol import BaseSymbol


logger = Logger(__name__)


def get_symbol(**kargs):
    return Symbol(**kargs).net


class Symbol(BaseSymbol):

    def get_shortnet(self, text, data, **kargs):
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
        shared_net=None,
        gesture_net=None,
        pose_net=None,
        gesture_loss_weight=None,
        pose_loss_weight=None,
        **kargs
    ):
        if kargs:
            logger.debug('kargs not used in get_symbol:\n{}', pformat(kargs))

        super(Symbol, self).__init__(**kargs)

        data = mx.symbol.Variable('semg')
        shared_net = self.get_shortnet(shared_net, data)

        has_gesture_branch = gesture_loss_weight > 0
        assert has_gesture_branch
        has_pose_branch = pose_loss_weight > 0

        gesture_branch = self.get_gesture_branch(
            self.get_shortnet('id:gesture_branch', shared_net),
            gesture_net,
            gesture_loss_weight
        )

        if self.for_training:
            loss = [gesture_branch]
            if has_pose_branch:
                pose_branch = self.get_pose_branch(
                    self.get_shortnet('id:pose_branch', shared_net),
                    pose_net,
                    pose_loss_weight
                )
                loss.append(pose_branch)
                #  loss.append(self.get_smooth_loss(pose_branch, 'pose'))
            loss.append(self.get_smooth_loss(
                self.get_shortnet(
                    'cg1,id:lc_smooth_loss_branch',
                    gesture_branch.get_internals()['lc2_relu_output']
                ),
                'lc'
            ))
            self.net = mx.symbol.Group(loss)
        else:
            self.net = gesture_branch

        self.net.num_semg_row = self.num_semg_row
        self.net.num_semg_col = self.num_semg_col
        self.net.num_semg_channel = self.num_semg_channel
        self.net.data_shape_1 = self.num_semg_channel

    def infer_shape(self, data):
        net = data
        data_shape = (self.batch_size * self.window,
                      self.num_semg_channel,
                      self.num_semg_row, self.num_semg_col)
        return tuple(int(s) for s in net.infer_shape(semg=data_shape)[1][0])

    def get_gesture_branch(self, data, text, gesture_loss_weight):
        net = data
        net = self.get_shortnet(text, net, prefix='gesture')
        net = self.get_softmax(
            data=net,
            name='gesture_softmax',
            label=mx.symbol.Variable('gesture'),
            grad_scale=gesture_loss_weight
        )
        return net

    def get_pose_branch(self, data, text, pose_loss_weight):
        net = data
        net = self.get_shortnet(text, net, prefix='pose')
        net = self.get_softmax(
            data=net,
            name='pose_softmax',
            label=mx.symbol.Variable('pose'),
            grad_scale=pose_loss_weight
        )
        return net

    def get_smooth_loss(self, data, tag):
        nt.assert_equal(self.window, 2)

        net = data

        #  net = self.get_shortnet('id:%s_smooth_loss_branch' % tag, net)
        #  net = mx.symbol.Reshape(
            #  net,
            #  shape=(self.batch_size, self.window, -1)
        #  )
        #  net = mx.symbol.SliceChannel(
            #  net,
            #  axis=1,
            #  num_outputs=self.window,
            #  squeeze_axis=True
        #  )
        #  shape = self.infer_shape(net[0])
        #  nt.assert_equal(len(shape), 2)
        #  nt.assert_equal(shape[0], self.batch_size)
        #  return mx.symbol.LinearRegressionOutput(
            #  data=net[0],
            #  label=net[1],
            #  grad_scale=self.smooth_loss_weight
        #  )

        net = mx.symbol.Reshape(
            net,
            shape=(self.batch_size, self.window, -1)
        )
        net = mx.symbol.SliceChannel(
            net,
            axis=1,
            num_outputs=self.window
        )
        net = [net[rhs] - net[lhs] for lhs, rhs in zip(range(self.window - 1), range(1, self.window))]
        net = mx.symbol.Concat(*net, dim=1)
        net = mx.symbol.square(net)
        net = mx.symbol.Flatten(net)
        net = mx.symbol.sum(net, axis=1)
        nt.assert_equal(self.infer_shape(net)[0], self.batch_size)
        return mx.symbol.MakeLoss(net, name=tag + '_smooth_loss', grad_scale=self.smooth_loss_weight)
