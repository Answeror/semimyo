from __future__ import division
import nose.tools as nt
import mxnet as mx


class StochasticTransformLossMixin(object):

    def stochastic_transform_loss_begin(self, data, n):
        shape = self.infer_shape(data)
        batch_size = shape[0]
        data = mx.symbol.Reshape(data, shape=(1,) + shape)
        data = mx.symbol.broadcast_axis(data, axis=0, size=n)
        data = mx.symbol.Reshape(data, shape=(n * batch_size,) + shape[1:])
        return data

    def stochastic_transform_loss_end(self, data, n, w, index):
        net = data
        shape = self.infer_shape(net)
        nt.assert_equal(len(shape), 2)
        batch_size = shape[0] // n
        num_class = shape[1]
        net = mx.symbol.Reshape(net, shape=(n, batch_size, num_class))
        net = mx.symbol.SwapAxis(net, dim1=0, dim2=1)
        W = mx.symbol.Variable('stochastic_transform_loss_W%d' % index,
                               attr={'__shape__': str((n * (n - 1) // 2, n)),
                                     '__type__': 'float32',
                                     'lr_mult': '0',
                                     'wd_mult': '0'})
        W = mx.symbol.expand_dims(W, axis=0)
        W = mx.symbol.broadcast_axis(W, axis=0, size=batch_size)
        net = mx.symbol.batch_dot(W, net)
        net = mx.symbol.Flatten(net)
        net = mx.symbol.sum(mx.symbol.square(net), axis=1)
        net = mx.symbol.MakeLoss(data=net, grad_scale=w)
        return net

    def get_stochastic_transform_loss(self, *data, **kargs):
        get_loss = kargs.pop('get_loss')
        n = kargs.pop('num_stochastic_transform_loss_sample')
        w = kargs.pop('stochastic_transform_loss_weight')
        if n and w:
            net = get_loss(*[self.stochastic_transform_loss_begin(data[i], n=n)
                             for i in range(len(data))])
            if isinstance(net, mx.symbol.Symbol):
                net = self.stochastic_transform_loss_end(net)
            else:
                net = net + [self.stochastic_transform_loss_end(net[i], n=n, w=w, index=i)
                             for i in range(len(net))]
        else:
            net = get_loss(*data)
        return net
