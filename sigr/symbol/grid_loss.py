from __future__ import division
import mxnet as mx


class GridLossMixin(object):

    def __init__(self, *args, **kargs):
        super(GridLossMixin, self).__init__(*args, **kargs)
        assert hasattr(self, 'im2col'), 'GridLossMixin must be used with im2col in BaseSymbol'

    def get_grid_loss(self, data, name, lam, get_loss,
                      kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                      context=None):
        net = data
        if not (kernel == (1, 1) and stride == (1, 1) and pad == (0, 0)):
            net = self.im2col(name=name + '_im2col',
                              data=net,
                              kernel=kernel,
                              pad=pad,
                              stride=stride)
        return self._get_grid_loss(net, name, lam, get_loss, context)

    def _get_grid_loss(self, data, name, lam, get_loss, context):
        net = data
        _, num_channel, num_row, num_col = self.infer_shape(net)
        num_grid = num_row * num_col

        major_name = name + '_major'
        minor_name = [name + '_minor%d' % i for i in range(num_grid)]

        if context is None:
            context = {}

        context = context.copy()
        if 'fc_weight' not in context:
            context['fc_weight'] = {}
        if 'fc_bias' not in context:
            context['fc_bias'] = {}

        major_net = net
        minor_net = mx.symbol.Reshape(net, shape=(0, 0, num_grid))
        branches = mx.symbol.SliceChannel(minor_net, num_outputs=num_grid, axis=2)
        minor_nets = [get_loss(data=mx.symbol.Flatten(branches[i]),
                               name=minor_name[i],
                               grad_scale=lam,
                               context=context) for i in range(1)]

        fc_weight = context['fc_weight']
        for key in list(fc_weight):
            if key.startswith(minor_name[0]) and not key.startswith(minor_name[0] + '_fc1'):
                suffix = key[len(minor_name[0]):]
                fc_weight[major_name + suffix] = fc_weight[key]
                for i in range(1, num_grid):
                    fc_weight[minor_name[i] + suffix] = fc_weight[key]

        fc_bias = context['fc_bias']
        for key in list(fc_bias):
            if key.startswith(minor_name[0]) and not key.startswith(minor_name[0] + '_fc1'):
                suffix = key[len(minor_name[0]):]
                fc_bias[major_name + suffix] = fc_bias[key]
                for i in range(1, num_grid):
                    fc_bias[minor_name[i] + suffix] = fc_bias[key]

        minor_nets += [get_loss(data=mx.symbol.Flatten(branches[i]),
                                name=minor_name[i],
                                grad_scale=lam,
                                context=context) for i in range(1, num_grid)]

        fc_weight = context['fc_weight']
        for key in list(fc_weight):
            for grid in range(num_grid):
                if key.startswith(minor_name[grid] + '_fc1'):
                    suffix = key[len(minor_name[grid]):]
                    name = major_name + suffix
                    if name in fc_weight:
                        break
                    W = mx.symbol.Concat(*[self.as_W(fc_weight[minor_name[i] + suffix])
                                           for i in range(num_grid)], dim=2)
                    fc_weight[name] = mx.symbol.Reshape(data, shape=(0, -1))

        fc_bias = context['fc_bias']
        for key in list(fc_bias):
            for grid in range(num_grid):
                if key.startswith(minor_name[grid] + '_fc1'):
                    suffix = key[len(minor_name[grid]):]
                    name = major_name + suffix
                    if name in fc_bias:
                        break
                    data = mx.symbol.ElementWiseSum(*[fc_bias[minor_name[i] + suffix]
                                                      for i in range(num_grid)])
                    fc_bias[name] = data

        major_net = get_loss(data=mx.symbol.Flatten(major_net),
                             name=major_name,
                             context=context)

        return [major_net] + minor_nets

    def get_grid_fc(self, data, name, num_hidden, no_bias,
                    kernel=(1, 1), stride=(1, 1), pad=(0, 0), context=None):
        net = data
        if not (kernel == (1, 1) and stride == (1, 1) and pad == (0, 0)):
            net = self.im2col(name=name + '_im2col',
                              data=net,
                              kernel=kernel,
                              pad=pad,
                              stride=stride)
        return self._get_grid_fc(net, name, num_hidden, no_bias, context)

    def _get_grid_fc(self, data, name, num_hidden, no_bias, context):
        net = data
        _, num_channel, num_row, num_col = self.infer_shape(net)
        num_grid = num_row * num_col

        major_name = name + '_major'
        minor_name = [name + '_minor%d' % i for i in range(num_grid)]

        if context is None:
            context = {}

        context = context.copy()
        if 'fc_weight' not in context:
            context['fc_weight'] = {}
        if 'fc_bias' not in context:
            context['fc_bias'] = {}

        major_net = net
        minor_net = mx.symbol.Reshape(net, shape=(0, 0, num_grid))
        branches = mx.symbol.SliceChannel(minor_net, num_outputs=num_grid, axis=2)
        minor_nets = [self.get_fc(data=mx.symbol.Flatten(branches[i]),
                                  name=minor_name[i] + '_fc',
                                  num_hidden=num_hidden,
                                  no_bias=no_bias,
                                  context=context) for i in range(num_grid)]

        fc_weight = context['fc_weight']
        for key in list(fc_weight):
            for grid in range(num_grid):
                suffix = key[len(minor_name[grid]):]
                name = major_name + suffix
                if name in fc_weight:
                    break
                    W = mx.symbol.Concat(*[self.as_W(fc_weight[minor_name[i] + suffix])
                                           for i in range(num_grid)], dim=2)
                    fc_weight[name] = mx.symbol.Reshape(data, shape=(0, -1))

        fc_bias = context['fc_bias']
        for key in list(fc_bias):
            for grid in range(num_grid):
                suffix = key[len(minor_name[grid]):]
                name = major_name + suffix
                if name in fc_bias:
                    break
                data = mx.symbol.ElementWiseSum(*[fc_bias[minor_name[i] + suffix]
                                                  for i in range(num_grid)])
                fc_bias[name] = data

        major_net = self.get_fc(data=mx.symbol.Flatten(major_net),
                                name=major_name + '_fc',
                                num_hidden=num_hidden,
                                no_bias=no_bias,
                                context=context)

        return [major_net] + minor_nets

    def as_W(self, W):
        W = mx.symbol.Cast(W, dtype='float32')
        W = mx.symbol.Reshape(W, shape=(0, 0, 1))
        return W
