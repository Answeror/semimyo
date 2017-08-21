from __future__ import division
from logbook import Logger
from contextlib import contextmanager
import mxnet as mx


logger = Logger(__name__)


class Base(object):

    def __init__(self, **kargs):
        kargs.setdefault('num_mini_batch', 1)
        kargs.setdefault('loss_normalization', 'null')
        self.__dict__.update(kargs)

    def get_softmax(self, data, name, label, grad_scale=1, **kargs):
        if self.for_training:
            kargs.setdefault('normalization', self.loss_normalization)
            return mx.symbol.SoftmaxOutput(
                name=name,
                data=data,
                label=label,
                grad_scale=grad_scale,
                **kargs
            )
        else:
            return mx.symbol.SoftmaxActivation(name=name, data=data)


class OneLineNetMixin(object):

    def get_one_line_net(self, context, text, data, **kargs):
        if not text:
            return data

        from .symbol_common import compile
        if kargs:
            context = self.push_context(context, **kargs)
        return compile(context, text, data)

    def push_context(self, context, **kargs):
        context = context.copy()
        context.update(kargs)
        return context


class ShortnetMixin(object):

    def __init__(self, *args, **kargs):
        super(ShortnetMixin, self).__init__(*args, **kargs)
        if hasattr(self, 'shortnet_args'):
            self.shortnet_args = self.shortnet_args.copy()
            for key, value in self.__dict__.items():
                if key != 'shortnet_args':
                    self.shortnet_args.setdefault(key, value)
            for key in self.__class__.__dict__:
                if key != 'shortnet_args':
                    self.shortnet_args.setdefault(key, getattr(self, key))

    @contextmanager
    def push_shortnet_context(self, *args, **kargs):
        from ..shortnet import ctx, Backend
        with ctx.push(Backend('mxnet')):
            with ctx.push(*args, **kargs):
                yield ctx

    def get_shortnet(self, text, data, *args, **kargs):
        if not text:
            return data

        from ..shortnet import build
        with self.push_shortnet_context(*args, **kargs):
            return build(text, data)


class Symbol(ShortnetMixin, OneLineNetMixin, Base):
    pass


BaseSymbol = Symbol


__all__ = ['BaseSymbol']
