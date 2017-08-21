from __future__ import division
from . import context_stack as cs
from .context_stack import ctx


class Context(object):

    def __init__(self, parent=None, **kargs):
        self.parent = parent
        self.__dict__.update(kargs)

    def __getattr__(self, name):
        if self.parent:
            return getattr(self.parent, name)
        return self.__getattribute__(name)

    def __enter__(self):
        cs.push_context(self)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        cs.pop_context()

    def push(self, cls=None, **kargs):
        if cls is None:
            #  cls = type(self)
            cls = Context
        return cls(parent=self, **kargs)

    def __contains__(self, name):
        return hasattr(self, name)

    def get(self, name, *args):
        try:
            return getattr(self, name, *args)
        except AttributeError:
            raise KeyError(name)

    def get_act(self, *args, **kargs):
        return getattr(self, 'get_' + self.act)(*args, **kargs)


BaseContext = Context


__all__ = ['ctx', 'Context', 'BaseContext']
