from __future__ import division
from .modlib import Modstack
from .re_scan import Scanner


class Null(object):
    pass


null = Null()


class Get(object):

    def __init__(self, host, key, parent=None):
        self.host = host
        self.key = key
        self.parent = parent

    def _try(self):
        value = null
        if self.parent:
            value = self.parent._try()
        if value is null:
            if self.host is not None:
                if hasattr(self.host, 'get'):
                    value = self.host.get(self.key, null)
                else:
                    value = getattr(self.host, self.key, null)
        return value

    def default(self, value):
        return self(default=value)

    def value(self):
        return self()

    def __call__(self, *args, **kargs):
        if len(args) == 0:
            value = self._try()
            if value is null:
                value = kargs.pop('default', null)
                if value is null:
                    raise KeyError(self.key)
            return value
        elif len(args) == 1:
            host = args[0]
            return Get(host, self.key, self)
        else:
            host, key = args
            return Get(host, key, self)


def get(host, key):
    return Get(host, key)


__all__ = ['get', 'Modstack', 'Scanner']
