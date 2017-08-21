#  inspired by https://github.com/pallets/click/blob/master/click/globals.py
from __future__ import division
from threading import local
import nose.tools as nt


_local = local()


def get_current_context():
    from .context import Context
    return _local.__dict__.setdefault('stack', [Context()])[-1]


def push_context(ctx):
    """Pushes a new context to the current stack."""
    from .context import Context
    _local.__dict__.setdefault('stack', [Context()]).append(ctx)


def pop_context():
    """Removes the top level from the stack."""
    nt.assert_greater(len(_local.stack), 0)
    if len(_local.stack) == 1:
        raise Exception('Cannot pop root context')
    _local.stack.pop()


class CurrentContextProxy(object):

    def __getattr__(self, name):
        return getattr(get_current_context(), name)

    def __setattr__(self, name, value):
        assert False, 'CurrentContextProxy cannot be changed, use push instead'

    def __contains__(self, name):
        return name in get_current_context()


ctx = CurrentContextProxy()


__all__ = ['ctx', 'get_current_context', 'push_context', 'pop_context']
