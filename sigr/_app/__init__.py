from __future__ import division
import click
from logbook import Logger, StderrHandler
from .. import utils


logger = Logger(__name__)
_modstack = utils.Modstack(
    formula='.app_{name}',
    package=__package__
)


@click.group()
def app():
    pass


def _load_all():
    import os
    for name in os.listdir(os.path.dirname(__file__)):
        if name.startswith('app_') and name.endswith('.py'):
            subname = name[len('app_'):-len('.py')]
            try:
                _modstack.get(name=subname)
            except:
                logger.exception('Load app subname failed')


def run():
    StderrHandler().push_application()
    _load_all()
    app(obj=utils.Bunch())


def Cmd(name, *args):
    from copy import copy

    class cls(object):

        def __init__(self, args):
            self.args = copy(args)

        def __call__(self, func):
            args = copy(self.args)
            args.insert(0, app.command())
            args.append(utils.packargs)
            args.append((utils.rename if name == 'main' else utils.subname)(name))
            for arg in reversed(args):
                func = arg(func)
            return func

        def option(self, *args, **kargs):
            from .options import options
            inst = cls(self.args)
            inst.args.append(options.option(*args, **kargs))
            return inst

    return cls(list(args))


def d(func):
    return func


__all__ = ['run']
