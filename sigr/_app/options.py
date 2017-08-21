from __future__ import division
import click
import six
from .. import constant


class Options(object):

    def __init__(self, **kargs):
        self.kargs = kargs

        self.gpu_option = self.option('--gpu', type=int, multiple=True, default=[0], **self.kargs)
        self.batch_size_option = self.option('--batch-size', type=int, default=constant.BATCH_SIZE, **self.kargs)
        self.epoch_option = self.option('--epoch', type=int, default=constant.NUM_EPOCH, **self.kargs)
        self.deprecated_option = self.option('--deprecated', is_flag=True, help='Show deprecated results')

    def json_option(self, *args, **kargs):
        import json
        kargs.setdefault('callback', lambda ctx, param, value:
                         json.loads(value) if isinstance(value, six.string_types) else value)
        return self.option(*args, **kargs)

    def option(self, *args, **kargs):
        for key, value in self.kargs.items():
            kargs.setdefault(key, value)
        return click.option(*args, **kargs)


options = Options()
for key in dir(options):
    locals()[key] = getattr(options, key)
