from contextlib import contextmanager
import os
import six
import numpy as np
import nose.tools as nt
from functools import partial
from .proxy import LazyProxy
assert LazyProxy
from .modlib import Modstack
assert Modstack


F = partial
g = {}


@contextmanager
def logging_context(path=None, level=None):
    from logbook import StderrHandler, FileHandler, NullHandler
    from logbook.compat import redirected_logging
    with NullHandler().applicationbound():
        with StderrHandler(level=level or 'INFO', bubble=False).applicationbound():
            if path:
                if not os.path.isdir(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))
                with FileHandler(path, bubble=True).applicationbound():
                    with redirected_logging():
                        yield
            else:
                with redirected_logging():
                    yield


def return_list(func):
    import inspect
    from functools import wraps
    assert inspect.isgeneratorfunction(func)

    @wraps(func)
    def wrapped(*args, **kargs):
        return list(func(*args, **kargs))

    return wrapped


@return_list
def continuous_segments(label):
    label = np.asarray(label)

    if not len(label):
        return

    breaks = list(np.where(label[:-1] != label[1:])[0] + 1)
    for begin, end in zip([0] + breaks, breaks + [len(label)]):
        assert begin < end
        yield begin, end


def cached(*args, **kargs):
    import joblib as jb
    from .. import CACHE
    memo = getattr(cached, 'memo', None)
    if memo is None:
        cached.memo = memo = jb.Memory(CACHE, verbose=0)
    return memo.cache(*args, **kargs)


def _return_cached(func, *args, **kargs):
    return func.call_and_shelve(*args, **kargs).get()


def return_cached(func):
    '''
    Ensure load from disk.
    Otherwise following cached methods like vote will have two caches,
    one for the first computation,
    and the other for the cached one.
    '''
    assert hasattr(func, 'call_and_shelve')
    return partial(_return_cached, func)


@return_cached
@cached(ignore=['args_not_cached'])
def cached_call(func, args_not_cached, *args, **kargs):
    return func(args_not_cached, *args, **kargs)


def get_segments(data, window):
    return windowed_view(
        data.flat,
        window * data.shape[1],
        (window - 1) * data.shape[1]
    )


def windowed_view(arr, window, overlap):
    from numpy.lib.stride_tricks import as_strided
    arr = np.asarray(arr)
    window_step = window - overlap
    new_shape = arr.shape[:-1] + ((arr.shape[-1] - overlap) // window_step,
                                  window)
    new_strides = (arr.strides[:-1] + (window_step * arr.strides[-1],) +
                   arr.strides[-1:])
    return as_strided(arr, shape=new_shape, strides=new_strides)


class Bunch(dict):

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def copy(self):
        return Bunch(**self)


def packargs(func):
    from functools import wraps
    import inspect
    import click

    @wraps(func)
    def wrapped(**kargs):
        ctx = click.get_current_context()
        args = ctx.obj.copy()
        ignore = inspect.getargspec(func).args
        args.update({k: v for k, v in kargs.items()
                     if k not in ignore and k not in args})
        ctx.obj = args
        return func(args, **{k: v for k, v in kargs.items() if k in ignore})
    return wrapped


class Append(object):

    def __init__(self, value):
        self.value = value

    def __call__(self, cont, key):
        from copy import copy
        if isinstance(self.value, dict):
            d = copy(cont.get(key, {}))
            d.update(self.value)
            cont[key] = d
        else:
            a = copy(cont.get(key, []))
            if isinstance(self.value, (tuple, list)):
                a.extend(self.value)
            else:
                a.append(self.value)
            cont[key] = a


@contextmanager
def pushargs(*args, **kargs):
    if not args and not kargs:
        yield
        return

    if args:
        with pushargs(**args[0]):
            with pushargs(*args[1:], **kargs):
                yield
        return

    import click
    from copy import copy
    ctx = click.get_current_context()
    try:
        orig = ctx.obj
        args = copy(orig)

        for key, value in kargs.items():
            if not isinstance(value, Append):
                args[key] = value
            else:
                value(args, key)

        ctx.obj = args
        yield ctx
    finally:
        ctx.obj = orig


use = pushargs


@contextmanager
def appendargs(**kargs):
    with pushargs(**{key: Append(value) for key, value in kargs.items()}) as ctx:
        yield ctx


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    from scipy.signal import butter, lfilter

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='bandpass')
    y = lfilter(b, a, data)
    return y


def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    from scipy.signal import butter, lfilter

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='bandstop')
    y = lfilter(b, a, data)
    return y


def butter_lowpass_filter(data, cut, fs, order, zero_phase=False):
    from scipy.signal import butter, lfilter, filtfilt

    nyq = 0.5 * fs
    cut = cut / nyq

    b, a = butter(order, cut, btype='low')
    y = (filtfilt if zero_phase else lfilter)(b, a, data)
    return y


def format_args(func, *args):
    return partial(_format_args, func, args)


def _format_args(func, args, **kargs):
    for arg in args:
        kargs[arg] = kargs[arg].format(**kargs)
    return func(**kargs)


def ignore_args(func, *args):
    return partial(_ignore_args, func, args)


def _ignore_args(func, args, **kargs):
    return func(**{k: v for k, v in kargs.items() if k not in args})


def get_index(a, ignores=None):
    '''Convert label to 0 based index'''
    b = set(a)
    if ignores:
        b -= set(ignores)
    b = sorted(b)
    return np.array([x if ignores and x in ignores else b.index(x) for x in a.ravel()]).reshape(a.shape)


def make_lru_dict(room):
    try:
        from lru import LRU
        return LRU(room)
    except:
        return {}


class DeprecatedError(Exception):

    def __init__(self, name, reason):
        message = name + 'is deprecated'
        if reason:
            message += ': ' + reason
        super(DeprecatedError, self).__init__(message)


def deprecated(name, reason=None):
    raise DeprecatedError(name, reason)


def logargs(logger, args, level='INFO'):
    from pprint import pformat
    getattr(logger, level.lower())('Args:\n{}', pformat(args))


def hash(data):
    import joblib as jb
    return jb.hash(data)


def rename(name):
    def wrap(func):
        func.__name__ = name
        return func
    return wrap


def subname(name):
    def wrap(func):
        func.__name__ = name + '_' + func.__name__
        return func
    return wrap


def appname(name):
    return name.split('.')[-1][len('app_'):]


class classproperty(object):

    def __init__(self, func):
        self.func = func

    def __get__(self, inst, cls):
        return self.func(cls)


def get_semg_prev(semg, steps):
    prevs = []
    for step in steps:
        prev = np.vstack([semg[[0 for _ in range(step)]], semg])
        prev = prev[:-step]
        nt.assert_equal(prev.shape, semg.shape)
        prev = np.expand_dims(prev, axis=1)
        prevs.append(prev)
    prev = np.concatenate(prevs, axis=1)
    return prev


def load_csl_semg(path):
    import scipy.io as sio
    semg = sio.loadmat(path)['gestures']
    semg = [np.transpose(np.delete(segment.astype(np.float32), np.s_[7:192:8], 0))
            for segment in semg.flat]
    return semg


def invoke(func, **kargs):
    #  import click
    #  with pushargs(**kargs):
        #  ctx = click.get_current_context()
        #  return ctx.invoke(func, **ctx.obj)

    import click
    ctx = click.get_current_context()
    args = {}
    args.update(ctx.obj)
    for key, value in kargs.items():
        if not isinstance(value, Append):
            args[key] = value
        else:
            value(args, key)

    with restore_field(ctx, obj=Bunch()):
        return ctx.invoke(func, **args)


def exclude(a, ignore):
    return [x for x in a if x not in ignore]


@contextmanager
def restore_field(*args, **kargs):
    nt.assert_equal(len(args), 1)
    obj = args[0]
    backup = {key: getattr(obj, key) for key in kargs}
    try:
        for key, value in kargs.items():
            setattr(obj, key, value)
        yield obj
    finally:
        for key, value in backup.items():
            setattr(obj, key, value)


def get(*args, **kargs):
    from ..shortnet.utils import get
    return get(*args, **kargs)


def g_set(obj, value):
    g[hash(obj)] = value


def shuf(a):
    import random
    from time import time
    random.seed(time())
    a = list(a)
    random.shuffle(a)
    return a


def get_downsample_index(n, downsample, random_state):
    index = np.arange(n)
    random_state.shuffle(index)
    index = index[::downsample]
    return index


def expand(begin, end, n, expand):
    def as_int(x, span):
        if isinstance(x, six.string_types):
            assert x.endswith('%')
            x = int(round(span * float(x[:-1]) / 100))
        return x

    if isinstance(expand, (tuple, list)):
        before, after = expand
    else:
        before, after = expand, expand

    span = end - begin
    before = as_int(before, span)
    after = as_int(after, span)

    begin = max(0, begin - before)
    end = min(n, end + after)
    return begin, end
