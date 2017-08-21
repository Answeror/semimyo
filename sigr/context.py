from __future__ import division
from contextlib import contextmanager
from .shortnet import ctx


@contextmanager
def Context(log=None, parallel=False, level=None, mxnet_context=None):
    @contextmanager
    def _mxnet_context():
        if mxnet_context is None:
            yield
        else:
            with mxnet_context:
                yield

    @contextmanager
    def _parallel_context():
        if not parallel:
            yield
        else:
            import joblib as jb
            from multiprocessing import cpu_count
            with jb.Parallel(n_jobs=cpu_count()) as par:
                Context.parallel = par
                yield

    from .utils import logging_context
    with logging_context(log, level=level):
        with _parallel_context():
            with _mxnet_context():
                yield


__all__ = ['Context', 'ctx']
