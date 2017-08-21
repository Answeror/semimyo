from __future__ import division
import os
import numpy as np
from logbook import Logger
import joblib as jb
import six
from . import base
from .. import data_iter as di
from ... import Context, utils


logger = Logger(__name__)


class Dataset(base.StepMixin, base.BaseDataset):

    def __init__(self, **kargs):
        self.expand = kargs.pop('expand')
        super(Dataset, self).__init__(**kargs)

    @property
    def get_trial_cls(dataset):
        class cls(super(Dataset, dataset).get_trial_cls):

            def update_memo(self, combo):
                path = os.path.join(
                    self.dataset.root,
                    'subject%d' % combo.subject,
                    'session%d' % combo.session,
                    'gest%d.mat' % combo.gesture
                )
                logger.debug('Load {}', path)
                for i, (semg, prev, mask) in enumerate(_load(path,
                                                             self.dataset.steps,
                                                             self.dataset.expand,
                                                             self.preprocess)):
                    trial = i + 1
                    combo = combo.copy()
                    combo.trial = trial
                    if not self.dataset.good_combo(combo):
                        continue

                    gesture = np.repeat(combo.gesture, len(semg))
                    gesture[~mask] = -1
                    self.set_memo(
                        combo,
                        self.trial_cls(
                            semg=semg,
                            prev=prev,
                            subject=np.repeat(combo.subject, len(semg)),
                            session=np.repeat(combo.session, len(semg)),
                            gesture=gesture
                        )
                    )

        return cls

    @utils.classproperty
    def dataiter_cls(dataset):
        class cls(
            di.InitMixin(data_shapes=[('semg', (-1, 1, dataset.num_semg_row, dataset.num_semg_col))],
                         label_shapes=[('gesture', (-1,)),
                                       ('prev', (-1, 1, dataset.num_semg_row, dataset.num_semg_col)),
                                       ('subject', (-1,)),
                                       ('session', (-1,))],
                         indices=['gesture', 'subject', 'session']),
            di.PrevMixin,
            base.BaseIter
        ):
            pass

        return cls


@utils.cached
def _load(path, steps, expand, preprocess):
    semg = utils.load_csl_semg(path)
    data = list(Context.parallel(jb.delayed(_load_aux)(segment, steps, expand, preprocess)
                                 for segment in semg))
    return data


def _load_aux(semg, steps, expand, preprocess):
    from ...data.preprocess import CSLBandpass, _csl_cut, Median3x3
    semg = CSLBandpass()(semg, **Dataset.get_preprocess_kargs())
    mask = np.zeros(len(semg), dtype=np.bool)
    begin, end = _csl_cut(semg, Dataset.framerate)
    mask[begin:end] = True

    semg = Median3x3()(semg, **Dataset.get_preprocess_kargs())
    prev = utils.get_semg_prev(semg, steps)

    begin, end = _expand(begin, end, len(semg), expand)
    semg = semg[begin:end].copy()
    prev = prev[begin:end].copy()
    mask = mask[begin:end].copy()

    if preprocess:
        semg, prev, mask = preprocess((semg, prev, mask))

    return semg, prev, mask


def _expand(begin, end, n, expand):
    if isinstance(expand, (tuple, list)):
        before, after = expand
    else:
        before, after = expand, expand

    span = end - begin
    before = _as_int(before, span)
    after = _as_int(after, span)

    begin = max(0, begin - before)
    end = min(n, end + after)
    return begin, end


def _as_int(x, span):
    if isinstance(x, six.string_types):
        assert x.endswith('%')
        x = int(round(span * float(x[:-1]) / 100))
    return x
