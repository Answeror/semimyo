from __future__ import division
import os
import numpy as np
from logbook import Logger
import joblib as jb
from . import base
from ... import Context, utils


logger = Logger(__name__)


class Dataset(base.BaseDataset):

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
                for i, semg in enumerate(_load(path, self.preprocess)):
                    trial = i + 1
                    combo = combo.copy()
                    combo.trial = trial
                    if not self.dataset.good_combo(combo):
                        continue

                    self.set_memo(
                        combo,
                        self.trial_cls(
                            semg=semg,
                            subject=np.repeat(combo.subject, len(semg)),
                            session=np.repeat(combo.session, len(semg)),
                            gesture=np.repeat(combo.gesture, len(semg))
                        )
                    )

        return cls


@utils.cached
def _load(path, preprocess):
    semg = utils.load_csl_semg(path)
    data = list(Context.parallel(jb.delayed(_load_aux)(segment, preprocess)
                                 for segment in semg))
    return data


def _load_aux(semg, preprocess):
    from ...data.preprocess import Preprocess
    semg = Preprocess.parse('(csl-bandpass,csl-cut,median)')(semg, **Dataset.get_preprocess_kargs())
    if preprocess:
        semg = preprocess(semg)
    return semg
