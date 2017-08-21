from __future__ import division
import numpy as np
from .. import utils


def get_amplitude_weight(data, segment, framerate):
    from .. import Context
    import joblib as jb
    indices = [np.where(segment == i)[0] for i in set(segment)]
    w = np.empty(len(segment), dtype=np.float)
    for i, ret in zip(
        indices,
        Context.parallel(jb.delayed(get_amplitude_weight_aux)(data[i], framerate)
                         for i in indices)
    ):
        w[i] = ret
    return w / max(w.sum(), 1e-8)


def get_amplitude_weight_aux(data, framerate):
    return _get_amplitude_weight_aux(data, framerate)


@utils.cached
def _get_amplitude_weight_aux(data, framerate):
    # High-Density Electromyography and Motor Skill Learning for Robust Long-Term Control of a 7-DoF Robot Arm
    lowpass = utils.butter_lowpass_filter
    shape = data.shape
    data = np.abs(data.reshape(shape[0], -1))
    data = np.transpose([lowpass(ch, 3, framerate, 4, zero_phase=True) for ch in data.T])
    data = data.mean(axis=1)
    data -= data.min()
    data /= max(data.max(), 1e-8)
    return data
