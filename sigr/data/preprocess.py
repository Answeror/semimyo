from __future__ import division
import re
import numpy as np
from nose.tools import assert_less_equal
from ..utils import cached, butter_lowpass_filter as lowpass
from .. import Context
import joblib as jb
from scipy.ndimage.filters import median_filter


class Preprocess(object):

    class __metaclass__(type):

        def __init__(cls, name, bases, fields):
            type.__init__(cls, name, bases, fields)
            if name != 'Preprocess':
                Preprocess.register(cls)

    impls = []

    def __call__(self, data, **kargs):
        return data

    @classmethod
    def parse(cls, text):
        if not text:
            return None
        if cls is Preprocess:
            for impl in cls.impls:
                inst = impl.parse(text)
                if inst is not None:
                    return inst

    @classmethod
    def register(cls, impl):
        cls.impls.append(impl)


class Identity(Preprocess):

    @classmethod
    def parse(cls, text):
        if text == 'identity':
            return cls()

    def __call__(self, data, **kargs):
        return data

    def __repr__(self):
        return 'Identity()'


class Branch(Preprocess):

    @classmethod
    def parse(cls, text):
        matched = re.search('{(.+)}', text)
        if matched:
            sep = ';' if ';' in text else ','
            return cls([Preprocess.parse(branch) for branch
                        in matched.group(1).split(sep)])

    def __init__(self, branches):
        self.branches = branches

    def __call__(self, data, **kargs):
        branches = self.branches[:]
        if len(data) > len(branches):
            branches += [Identity()] * (len(data) - len(branches))
        return tuple(branch(data, **kargs)
                     for branch, data in zip(branches, data))

    def __repr__(self):
        return 'Branch(%s)' % ','.join(str(branch) for branch in self.branches)


class Sequence(Preprocess):

    @classmethod
    def parse(cls, text):
        matched = re.search('\((.+)\)', text)
        if matched:
            return cls([Preprocess.parse(stage) for stage
                        in matched.group(1).split(',')])

    def __init__(self, stages):
        self.stages = stages

    def __call__(self, data, **kargs):
        for stage in self.stages:
            data = stage(data, **kargs)
        return data

    def __repr__(self):
        return 'Sequence(%s)' % ','.join(str(stage) for stage in self.stages)


class Bandstop(Preprocess):

    @classmethod
    def parse(cls, text):
        if re.search('bandstop', text):
            return cls()

    def __call__(self, data, framerate, **kargs):
        from ..utils import butter_bandstop_filter as bandstop
        return np.transpose([bandstop(ch, 45, 55, framerate, 2) for ch in data.T])

    def __repr__(self):
        return 'Bandstop()'


class CSLBandpass(Preprocess):

    @classmethod
    def parse(cls, text):
        if re.search('csl-bandpass', text):
            return cls()

    def __call__(self, data, framerate, **kargs):
        from ..utils import butter_bandpass_filter as bandpass
        return np.transpose([bandpass(ch, 20, 400, framerate, 4) for ch in data.T])

    def __repr__(self):
        return 'CSLBandpass()'


class NinaProLowpass(Preprocess):

    @classmethod
    def parse(cls, text):
        if re.match('ninapro-lowpass', text):
            return cls()

    def __call__(self, data, framerate, **kargs):
        return np.transpose([lowpass(ch, 1, framerate, 1, zero_phase=True) for ch in data.T])

    def __repr__(self):
        return 'NinaProLowpass()'


class NinaProDivideChannelMean(Preprocess):

    @classmethod
    def parse(cls, text):
        if re.match('ninapro-div-chan-mean', text):
            return cls()

    def __call__(self, data, **kargs):
        return data / [0.42199532, 0.28549964, 0.27955272, 0.14442604, 0.04973563,
                       0.1179766, 0.59779046, 0.5951683, 0.26667204, 0.42599725]

    def __repr__(self):
        return 'NinaProDivideChannelMean()'


class NinaProDivideChannelMeanSubject18(Preprocess):

    @classmethod
    def parse(cls, text):
        if re.match('ninapro-div-chan-mean-sb18', text):
            return cls()

    def __call__(self, data, **kargs):
        return data / [0.70156931, 0.24842356, 0.22025822, 0.13020342, 0.03374223,
                       0.07109487, 0.37028435, 0.66385845, 0.22424404, 0.39227495]

    def __repr__(self):
        return 'NinaProDivideChannelMeanSubject18()'


class NinaProGloveZScore(Preprocess):

    @classmethod
    def parse(cls, text):
        if re.match('ninapro-glove-zscore', text):
            return cls()

    def __call__(self, data, **kargs):
        mean = np.array([134.26307581, 109.05141675, 109.10695646, 128.65405278,
                         112.77693178, 86.87949749, 86.22352834, 99.66383533,
                         102.02838089, 79.13727793, 145.08524347, 114.8055296,
                         99.75865728, 75.17377044, 141.32557992, 116.05385963,
                         112.87891035, 94.47715237, 130.54155619, 132.79932581,
                         144.11695147, 121.33460983]),
        std = np.array([36.10707151, 19.81685387, 19.4688742, 18.88220506,
                        25.26445764, 38.37749572, 23.21720221, 24.64400755,
                        35.54027659, 18.20137609, 18.86109485, 23.92231182,
                        23.36339215, 24.24276578, 28.75868323, 23.49758666,
                        43.1524527, 22.07180592, 17.39294789, 37.85575123,
                        12.06528808, 12.06210919])
        return (data - mean) / std

    def __repr__(self):
        return 'NinaProGloveZScore()'


class NinaProGloveZScoreGlobal(Preprocess):

    @classmethod
    def parse(cls, text):
        if re.match('ninapro-glove-zscore-global', text):
            return cls()

    def __call__(self, data, **kargs):
        mean = 112.55164090500675
        std = 33.165856233914077
        return (data - mean) / std

    def __repr__(self):
        return 'NinaProGloveZScoreGlobal()'


class NinaProLowpassParallel(Preprocess):

    @classmethod
    def parse(cls, text):
        if re.match('ninapro-lowpass-parallel', text):
            return cls()

    def __call__(self, data, framerate, **kargs):
        return np.transpose(list(Context.parallel(
            jb.delayed(lowpass)(ch, 1, framerate, 1, zero_phase=True)
            for ch in data.T)))

    def __repr__(self):
        return 'NinaProLowpassParallel()'


class Downsample(Preprocess):

    @classmethod
    def parse(cls, text):
        matched = re.search('downsample-(\d+)', text)
        if matched:
            return cls(int(matched.group(1)))

    def __init__(self, step):
        self.step = step

    def __call__(self, data, **kargs):
        if not isinstance(data, tuple):
            return data[::self.step].copy()
        return tuple(a[::self.step].copy() for a in data)

    def __repr__(self):
        return 'Downsample(step=%d)' % self.step


class Median3x3(Preprocess):

    @classmethod
    def parse(cls, text):
        if re.search('median', text):
            return cls()

    def __call__(self, data, num_semg_row, num_semg_col, **kargs):
        return np.array([median_filter(image, 3).ravel() for image
                         in data.reshape(-1, num_semg_row, num_semg_col)])

    def __repr__(self):
        return 'Median3x3()'


class Abs(Preprocess):

    @classmethod
    def parse(cls, text):
        if re.search('abs', text):
            return cls()

    def __call__(self, data, **kargs):
        return np.abs(data)

    def __repr__(self):
        return 'Abs()'


class RMS(Preprocess):

    @classmethod
    def parse(cls, text):
        matched = re.search('rms-(\d+)', text)
        if matched:
            return cls(int(matched.group(1)))

    def __init__(self, window):
        self.window = window

    def __call__(self, data, **kargs):
        window = min(self.window, len(data))
        return np.transpose([moving_rms(ch, window) for ch in data.T])

    def __repr__(self):
        return 'RMS(window=%d)' % self.window


class Cut(Preprocess):
    pass


class MiddleCut(Cut):

    @classmethod
    def parse(cls, text):
        matched = re.search('mid-(\d+)', text)
        if matched:
            return cls(int(matched.group(1)))

    def __init__(self, window):
        self.window = window

    def __call__(self, data, **kargs):
        if len(data) < self.window:
            return data
        begin = (len(data) - self.window) // 2
        return data[begin:begin + self.window].copy()

    def __repr__(self):
        return 'MiddleCut(window=%d)' % self.window


class PeakCut(Cut):

    @classmethod
    def parse(cls, text):
        matched = re.search('^peak-(\d+)$', text)
        if matched:
            return cls(int(matched.group(1)))

    def __init__(self, window):
        self.window = window

    def __call__(self, data, framerate, num_semg_row, num_semg_col, **kargs):
        if len(data) < self.window:
            return data

        begin = np.argmax(_get_amp(data, framerate, num_semg_row, num_semg_col)
                          [self.window // 2:-(self.window - self.window // 2 - 1)])
        assert_less_equal(begin + self.window, len(data))
        return data[begin:begin + self.window]

    def __repr__(self):
        return 'PeakCut(window=%d)' % self.window


class NinaProPeakCut(Cut):

    @classmethod
    def parse(cls, text):
        matched = re.search('^ninapro-peak-(\d+)$', text)
        if matched:
            return cls(int(matched.group(1)))

    def __init__(self, window):
        self.window = window

    def __call__(self, data, framerate, **kargs):
        if len(data) < self.window:
            return data

        begin = np.argmax(_get_ninapro_amp(data, framerate)
                          [self.window // 2:-(self.window - self.window // 2 - 1)])
        assert_less_equal(begin + self.window, len(data))
        return data[begin:begin + self.window]

    def __repr__(self):
        return 'NinaProPeakCut(window=%d)' % self.window


class CSLCut(Cut):

    @classmethod
    def parse(cls, text):
        if re.search('csl-cut', text):
            return cls()

    def __call__(self, data, framerate, **kargs):
        begin, end = _csl_cut(data, framerate)
        return data[begin:end]

    def __repr__(self):
        return 'CSLCut()'


def _csl_cut(data, framerate):
    window = int(np.round(150 * framerate / 2048))
    data = data[:len(data) // window * window].reshape(-1, 150, data.shape[1])
    rms = np.sqrt(np.mean(np.square(data), axis=1))
    rms = [median_filter(image, 3).ravel() for image in rms.reshape(-1, 24, 7)]
    rms = np.mean(rms, axis=1)
    threshold = np.mean(rms)
    mask = rms > threshold
    for i in range(1, len(mask) - 1):
        if not mask[i] and mask[i - 1] and mask[i + 1]:
            mask[i] = True
    from .. import utils
    begin, end = max(utils.continuous_segments(mask),
                     key=lambda s: (mask[s[0]], s[1] - s[0]))
    return begin * window, end * window


@cached
def _get_amp(data, framerate, num_semg_row, num_semg_col):
    data = np.abs(data)
    data = np.transpose([lowpass(ch, 2, framerate, 4, zero_phase=True) for ch in data.T])
    return [median_filter(image, 3).mean() for image in data.reshape(-1, num_semg_row, num_semg_col)]


def _get_ninapro_amp(data, framerate):
    data = np.abs(data)
    data = np.transpose([lowpass(ch, 2, framerate, 4, zero_phase=True) for ch in data.T])
    return data.mean(axis=1)


def moving_rms(a, window):
    a2 = np.square(a)
    window = np.ones(window) / window
    return np.sqrt(np.convolve(a2, window, 'valid'))
