from __future__ import division
import numpy as np
import scipy.io as sio
from logbook import Logger
from nose.tools import assert_equal
import os


logger = Logger(__name__)


def get_ninapro_db1_semg_pose_data(paths, preprocess, tag):
    return [_get_ninapro_db1_semg_pose_data(path, preprocess, tag) for path in paths]


def _get_ninapro_db1_semg_pose_data(path, preprocess, tag):
    logger.debug('Load {}', path)
    semg = sio.loadmat(path)['emg'].astype(np.float32)
    pose = sio.loadmat(_get_pose_path(path, tag))['pose'].astype(np.int32).ravel()
    assert_equal(len(semg), len(pose))
    assert_equal(pose.ndim, 1)
    gesture = sio.loadmat(path)['restimulus'].astype(np.uint8).ravel()
    if preprocess:
        (semg, pose, gesture) = preprocess((semg, pose, gesture),
                                           framerate=100,
                                           num_semg_row=1,
                                           num_semg_col=10)
    return semg, pose, gesture


def _get_pose_path(path, tag):
    parts = path.split(os.sep)
    return os.sep.join(parts[:-2] + ['pose-' + tag] + parts[-2:])


def get_ninapro_db1_semg_glove_data(paths, preprocess):
    return [_get_ninapro_db1_semg_glove_data(path, preprocess) for path in paths]


def _get_ninapro_db1_semg_glove_data(path, preprocess):
    logger.debug('Load {}', path)
    semg = sio.loadmat(path)['emg'].astype(np.float32)
    glove = sio.loadmat(path)['glove'].astype(np.float32)
    gesture = sio.loadmat(path)['restimulus'].astype(np.uint8).ravel()
    if preprocess:
        (semg, glove, gesture) = preprocess((semg, glove, gesture),
                                            framerate=100,
                                            num_semg_row=1,
                                            num_semg_col=10)
    return semg, glove, gesture


def get_ninapro_db1_semg_position_data(paths, preprocess):
    return [_get_ninapro_db1_semg_position_data(path, preprocess) for path in paths]


def _get_ninapro_db1_semg_position_data(path, preprocess):
    logger.debug('Load {}', path)
    semg = sio.loadmat(path)['emg'].astype(np.float32)
    position = sio.loadmat(_get_extension_path(path, 'position'))['position'].astype(np.float32)
    assert_equal(len(semg), len(position))
    assert_equal(position.ndim, 2)
    gesture = sio.loadmat(path)['restimulus'].astype(np.uint8).ravel()
    if preprocess:
        (semg, position, gesture) = preprocess((semg, position, gesture),
                                               framerate=100,
                                               num_semg_row=1,
                                               num_semg_col=10)
    return semg, position, gesture


def _get_extension_path(path, tag):
    parts = path.split(os.sep)
    return os.sep.join(parts[:-2] + [tag] + parts[-2:])


def get_ninapro_db1_semg_angle_data(paths, preprocess):
    return [_get_ninapro_db1_semg_angle_data(path, preprocess) for path in paths]


def _get_ninapro_db1_semg_angle_data(path, preprocess):
    logger.debug('Load {}', path)
    semg = sio.loadmat(path)['emg'].astype(np.float32)
    angle = sio.loadmat(_get_extension_path(path, 'angle'))['angle'].astype(np.float32)
    angle = angle[:, :19]
    assert_equal(len(semg), len(angle))
    assert_equal(angle.ndim, 2)
    gesture = sio.loadmat(path)['restimulus'].astype(np.uint8).ravel()
    if preprocess:
        (semg, angle, gesture) = preprocess((semg, angle, gesture),
                                            framerate=100,
                                            num_semg_row=1,
                                            num_semg_col=10)
    return semg, angle, gesture


def get_ninapro_db1_semg_data(paths, preprocess):
    return [_get_ninapro_db1_semg_data(path, preprocess) for path in paths]


def _get_ninapro_db1_semg_data(path, preprocess):
    logger.debug('Load {}', path)
    semg = sio.loadmat(path)['emg'].astype(np.float32)
    gesture = sio.loadmat(path)['restimulus'].astype(np.uint8).ravel()
    if preprocess:
        (semg, gesture) = preprocess((semg, gesture),
                                     framerate=100,
                                     num_semg_row=1,
                                     num_semg_col=10)
    return semg, gesture


def get_ninapro_db1_semg_binary_data(paths, preprocess):
    return [_get_ninapro_db1_semg_binary_data(path, preprocess) for path in paths]


def _get_ninapro_db1_semg_binary_data(path, preprocess):
    logger.debug('Load {}', path)
    semg = sio.loadmat(path)['emg'].astype(np.float32)
    binary = sio.loadmat(_get_extension_path(path, 'binary'))['binary'].astype(np.int32)
    assert_equal(len(semg), len(binary))
    assert_equal(binary.ndim, 2)
    gesture = sio.loadmat(path)['restimulus'].astype(np.uint8).ravel()
    if preprocess:
        (semg, binary, gesture) = preprocess((semg, binary, gesture),
                                             framerate=100,
                                             num_semg_row=1,
                                             num_semg_col=10)
    return semg, binary, gesture
