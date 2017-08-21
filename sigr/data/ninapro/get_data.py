from __future__ import division
import numpy as np
import scipy.io as sio
from logbook import Logger
from nose.tools import assert_equal
import os
from ... import utils


logger = Logger(__name__)


@utils.cached
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


#  @utils.cached
def get_ninapro_db1_semg_latent_data(paths, preprocess, num_latent):
    return [_get_ninapro_db1_semg_latent_data(path, preprocess, num_latent) for path in paths]


def _get_ninapro_db1_semg_latent_data(path, preprocess, num_latent):
    logger.debug('Load {}', path)
    semg = sio.loadmat(path)['emg'].astype(np.float32)
    latent = sio.loadmat(_get_latent_path(path, num_latent))['latent'].astype(np.int32).ravel()
    assert_equal(len(semg), len(latent))
    assert_equal(latent.ndim, 1)
    gesture = sio.loadmat(path)['restimulus'].astype(np.uint8).ravel()
    if preprocess:
        (semg, latent, gesture) = preprocess((semg, latent, gesture),
                                             framerate=100,
                                             num_semg_row=1,
                                             num_semg_col=10)
    return semg, latent, gesture


def _get_latent_path(path, num_latent):
    parts = path.split(os.sep)
    return os.sep.join(parts[:-2] + ['latent-%d' % num_latent] + parts[-2:])


def get_ninapro_db1_semg_pose_data(paths, preprocess, num_pose):
    return [_get_ninapro_db1_semg_pose_data(path, preprocess, num_pose) for path in paths]


def _get_ninapro_db1_semg_pose_data(path, preprocess, num_pose):
    logger.debug('Load {}', path)
    semg = sio.loadmat(path)['emg'].astype(np.float32)
    pose = sio.loadmat(_get_pose_path(path, num_pose))['pose'].astype(np.int32).ravel()
    assert_equal(len(semg), len(pose))
    assert_equal(pose.ndim, 1)
    gesture = sio.loadmat(path)['restimulus'].astype(np.uint8).ravel()
    if preprocess:
        (semg, pose, gesture) = preprocess((semg, pose, gesture),
                                             framerate=100,
                                             num_semg_row=1,
                                             num_semg_col=10)
    return semg, pose, gesture


def _get_pose_path(path, num_pose):
    parts = path.split(os.sep)
    return os.sep.join(parts[:-2] + ['pose-%d' % num_pose] + parts[-2:])
