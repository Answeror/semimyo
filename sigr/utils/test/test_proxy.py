import joblib as jb
import pickle
import io
from ..proxy import LazyProxy


def nocall():
    assert False


def test_proxy():
    inst = LazyProxy(nocall)
    pickle.Pickler(io.BytesIO(), pickle.HIGHEST_PROTOCOL).dump(inst)
    pickle.Pickler(io.BytesIO()).dump(inst)
    jb.hash(inst)
