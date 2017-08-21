from __future__ import division


def make_lru_dict(room):
    try:
        from lru import LRU
        return LRU(room)
    except:
        return {}
