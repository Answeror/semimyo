from __future__ import division
import nose.tools as nt


def downsample_even_segment(step, segment):
    nt.assert_greater(step, 1)
    ignored = {s: 0 for s in set(segment)}
    index = []
    for i, s in enumerate(segment):
        if ignored[s] == 0:
            ignored[s] = step
            index.append(i)
        ignored[s] -= 1
    return index
