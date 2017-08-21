from __future__ import division
import mxnet as mx


class ProxyDataIter(object):

    def __init__(self, base):
        self.base = base

    def __iter__(self):
        return self

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __setattr__(self, name, value):
        if name not in ['base'] and hasattr(self.base, name):
            return setattr(self.base, name, value)
        return super(ProxyDataIter, self).__setattr__(name, value)

    def reset(self):
        return self.base.reset()

    def next(self):
        """Get next data batch from iterator. Equivalent to
        self.iter_next()
        DataBatch(self.getdata(), self.getlabel(), self.getpad(), None)

        Returns
        -------
        data : DataBatch
            The data of next batch.
        """
        if self.iter_next():
            return mx.io.DataBatch(data=self.getdata(), label=self.getlabel(),
                                   pad=self.getpad(), index=self.getindex())
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def iter_next(self):
        return self.base.iter_next()

    @property
    def batch_size(self):
        return self.base.batch_size

    @property
    def provide_data(self):
        return [item for i, item in enumerate(self.base.provide_data)
                if i not in self.ignore_data]

    @property
    def provide_label(self):
        return [item for i, item in enumerate(self.base.provide_label)
                if i not in self.ignore_label]

    def getdata(self):
        return self.base.getdata()

    def getlabel(self):
        return self.base.getlabel()

    def getindex(self):
        return self.base.getindex()

    def getpad(self):
        return self.base.getpad()


class IgnoreInputDataIter(ProxyDataIter):

    def __init__(self, base, ignore_data=[], ignore_label=[]):
        super(IgnoreInputDataIter, self).__init__(base)

        self.ignore_data = []
        for i, (name, _) in enumerate(base.provide_data):
            if name in ignore_data:
                self.ignore_data.append(i)
        self.ignore_label = []
        for i, (name, _) in enumerate(base.provide_label):
            if name in ignore_label:
                self.ignore_label.append(i)

    @property
    def provide_data(self):
        return [item for i, item in enumerate(self.base.provide_data)
                if i not in self.ignore_data]

    @property
    def provide_label(self):
        return [item for i, item in enumerate(self.base.provide_label)
                if i not in self.ignore_label]

    def getdata(self):
        return [item for i, item in enumerate(self.base.getdata())
                if i not in self.ignore_data]

    def getlabel(self):
        return [item for i, item in enumerate(self.base.getlabel())
                if i not in self.ignore_label]
