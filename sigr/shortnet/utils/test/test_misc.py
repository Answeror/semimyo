import nose.tools as nt


def test_get():
    from .. import get

    class cls(object):
        foo = 1

    nt.assert_equal(get({'foo': 1}, 'foo')(), 1)
    nt.assert_equal(get(cls(), 'foo')(), 1)
    nt.assert_equal(get({'foo': 1}, 'foo').value(), 1)
    nt.assert_equal(get({'foo': 1}, 'bar')({'bar': 2}, 'bar')(), 2)
    nt.assert_equal(get({'foo': 1}, 'bar')({'bar': 2})(), 2)
    nt.assert_equal(get({'foo': 1}, 'foobar')({'bar': 2}).default(3), 3)
    nt.assert_equal(get({'foo': 1}, 'foobar')({'bar': 2})(default=3), 3)
    nt.assert_raises(KeyError, get({'foo': 1}, 'foobar')({'bar': 2}))
