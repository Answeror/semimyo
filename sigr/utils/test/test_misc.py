import nose.tools as nt


def test_restore_field():
    from .. import restore_field, Bunch
    obj = Bunch(foo=1, bar=2)
    with restore_field(obj, foo=3, bar=4):
        nt.assert_equal(obj.foo, 3)
        nt.assert_equal(obj.bar, 4)
    nt.assert_equal(obj.foo, 1)
    nt.assert_equal(obj.bar, 2)
