from nose.tools import assert_equal
from .symbol_common import parse
from . import symbol_base
from .. import _mxnet_operator_registered


def test_parse_simple():
    context = {}
    text = 'fc512'
    assert_equal('[%s]' % text, parse(context, text).encode(context))


def test_parse_bn():
    context = {}
    text = 'bn:zscore'
    assert_equal('[%s]' % text, parse(context, text).encode(context))


def test_parse_sequence():
    context = {}
    text = 'fc512fc512'
    assert_equal('[%s]' % text, parse(context, text).encode(context))


def test_parse_repeat():
    context = {}
    text = 'fc512x2'
    assert_equal('[%s]' % text, parse(context, text).encode(context))


def test_parse_complex():
    context = dict(dropout=0.5)
    text = 'fc512?fc512?fc128'
    assert_equal('[%s]' % text, parse(context, text).encode(context))
    context = dict(dropout=0)
    assert_equal('[%s]' % text, parse(context, text).encode(context))

    context = dict(dropout=0.5)
    text = 'conv64x2 lc64x2? (fc512?)x2 fc128'
    encoded = text.replace(' ', '')
    assert_equal('[%s]' % encoded, parse(context, text).encode(context))
    context = dict(dropout=0)
    assert_equal('[%s]' % encoded, parse(context, text).encode(context))

    context = dict(dropout=0.5)
    text = 'bg,lr0.1(lr0.1(lr0.1(conv64x2),lc64x2?),(fc512?)x2),fc128'
    encoded = text.replace(',', '')
    assert_equal('[%s]' % encoded, parse(context, text).encode(context))
    context = dict(dropout=0)
    assert_equal('[%s]' % encoded, parse(context, text).encode(context))


def test_compile():
    import mxnet as mx

    class Symbol(symbol_base.Symbol):

        def __init__(self, text):
            super(Symbol, self).__init__()

            self.context = dict(
                dropout=0.5,
                get_act=self.get_bn_relu,
                get_bn=self.get_bn,
                get_fc=self.get_fc,
                get_lc=self.get_lc,
                get_conv=self.get_conv,
                operator_registered=lambda name: name in _mxnet_operator_registered
            )
            data = mx.symbol.Variable('data')
            self.net = self.get_one_line_net(self.context, text, data)

        def get_bn_relu(self, data, name, context):
            if name == 'fc3':
                assert 'lr_mult' not in context
                assert 'fix_batch_norm' not in context
            else:
                assert_equal(context['lr_mult'], 0)
                assert context['fix_batch_norm']

            return super(Symbol, self).get_bn_relu(
                data=data,
                name=name,
                context=context
            )

        def infer_shape(self, data):
            net = data
            data_shape = (1, 1, 1, 10)
            return tuple(int(s) for s in net.infer_shape(data=data_shape)[1][0])

    Symbol('fix(bn:zscore,conv64x2,lc64x2?,(fc512?)x2),fc128').net


#  def test_compile():
    #  context = {}
    #  text = 'fc512?'
    #  parse(context, text).compile(context, 'data')
