from __future__ import division
import mxnet as mx
from nose.tools import (
    assert_less,
    assert_greater,
    assert_is_instance,
    assert_equal
)
from .. import constant
from ..re_scan import Scanner


def get_get_symbol(network):
    import importlib
    return importlib.import_module('.symbol_' + network,
                                   package=__package__).get_symbol


def compile(context, text, data):
    '''Shortcut function'''
    return parse(context, text).compile(context, data)


class Q(object):

    def __init__(self, data):
        self.data = data
        self.begin = 0
        self.end = len(data)

    @property
    def current(self):
        return self.begin

    def forward(self):
        assert_less(self.begin, self.end)
        token, match = self.data[self.begin]
        self.begin += 1
        return token, match

    def backward(self):
        assert_greater(self, self.begin, 0)
        self.begin -= 1

    def __bool__(self):
        return self.begin < self.end

    __nonzero__ = __bool__


def parse(context, text):
    '''
    fc256?fc256?fc128
    conv64x2 lc64x2? (fc512?)x2 fc128
    '''
    return Sequence.parse(context, Q(list(tokenize(text))))


def parse_operators(context, q, previous=None, **kargs):
    operators = []
    while q:
        ret = Operator.parse(context, q, previous=operators, **kargs)
        if ret is None:
            break
        if isinstance(ret, list):
            if ret == operators:
                break
            operators = ret
        else:
            operators.append(ret)
    return operators


NAME_PATTERN = '(?::([_\w]+))'


def tokenize(text):
    scanner = Scanner([
        ('space', r'\s+'),
        ('comma', r','),
        ('group_left', r'\('),
        ('group_right', r'\)'),
        ('conv', r'conv(\d+)'),
        ('fc', r'fc(\d+)'),
        ('out_fc', r'o(\d+)'),
        ('last_fc', r'\$(\d+)'),
        ('inner_product', r'ip(\d+)'),
        ('grid', r'grid(\d+)'),
        ('lc', r'lc(\d+)'),
        ('bng', 'bng'),
        ('batch_norm', 'bn%s?' % NAME_PATTERN),
        ('dropout', r'\?'),
        ('act', r'!'),
        ('block_grad', r'bg'),
        ('scale_grad', r'sg([\.\d]+)'),
        ('repeat', r'x(\d+)'),
        ('scale_learning_rate', 'lr([\.\d]+)'),
        ('scale_weight_decay', 'wd([\.\d]+)'),
        ('fixbn', 'fixbn'),
        ('fix', 'fix'),
        ('share', 'share'),
        ('tee', 'tee'),
        ('identity', 'id%s?' % NAME_PATTERN)
    ])
    for token, match in scanner.scan(text):
        if token not in ['space', 'comma']:
            yield token, match


class OperatorMeta(type):

    impls = []

    def __init__(cls, name, bases, fields):
        type.__init__(cls, name, bases, fields)
        OperatorMeta.impls.append(cls)


class Operator(object):

    __metaclass__ = OperatorMeta

    @classmethod
    def parse(cls, context, q, **kargs):
        if cls is Operator:
            for impl in OperatorMeta.impls:
                if impl is not Operator:
                    inst = impl.parse(context, q, **kargs)
                    if inst is not None:
                        return inst

    def get_last_name(self, context):
        return context.get('names', [None])[-1]

    def set_last_name(self, context, name):
        names = context.get('names', None)
        if names is None:
            names = context['names'] = [None]
        names.append(name)


class Group(Operator):

    @classmethod
    def parse(cls, context, q, **kargs):
        token, match = q.forward()
        if token == 'group_left':
            inst = cls()
            inst.operators = parse_operators(context, q, **kargs)
            token, match = q.forward()
            if token != 'group_right':
                raise Exception('Missing ")" at {}'.format(q.current))
            return inst
        q.backward()

    def compile(self, context, data):
        net = data
        for operator in self.operators:
            net = operator.compile(context, net)
        return net

    def encode(self, context):
        return '(' + ''.join(op.encode(context) for op in self.operators) + ')'


class Repeat(Operator):

    @classmethod
    def parse(cls, context, q, previous, **kargs):
        token, match = q.forward()
        if token == 'repeat':
            inst = cls()
            inst.times = int(match.group(1))
            assert_is_instance(previous, list)
            inst.operator = previous[-1]
            return previous[:-1] + [inst]
        q.backward()

    def compile(self, context, data):
        net = data
        for i in range(self.times):
            net = self.operator.compile(context, net)
        return net

    def encode(self, context):
        return '{}x{}'.format(self.operator.encode(context), self.times)


class Convolution(Operator):

    @classmethod
    def parse(cls, context, q, **kargs):
        token, match = q.forward()
        if token == 'conv':
            inst = cls()
            inst.num_filter = int(match.group(1))
            return inst
        q.backward()

    def format_name(self, index):
        return 'conv%d' % index

    def compile(self, context, data):
        name = context['gen_next_name'](self)
        net = context['get_conv'](
            name=name,
            data=data,
            num_filter=self.num_filter,
            kernel=(3, 3),
            stride=(1, 1),
            pad=(1, 1),
            no_bias=True,
            context=context
        )
        net = context['get_act'](
            name=name,
            data=net,
            context=context
        )
        self.set_last_name(context, name)
        return net

    def encode(self, context):
        return 'conv%d' % self.num_filter


class FullyConnected(Operator):

    @classmethod
    def parse(cls, context, q, **kargs):
        token, match = q.forward()
        if token == 'fc':
            inst = cls()
            inst.num_hidden = int(match.group(1))
            return inst
        q.backward()

    def format_name(self, index):
        return 'fc%d' % index

    def compile(self, context, data):
        name = context['gen_next_name'](self)
        net = context['get_fc'](
            name=name,
            data=data,
            num_hidden=self.num_hidden,
            no_bias=True,
            context=context
        )
        net = context['get_act'](
            name=name,
            data=net,
            context=context
        )
        self.set_last_name(context, name)
        return net

    def encode(self, context):
        return 'fc%d' % self.num_hidden


class OutputFullyConnected(Operator):

    @classmethod
    def parse(cls, context, q, **kargs):
        token, match = q.forward()
        if token == 'out_fc':
            inst = cls()
            inst.num_hidden = int(match.group(1))
            return inst
        q.backward()

    def format_name(self, index):
        return 'out_fc%d' % index

    def compile(self, context, data):
        name = context['gen_next_name'](self)
        net = context['get_fc'](
            name=name,
            data=data,
            num_hidden=self.num_hidden,
            no_bias=False,
            context=context
        )
        self.set_last_name(context, name)
        return net

    def encode(self, context):
        return 'o%d' % self.num_hidden


class LastFullyConnected(Operator):

    @classmethod
    def parse(cls, context, q, **kargs):
        token, match = q.forward()
        if token == 'last_fc':
            inst = cls()
            inst.num_hidden = int(match.group(1))
            return inst
        q.backward()

    def format_name(self, index):
        assert_equal(index, 1)
        return 'last_fc'

    def compile(self, context, data):
        name = context['gen_next_name'](self)
        net = context['get_fc'](
            name=name,
            data=data,
            num_hidden=self.num_hidden,
            no_bias=False,
            context=context
        )
        self.set_last_name(context, name)
        return net

    def encode(self, context):
        return '$%d' % self.num_hidden


class InnerProduct(Operator):

    @classmethod
    def parse(cls, context, q, **kargs):
        token, match = q.forward()
        if token == 'inner_product':
            inst = cls()
            inst.num_hidden = int(match.group(1))
            return inst
        q.backward()

    def format_name(self, index):
        return 'ip%d' % index

    def compile(self, context, data):
        name = context['gen_next_name'](self)
        net = context['get_fc'](
            name=name,
            data=data,
            num_hidden=self.num_hidden,
            no_bias=True,
            context=context
        )
        self.set_last_name(context, name)
        return net

    def encode(self, context):
        return 'ip%d' % self.num_hidden


class GridFullyConnected(Operator):

    @classmethod
    def parse(cls, context, q, **kargs):
        token, match = q.forward()
        if token == 'grid':
            inst = cls()
            inst.num_hidden = int(match.group(1))
            return inst
        q.backward()

    def format_name(self, index):
        return 'grid%d' % index

    def compile(self, context, data):
        name = context['gen_next_name'](self)
        net = context['get_grid_fc'](
            name=name,
            data=data,
            num_hidden=self.num_hidden,
            no_bias=True,
            context=context
        )
        self.set_last_name(context, name)
        return net

    def encode(self, context):
        return 'grid%d' % self.num_hidden


class LocallyConnected(Operator):

    @classmethod
    def parse(cls, context, q, **kargs):
        token, match = q.forward()
        if token == 'lc':
            inst = cls()
            inst.num_filter = int(match.group(1))
            return inst
        q.backward()

    def format_name(self, index):
        return 'lc%d' % index

    def compile(self, context, data):
        name = context['gen_next_name'](self)
        net = context['get_lc'](
            name=name,
            data=data,
            num_filter=self.num_filter,
            no_bias=True,
            context=context
        )
        net = context['get_act'](
            name=name,
            data=net,
            context=context
        )
        self.set_last_name(context, name)
        return net

    def encode(self, context):
        return 'lc%d' % self.num_filter


class BatchNorm(Operator):

    @classmethod
    def parse(cls, context, q, **kargs):
        token, match = q.forward()
        if token == 'batch_norm':
            inst = cls()
            inst.name = match.group(1)
            return inst
        q.backward()

    def compile(self, context, data):
        return context['get_bn'](
            name=self.name,
            data=data,
            context=context
        )

    def encode(self, context):
        return 'bn' if self.name is None else 'bn:' + self.name


class Dropout(Operator):

    @classmethod
    def parse(cls, context, q, **kargs):
        token, match = q.forward()
        if token == 'dropout':
            inst = cls()
            inst.p = context.get('dropout', constant.DROPOUT)
            return inst
        q.backward()

    def compile(self, context, data):
        if self.p == 0:
            return data

        name = context['gen_next_dropout_name']()
        net = mx.symbol.Dropout(
            name=name,
            data=data,
            p=self.p
        )
        #  self.set_last_name(context, name)
        return net

    def encode(self, context):
        text = '?'
        if self.p != context.get('dropout', constant.DROPOUT):
            text += str(self.p)
        return text


class Activation(Operator):

    @classmethod
    def parse(cls, context, q, **kargs):
        token, match = q.forward()
        if token == 'act':
            return cls()
        q.backward()

    def compile(self, context, data):
        return context['get_act'](
            name=self.get_last_name(context),
            data=data
        )

    def encode(self, context):
        return '!'


class BlockGrad(Operator):

    @classmethod
    def parse(cls, context, q, **kargs):
        token, match = q.forward()
        if token == 'block_grad':
            inst = cls()
            return inst
        q.backward()

    def compile(self, context, data):
        net = mx.symbol.BlockGrad(data=data)
        return net

    def encode(self, context):
        return 'bg'


class ScaleGrad(Operator):

    @classmethod
    def parse(cls, context, q, **kargs):
        token, match = q.forward()
        if token == 'scale_grad':
            inst = cls()
            inst.scale_text = match.group(1)
            inst.scale = float(inst.scale_text)
            return inst
        q.backward()

    def compile(self, context, data):
        opname = 'ScaleGrad({})'.format(self.scale)
        if not context['operator_registered'](opname):
            self._register_operator(opname)
        return mx.symbol.Custom(data=data, op_type=opname)

    def encode(self, context):
        return 'sg{}'.format(self.scale_text)

    def _register_operator(self, opname):
        scale = self.scale

        class ScaleGrad(mx.operator.CustomOp):

            def forward(self, is_train, req, in_data, out_data, aux):
                self.assign(out_data[0], req[0], 0 + in_data[0])

            def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
                self.assign(
                    in_grad[0],
                    req[0],
                    scale * out_grad[0]
                )

        @mx.operator.register(opname)
        class ScaleGradProp(mx.operator.CustomOpProp):

            def __init__(self):
                super(ScaleGradProp, self).__init__(need_top_grad=True)

            def infer_shape(self, in_shape):
                data_shape = in_shape[0]
                output_shape = in_shape[0]
                return [data_shape], [output_shape], []

            def create_operator(self, ctx, shapes, dtypes):
                return ScaleGrad()


class ScaleLearningRate(Operator):

    @classmethod
    def parse(cls, context, q, **kargs):
        token, match = q.forward()
        if token == 'scale_learning_rate':
            inst = cls()
            inst.scale_text = match.group(1)
            inst.scale = float(inst.scale_text)
            inst.operator = Operator.parse(context, q, **kargs)
            if not inst.operator:
                raise Exception('Parse ScaleLearningRate at {} failed'.format(q.begin))
            return inst
        q.backward()

    def compile(self, context, data):
        context = self._push_context(context)
        return self.operator.compile(context, data)

    def encode(self, context):
        return 'lr{}{}'.format(self.scale_text, self.operator.encode(context))

    def _push_context(self, context):
        context = context.copy()
        context['lr_mult'] = self.scale * context.get('lr_mult', 1)
        return context


class ScaleWeightDecay(Operator):

    @classmethod
    def parse(cls, context, q, **kargs):
        token, match = q.forward()
        if token == 'scale_weight_decay':
            inst = cls()
            inst.scale_text = match.group(1)
            inst.scale = float(inst.scale_text)
            inst.operator = Operator.parse(context, q, **kargs)
            if not inst.operator:
                raise Exception('Parse ScaleWeightDecay at {} failed'.format(q.begin))
            return inst
        q.backward()

    def compile(self, context, data):
        context = self._push_context(context)
        return self.operator.compile(context, data)

    def encode(self, context):
        return 'wd{}{}'.format(self.scale_text, self.operator.encode(context))

    def _push_context(self, context):
        context = context.copy()
        context['wd_mult'] = self.scale * context.get('wd_mult', 1)
        return context


class FixBN(Operator):

    @classmethod
    def parse(cls, context, q, **kargs):
        token, match = q.forward()
        if token == 'fixbn':
            inst = cls()
            inst.operator = Operator.parse(context, q, **kargs)
            if not inst.operator:
                raise Exception('Parse FixBN at {} failed'.format(q.begin))
            return inst
        q.backward()

    def compile(self, context, data):
        context = self._push_context(context)
        return self.operator.compile(context, data)

    def encode(self, context):
        return 'fixbn' + self.operator.encode(context)

    def _push_context(self, context):
        context = context.copy()
        context['fix_batch_norm'] = True
        return context


class Fix(Operator):

    @classmethod
    def parse(cls, context, q, **kargs):
        token, match = q.forward()
        if token == 'fix':
            inst = cls()
            inst.operator = Operator.parse(context, q, **kargs)
            if not inst.operator:
                raise Exception('Parse Fix at {} failed'.format(q.begin))
            return inst
        q.backward()

    def compile(self, context, data):
        context = self._push_context(context)
        return self.operator.compile(context, data)

    def encode(self, context):
        return 'fix' + self.operator.encode(context)

    def _push_context(self, context):
        context = context.copy()
        context['lr_mult'] = 0
        context['fix_batch_norm'] = True
        return context


class Bng(Operator):

    @classmethod
    def parse(cls, context, q, **kargs):
        token, match = q.forward()
        if token == 'bng':
            inst = cls()
            inst.operator = Operator.parse(context, q, **kargs)
            if not inst.operator:
                raise Exception('Parse Bng at {} failed'.format(q.begin))
            return inst
        q.backward()

    def compile(self, context, data):
        context = self._push_context(context)
        return self.operator.compile(context, data)

    def encode(self, context):
        return 'bng' + self.operator.encode(context)

    def _push_context(self, context):
        context = context.copy()
        context['batch_norm_use_global_stats'] = True
        return context


class Share(Operator):

    @classmethod
    def parse(cls, context, q, **kargs):
        token, match = q.forward()
        if token == 'share':
            inst = cls()
            inst.operator = Operator.parse(context, q, **kargs)
            if not inst.operator:
                raise Exception('Parse Share at {} failed'.format(q.begin))
            return inst
        q.backward()

    def compile(self, context, data):
        data, unpack = context['pack'](data)
        return unpack(self.operator.compile(context, data))

    def encode(self, context):
        return 'share' + self.operator.encode(context)


class Tee(Operator):

    @classmethod
    def parse(cls, context, q, **kargs):
        token, match = q.forward()
        if token == 'tee':
            inst = cls()
            inst.operator = Operator.parse(context, q, **kargs)
            if not inst.operator:
                raise Exception('Parse Tee at {} failed'.format(q.begin))
            return inst
        q.backward()

    def compile(self, context, data):
        return [self.operator.compile(context, branch) for branch in data]

    def encode(self, context):
        return 'tee' + self.operator.encode(context)


class Identity(Operator):

    class Op(mx.operator.CustomOp):

        def forward(self, is_train, req, in_data, out_data, aux):
            self.assign(out_data[0], req[0], 0 + in_data[0])

        def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
            self.assign(in_grad[0], req[0], 0 + out_grad[0])

    @mx.operator.register('Identity')
    class Prop(mx.operator.CustomOpProp):

        def __init__(self):
            super(Identity.Prop, self).__init__(need_top_grad=True)

        def infer_shape(self, in_shape):
            data_shape = in_shape[0]
            output_shape = in_shape[0]
            return [data_shape], [output_shape], []

        def create_operator(self, ctx, shapes, dtypes):
            return Identity.Op()

    @classmethod
    def parse(cls, context, q, **kargs):
        token, match = q.forward()
        if token == 'identity':
            inst = cls()
            inst.name = match.group(1)
            return inst
        q.backward()

    def compile(self, context, data):
        return mx.symbol.Custom(data=data, name=self.name, op_type='Identity')

    def encode(self, context):
        return 'id' if self.name is None else 'id:' + self.name


class Sequence(Operator):
    '''This class must be defined at the end of this file'''

    @classmethod
    def parse(cls, context, q, **kargs):
        top = q.begin == 0
        token, match = q.forward()
        has_left = token == 'sequence_left'
        if top or has_left:
            if not has_left:
                q.backward()
            inst = cls()
            inst.operators = parse_operators(context, q, **kargs)
            has_right = False
            if has_left:
                token, match = q.forward()
                has_right = token == 'sequence_right'
            if not top and not has_right:
                raise Exception('Missing "]" at {}'.format(q.current))
            return inst
        q.backward()

    def compile(self, context, data):
        net = data
        context = self._push_context(context)
        for operator in self.operators:
            net = operator.compile(context, net)
        return net

    def encode(self, context):
        return '[' + ''.join(op.encode(context) for op in self.operators) + ']'

    def _push_context(self, context):
        context = context.copy()

        def gen_next_name(operator):
            index = context['counter'](type(operator))
            return add_prefix(operator.format_name(index=index))

        def gen_next_dropout_name():
            last_name = self.get_last_name(context)
            return last_name + '_drop' if last_name else None

        def add_prefix(name):
            return prefix + '_' + name if prefix else name

        def counter(operator):
            memo = getattr(counter, 'memo', None)
            if memo is None:
                memo = counter.memo = {}
            if operator in memo:
                memo[operator] += 1
            else:
                memo[operator] = 1
            return memo[operator]

        prefix = context.pop('prefix', '')
        context['counter'] = counter
        context['gen_next_name'] = gen_next_name
        context['gen_next_dropout_name'] = gen_next_dropout_name
        context['names'] = [None]
        return context
