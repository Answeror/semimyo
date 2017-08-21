from __future__ import division
import nose.tools as nt
from .context import ctx, BaseContext


def build(text, data):
    '''Shortcut function'''
    return parse(text).build(data)


class Q(object):

    def __init__(self, data):
        self.data = data
        self.begin = 0
        self.end = len(data)

    @property
    def current(self):
        return self.begin

    def forward(self):
        nt.assert_less(self.begin, self.end)
        token, match = self.data[self.begin]
        self.begin += 1
        return token, match

    def backward(self):
        nt.assert_greater(self, self.begin, 0)
        self.begin -= 1

    def __bool__(self):
        return self.begin < self.end

    __nonzero__ = __bool__


def parse(text):
    '''
    fc256?fc256?fc128
    conv64x2 lc64x2? (fc512?)x2 fc128
    '''
    return Sequence.parse(Q(list(tokenize(text))))


def parse_operators(q, previous=None, **kargs):
    operators = []
    while q:
        ret = Operator.parse(q, previous=operators, **kargs)
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
    from .utils import Scanner
    scanner = Scanner([
        ('space', r'\s+'),
        ('reshape', r'\(([-\d,]+)\)'),
        ('comma', r','),
        ('group_left', r'\('),
        ('group_right', r'\)'),
        ('conv', r'conv(\d+)'),
        ('fc', r'fc(\d+)'),
        ('fsd', r'fsd(\d+)(\?)?x(\d+)(?:@([\.\d]+))?'),
        ('lsd', r'lsd(\d+)(?:@(\d+)x(\d+))?(?:-(\d+)-(\d+))?(?:s([\.\d]+))?(\?)?x(\d+)(?:p([\.\d]+)?(\+)?)?'),
        ('sd', r'sd(\d+)x(\d+)(?:@([\.\d]+))?'),
        ('out_fc', r'o(\d+)'),
        ('last_fc', r'\$(\d+)'),
        ('inner_product', r'ip(\d+)'),
        ('grid', r'grid(\d+)'),
        ('lc', r'lc(\d+)(?:@(\d+)x(\d+))?(?:#(\d+)#(\d+))?(?:-(\d+)-(\d+))?'),
        ('avg', r'avg(?:@(\d+)x(\d+))?(?:#(\d+)#(\d+))?(?:-(\d+)-(\d+))?'),
        ('bng', 'bng'),
        ('batch_norm', 'bn%s?' % NAME_PATTERN),
        ('let', r'let(?::([_\.\w]+)=([\$_\.\w]+))(?::([_\.\w]+)=([\$_\.\w]+))?(?::([_\.\w]+)=([\$_\.\w]+))?'),
        #  ('set', r'set(?::([_\.\w]+)=([_\.\w]+))(?::([_\.\w]+)=([_\.\w]+))?(?::([_\.\w]+)=([_\.\w]+))?'),
        #  ('map', r'map(?::([_\.\w]+)=([_\.\w]+))(?::([_\.\w]+)=([_\.\w]+))?(?::([_\.\w]+)=([_\.\w]+))?'),
        ('dropout', r'\?(?:([\.\d]+))?'),
        ('act', r'!'),
        ('block_grad', r'bg'),
        ('scale_grad', r'sg([\.\d]+)'),
        ('clip_grad', r'cg([\.\d]+)'),
        ('repeat', r'x(\d+)'),
        ('scale_learning_rate', 'lr([\.\d]+)'),
        ('scale_weight_decay', 'wd([\.\d]+)'),
        ('linear_lc_loss', 'lll([\.\d]+)'),
        ('exp_lc_loss', 'ell([\.\d]+)'),
        ('param_lc_loss', 'pll([\.\d]+)'),
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
    def parse(cls, q, **kargs):
        if cls is Operator:
            for impl in OperatorMeta.impls:
                if impl is not Operator:
                    inst = impl.parse(q, **kargs)
                    if inst is not None:
                        return inst

    def get_last_name(self):
        return ctx.get('names', [None])[-1]

    def set_last_name(self, name):
        names = ctx.get('names', None)
        if names is None:
            names = ctx.names = [None]
        names.append(name)


class Reshape(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'reshape':
            inst = cls()
            inst.shape = tuple(int(word) for word in match.group(1).split(','))
            inst.shape_text = match.group(1)
            return inst
        q.backward()

    def build(self, data):
        return ctx.reshape(data=data, shape=self.shape)

    def encode(self):
        return 'S(%s)' % self.shape_text


class Group(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'group_left':
            inst = cls()
            inst.operators = parse_operators(q, **kargs)
            token, match = q.forward()
            if token != 'group_right':
                raise Exception('Missing ")" at {}'.format(q.current))
            return inst
        q.backward()

    def build(self, data):
        net = data
        for operator in self.operators:
            net = operator.build(net)
        return net

    def encode(self):
        return '(' + ''.join(op.encode() for op in self.operators) + ')'


class Repeat(Operator):

    @classmethod
    def parse(cls, q, previous, **kargs):
        token, match = q.forward()
        if token == 'repeat':
            inst = cls()
            inst.times = int(match.group(1))
            nt.assert_is_instance(previous, list)
            inst.operator = previous[-1]
            return previous[:-1] + [inst]
        q.backward()

    def build(self, data):
        net = data
        for i in range(self.times):
            net = self.operator.build(net)
        return net

    def encode(self):
        return '{}x{}'.format(self.operator.encode(), self.times)


class Convolution(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'conv':
            inst = cls()
            inst.num_filter = int(match.group(1))
            return inst
        q.backward()

    def format_name(self, index):
        return 'conv%d' % index

    def build(self, data):
        name = ctx.gen_next_name(self)
        net = ctx.get_conv(
            name=name,
            data=data,
            num_filter=self.num_filter,
            kernel=(3, 3),
            stride=(1, 1),
            pad=(1, 1),
            no_bias=True
        )
        net = ctx.get_act(
            name=name,
            data=net
        )
        self.set_last_name(name)
        return net

    def encode(self):
        return 'conv%d' % self.num_filter


class FullyConnected(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'fc':
            inst = cls()
            inst.num_hidden = int(match.group(1))
            return inst
        q.backward()

    def format_name(self, index):
        return 'fc%d' % index

    def build(self, data):
        name = ctx.gen_next_name(self)
        net = ctx.get_fc(
            name=name,
            data=data,
            num_hidden=self.num_hidden,
            no_bias=True,
        )
        net = ctx.get_act(
            name=name,
            data=net,
        )
        self.set_last_name(name)
        return net

    def encode(self):
        return 'fc%d' % self.num_hidden


class FSD(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'fsd':
            inst = cls()
            inst.num_hidden = int(match.group(1))
            inst.dropout = ctx.dropout if match.group(2) else 0
            inst.num_block = int(match.group(3))
            inst.p = float(match.group(4) or ctx.sd_p)
            inst.p_text = match.group(4)
            return inst
        q.backward()

    def format_name(self, index):
        return 'fsd%d' % index

    def build(self, data):
        name = ctx.gen_next_name(self)
        net = ctx.get_fsd(
            name=name,
            data=data,
            num_block=self.num_block,
            num_hidden=self.num_hidden,
            p=self.p,
            dropout=self.dropout
        )
        self.set_last_name(name)
        return net

    def encode(self):
        text = 'fsd%d' % self.num_hidden
        if self.dropout:
            text += '?'
        text += 'x%d' % self.num_block
        if self.p_text:
            text += '@' + self.p_text
        return text


class LSD(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'lsd':
            inst = cls()
            inst.num_filter = int(match.group(1))
            inst.kernel = (int(match.group(2)), int(match.group(3))) if match.group(2) and match.group(3) else (1, 1)
            if inst.kernel != (1, 1):
                inst.pad = (1, 1)
            else:
                inst.pad = (0, 0)
            if match.group(4) and match.group(5):
                inst.pad = (int(match.group(4)), int(match.group(5)))
            inst.scale = float(match.group(6) or 1)
            inst.scale_text = match.group(6)
            inst.dropout = ctx.dropout if match.group(7) else 0
            inst.num_block = int(match.group(8))
            inst.p = float(match.group(9) or ctx.sd_p)
            inst.p_text = match.group(9)
            inst.decay = match.group(10) is not None
            return inst
        q.backward()

    def format_name(self, index):
        return 'lsd%d' % index

    def build(self, data):
        name = ctx.gen_next_name(self)
        net = ctx.get_lsd(
            name=name,
            data=data,
            num_block=self.num_block,
            num_filter=self.num_filter,
            kernel=self.kernel,
            pad=self.pad,
            p=self.p,
            decay=self.decay,
            dropout=self.dropout,
            scale=self.scale,
            no_bias=ctx.lc_no_bias
        )
        self.set_last_name(name)
        return net

    def encode(self):
        text = 'lsd%d' % self.num_filter
        if self.kernel != (1, 1):
            text += '@%dx%d' % self.kernel
        if self.scale_text:
            text += 's' + self.scale_text
        if self.dropout:
            text += '?'
        text += 'x%d' % self.num_block
        if self.p_text:
            text += 'p' + self.p_text
        if self.decay:
            text += '+'
        return text


class SD(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'sd':
            inst = cls()
            inst.num_filter = int(match.group(1))
            inst.num_block = int(match.group(2))
            inst.p = float(match.group(3) or ctx.sd_p)
            inst.p_text = match.group(3)
            return inst
        q.backward()

    def format_name(self, index):
        return 'sd%d' % index

    def build(self, data):
        name = ctx.gen_next_name(self)
        net = ctx.get_sd(
            name=name,
            data=data,
            num_block=self.num_block,
            num_filter=self.num_filter,
            p=self.p
        )
        self.set_last_name(name)
        return net

    def encode(self):
        text = 'sd%dx%d' % (self.num_filter, self.num_block)
        if self.p_text:
            text += '@' + self.p_text
        return text


class OutputFullyConnected(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'out_fc':
            inst = cls()
            inst.num_hidden = int(match.group(1))
            return inst
        q.backward()

    def format_name(self, index):
        return 'out_fc%d' % index

    def build(self, data):
        name = ctx.gen_next_name(self)
        net = ctx.get_fc(
            name=name,
            data=data,
            num_hidden=self.num_hidden,
            no_bias=False,
        )
        self.set_last_name(name)
        return net

    def encode(self):
        return 'o%d' % self.num_hidden


class LastFullyConnected(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'last_fc':
            inst = cls()
            inst.num_hidden = int(match.group(1))
            return inst
        q.backward()

    def format_name(self, index):
        nt.assert_equal(index, 1)
        return 'last_fc'

    def build(self, data):
        name = ctx.gen_next_name(self)
        net = ctx.get_fc(
            name=name,
            data=data,
            num_hidden=self.num_hidden,
            no_bias=False,
        )
        self.set_last_name(name)
        return net

    def encode(self):
        return '$%d' % self.num_hidden


class InnerProduct(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'inner_product':
            inst = cls()
            inst.num_hidden = int(match.group(1))
            return inst
        q.backward()

    def format_name(self, index):
        return 'ip%d' % index

    def build(self, data):
        name = ctx.gen_next_name(self)
        net = ctx.get_fc(
            name=name,
            data=data,
            num_hidden=self.num_hidden,
            no_bias=True,
        )
        self.set_last_name(name)
        return net

    def encode(self):
        return 'ip%d' % self.num_hidden


class GridFullyConnected(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'grid':
            inst = cls()
            inst.num_hidden = int(match.group(1))
            return inst
        q.backward()

    def format_name(self, index):
        return 'grid%d' % index

    def build(self, data):
        name = ctx.gen_next_name(self)
        net = ctx.get_grid_fc(
            name=name,
            data=data,
            num_hidden=self.num_hidden,
            no_bias=True,
        )
        self.set_last_name(name)
        return net

    def encode(self):
        return 'grid%d' % self.num_hidden


class LocallyConnected(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'lc':
            inst = cls()
            g = 1
            inst.num_filter = int(match.group(g))
            g += 1
            if match.group(g) and match.group(g + 1):
                inst.kernel = (int(match.group(g)), int(match.group(g + 1)))
            else:
                inst.kernel = (1, 1)
            if inst.kernel != (1, 1):
                inst.pad = (1, 1)
            else:
                inst.pad = (0, 0)
            g += 2
            if match.group(g) and match.group(g + 1):
                inst.stride = (int(match.group(g)), int(match.group(g + 1)))
            else:
                inst.stride = (1, 1)
            g += 2
            if match.group(g) and match.group(g + 1):
                inst.pad = (int(match.group(g)), int(match.group(g + 1)))
            g += 2
            return inst
        q.backward()

    def format_name(self, index):
        return 'lc%d' % index

    def build(self, data):
        name = ctx.gen_next_name(self)
        net = ctx.get_lc(
            name=name,
            data=data,
            num_filter=self.num_filter,
            kernel=self.kernel,
            stride=self.stride,
            pad=self.pad,
            no_bias=ctx.lc_no_bias,
        )
        net = ctx.get_act(
            name=name,
            data=net,
        )
        self.set_last_name(name)
        return net

    def encode(self):
        raise NotImplementedError
        text = 'lc%d' % self.num_filter
        if self.kernel != (1, 1):
            text += '@%dx%d' % self.kernel
        return text


class Avg(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'avg':
            inst = cls()
            g = 1
            if match.group(g) and match.group(g + 1):
                inst.kernel = (int(match.group(g)), int(match.group(g + 1)))
            else:
                inst.kernel = (0, 0)
            g += 2
            if match.group(g) and match.group(g + 1):
                inst.stride = (int(match.group(g)), int(match.group(g + 1)))
            else:
                inst.stride = (1, 1)
            g += 2
            if match.group(g) and match.group(g + 1):
                inst.pad = (int(match.group(g)), int(match.group(g + 1)))
            else:
                inst.pad = (0, 0)
            g += 2
            return inst
        q.backward()

    def format_name(self, index):
        return 'avg%d' % index

    def build(self, data):
        net = ctx.get_avg(
            data=data,
            kernel=self.kernel,
            stride=self.stride,
            pad=self.pad
        )
        return net

    def encode(self):
        raise NotImplementedError


class BatchNorm(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'batch_norm':
            inst = cls()
            inst.name = match.group(1)
            return inst
        q.backward()

    def format_name(self, index):
        return self.name or 'bn%d' % index

    def build(self, data):
        return ctx.get_bn(
            name=ctx.gen_next_name(self),
            data=data,
        )

    def encode(self):
        return 'bn' if self.name is None else 'bn:' + self.name


class Let(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'let':
            inst = cls()
            gs = match.groups()
            inst.set_params = {}
            inst.map_params = {}
            for key, value in zip(gs[::2], gs[1::2]):
                if key is not None:
                    if value.startswith('$'):
                        inst.map_params[key] = value[1:]
                    else:
                        inst.set_params[key] = inst.__parse(value)
            inst.operator = Operator.parse(q, **kargs)
            if not inst.operator:
                raise Exception('Parse Let at {} failed'.format(q.begin))
            return inst
        q.backward()

    def __parse(self, value):
        try:
            return int(value)
        except:
            try:
                return float(value)
            except:
                return value

    def build(self, data):
        with ctx.push(self.get_context_cls(), **self.set_params):
            return self.operator.build(data)

    def encode(self):
        raise NotImplementedError

    def get_context_cls(host):
        class Map(BaseContext):
            pass

        for key, value in host.map_params.items():
            setattr(Map, key, property(lambda self: getattr(ctx, value)))
        return Map


#  class Set(Operator):

    #  @classmethod
    #  def parse(cls, q, **kargs):
        #  token, match = q.forward()
        #  if token == 'set':
            #  inst = cls()
            #  gs = match.groups()
            #  inst.params = {}
            #  for key, value in zip(gs[::2], gs[1::2]):
                #  if key is not None:
                    #  inst.params[key] = inst.__parse(value)
            #  inst.operator = Operator.parse(q, **kargs)
            #  if not inst.operator:
                #  raise Exception('Parse Set at {} failed'.format(q.begin))
            #  return inst
        #  q.backward()

    #  def __parse(self, value):
        #  try:
            #  return int(value)
        #  except:
            #  try:
                #  return float(value)
            #  except:
                #  return value

    #  def build(self, data):
        #  with ctx.push(**self.params):
            #  return self.operator.build(data)

    #  def encode(self):
        #  raise NotImplementedError


#  class Map(Operator):

    #  @classmethod
    #  def parse(cls, q, **kargs):
        #  token, match = q.forward()
        #  if token == 'map':
            #  inst = cls()
            #  gs = match.groups()
            #  inst.params = {}
            #  for key, value in zip(gs[::2], gs[1::2]):
                #  if key is not None:
                    #  inst.params[key] = value
            #  inst.operator = Operator.parse(q, **kargs)
            #  if not inst.operator:
                #  raise Exception('Parse Map at {} failed'.format(q.begin))
            #  return inst
        #  q.backward()

    #  def build(self, data):
        #  with ctx.push(self.get_context_cls()):
            #  return self.operator.build(data)

    #  def encode(self):
        #  raise NotImplementedError

    #  def get_context_cls(host):
        #  class Foo(BaseContext):
            #  pass

        #  for key, value in host.params.items():
            #  setattr(Foo, key, property(lambda self: getattr(ctx, value)))

        #  #  Note: following is **wrong**
        #  #  setattr(Foo, host.name, property(lambda self: getattr(self, host.value)))
        #  return Foo


class Dropout(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'dropout':
            inst = cls()
            inst.p = float(match.group(1)) if match.group(1) else ctx.dropout
            return inst
        q.backward()

    def build(self, data):
        if self.p == 0:
            return data

        name = ctx.gen_next_dropout_name()
        net = ctx.get_dropout(name=name, data=data, p=self.p)
        return net

    def encode(self):
        text = '?'
        if self.p != ctx.dropout:
            text += str(self.p)
        return text


class Activation(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'act':
            return cls()
        q.backward()

    def build(self, data):
        return ctx.get_act(
            name=self.get_last_name(),
            data=data
        )

    def encode(self):
        return '!'


class BlockGrad(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'block_grad':
            inst = cls()
            return inst
        q.backward()

    def build(self, data):
        return ctx.get_block_grad(data=data)

    def encode(self):
        return 'bg'


class ScaleGrad(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'scale_grad':
            inst = cls()
            inst.scale_text = match.group(1)
            inst.scale = float(inst.scale_text)
            return inst
        q.backward()

    def build(self, data):
        return ctx.get_scale_grad(data=data, scale=self.scale)

    def encode(self):
        return 'sg{}'.format(self.scale_text)


class ClipGrad(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'clip_grad':
            inst = cls()
            inst.clip_text = match.group(1)
            inst.clip = float(inst.clip_text)
            return inst
        q.backward()

    def build(self, data):
        return ctx.get_clip_grad(data=data, clip=self.clip)

    def encode(self):
        return 'cg{}'.format(self.clip_text)


class ScaleLearningRate(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'scale_learning_rate':
            inst = cls()
            inst.scale_text = match.group(1)
            inst.scale = float(inst.scale_text)
            inst.operator = Operator.parse(q, **kargs)
            if not inst.operator:
                raise Exception('Parse ScaleLearningRate at {} failed'.format(q.begin))
            return inst
        q.backward()

    def build(self, data):
        with ctx.push(lr_mult=self.scale * ctx.get('lr_mult', 1)):
            return self.operator.build(data)

    def encode(self):
        return 'lr{}{}'.format(self.scale_text, self.operator.encode())


class ScaleWeightDecay(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'scale_weight_decay':
            inst = cls()
            inst.scale_text = match.group(1)
            inst.scale = float(inst.scale_text)
            inst.operator = Operator.parse(q, **kargs)
            if not inst.operator:
                raise Exception('Parse ScaleWeightDecay at {} failed'.format(q.begin))
            return inst
        q.backward()

    def build(self, data):
        with ctx.push(wd_mult=self.scale * ctx.get('wd_mult', 1)):
            return self.operator.build(data)

    def encode(self):
        return 'wd{}{}'.format(self.scale_text, self.operator.encode())


class LinearLocallyConnectedLoss(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'linear_lc_loss':
            inst = cls()
            inst.step_text = match.group(1)
            inst.step = float(inst.step_text)
            inst.operator = Operator.parse(q, **kargs)
            if not inst.operator:
                raise Exception('Parse LinearLocallyConnectedLoss at {} failed'.format(q.begin))
            return inst
        q.backward()

    def build(self, data):
        with ctx.push(self.get_context_cls()):
            return self.operator.build(data)

    def encode(self):
        return 'lll{}{}'.format(self.step_text, self.operator.encode())

    def get_context_cls(host):
        class Context(BaseContext):

            def __init__(self, parent, **kargs):
                super(Context, self).__init__(parent, **kargs)
                self.__num_lc = 0

            @property
            def lc_loss_weight(self):
                return self.parent.lc_loss_weight * self.__next_decay()

            def __next_decay(self):
                decay = 1 - host.step * self.__num_lc
                nt.assert_greater_equal(decay, 0)
                self.__num_lc += 1
                return decay

        return Context


class ExpLocallyConnectedLoss(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'exp_lc_loss':
            inst = cls()
            inst.step_text = match.group(1)
            inst.step = float(inst.step_text)
            inst.operator = Operator.parse(q, **kargs)
            if not inst.operator:
                raise Exception('Parse ExpLocallyConnectedLoss at {} failed'.format(q.begin))
            return inst
        q.backward()

    def build(self, data):
        with ctx.push(self.get_context_cls()):
            return self.operator.build(data)

    def encode(self):
        return 'ell{}{}'.format(self.step_text, self.operator.encode())

    def get_context_cls(host):
        class Context(BaseContext):

            def __init__(self, parent, **kargs):
                super(Context, self).__init__(parent, **kargs)
                self.__num_lc = 0

            @property
            def lc_loss_weight(self):
                return self.parent.lc_loss_weight * self.__next_decay()

            def __next_decay(self):
                decay = host.step ** self.__num_lc
                nt.assert_greater_equal(decay, 0)
                self.__num_lc += 1
                return decay

        return Context


class ParametricLocallyConnectedLoss(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'param_lc_loss':
            inst = cls()
            inst.operator = Operator.parse(q, **kargs)
            inst.pll_text = match.group(1)
            inst.pll = float(inst.pll_text)
            if not inst.operator:
                raise Exception('Parse ParametricLocallyConnectedLoss at {} failed'.format(q.begin))
            return inst
        q.backward()

    def build(self, data):
        with ctx.push(self.get_context_cls()):
            return self.operator.build(data)

    def encode(self):
        return self.pll_text + self.operator.encode()

    def get_context_cls(host):
        class Context(BaseContext):

            def __init__(self, parent, **kargs):
                super(Context, self).__init__(parent, **kargs)
                self.__num_lc = 0
                import mxnet as mx
                name = 'pll_%f' % host.pll
                pll = mx.symbol.Variable(
                    name,
                    attr={'__wd_mult__': '0',
                          '__shape__': str((1,)),
                          '__dtype__': 'float32'}
                )
                self.__gamma = mx.symbol.Activation(
                    pll,
                    name=name + '_gamma',
                    act_type='sigmoid'
                )

            @property
            def lc_loss_weight(self):
                return self.parent.lc_loss_weight * self.__next_decay()

            def __next_decay(self):
                decay = self.__gamma ** self.__num_lc
                self.__num_lc += 1
                return decay

        return Context


class FixBN(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'fixbn':
            inst = cls()
            inst.operator = Operator.parse(q, **kargs)
            if not inst.operator:
                raise Exception('Parse FixBN at {} failed'.format(q.begin))
            return inst
        q.backward()

    def build(self, data):
        with ctx.push(fix_batch_norm=True):
            return self.operator.build(data)

    def encode(self):
        return 'fixbn' + self.operator.encode()


class Fix(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'fix':
            inst = cls()
            inst.operator = Operator.parse(q, **kargs)
            if not inst.operator:
                raise Exception('Parse Fix at {} failed'.format(q.begin))
            return inst
        q.backward()

    def build(self, data):
        with ctx.push(lr_mult=0, fix_batch_norm=True):
            return self.operator.build(data)

    def encode(self):
        return 'fix' + self.operator.encode()


class Bng(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'bng':
            inst = cls()
            inst.operator = Operator.parse(q, **kargs)
            if not inst.operator:
                raise Exception('Parse Bng at {} failed'.format(q.begin))
            return inst
        q.backward()

    def build(self, data):
        with ctx.push(batch_norm_use_global_stats=True):
            return self.operator.build(data)

    def encode(self):
        return 'bng' + self.operator.encode()


class Share(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'share':
            inst = cls()
            inst.operator = Operator.parse(q, **kargs)
            if not inst.operator:
                raise Exception('Parse Share at {} failed'.format(q.begin))
            return inst
        q.backward()

    def build(self, data):
        data, unpack = ctx.pack(data)
        return unpack(self.operator.build(data))

    def encode(self):
        return 'share' + self.operator.encode()


class Tee(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'tee':
            inst = cls()
            inst.operator = Operator.parse(q, **kargs)
            if not inst.operator:
                raise Exception('Parse Tee at {} failed'.format(q.begin))
            return inst
        q.backward()

    def build(self, data):
        #  return [self.operator.build(branch) for branch in data]
        net = data
        branch = self.operator.build(net)
        net = ctx.concat(ctx.flatten(net), ctx.flatten(branch), axis=1)
        return net

    def encode(self):
        return 'tee' + self.operator.encode()


class Identity(Operator):

    @classmethod
    def parse(cls, q, **kargs):
        token, match = q.forward()
        if token == 'identity':
            inst = cls()
            inst.name = match.group(1)
            return inst
        q.backward()

    def build(self, data):
        return ctx.get_identity(data=data, name=self.name)

    def encode(self):
        return 'id' if self.name is None else 'id:' + self.name


class Sequence(Operator):
    '''This class must be defined at the end of this file'''

    @classmethod
    def parse(cls, q, **kargs):
        top = q.begin == 0
        token, match = q.forward()
        has_left = token == 'sequence_left'
        if top or has_left:
            if not has_left:
                q.backward()
            inst = cls()
            inst.operators = parse_operators(q, **kargs)
            has_right = False
            if has_left:
                token, match = q.forward()
                has_right = token == 'sequence_right'
            if not top and not has_right:
                raise Exception('Missing "]" at {}'.format(q.current))
            return inst
        q.backward()

    def build(self, data):
        net = data
        with ctx.push(self.get_context_cls()):
            for operator in self.operators:
                net = operator.build(net)
        return net

    def encode(self):
        return '[' + ''.join(op.encode() for op in self.operators) + ']'

    def get_context_cls(host):
        class Context(BaseContext):

            def __init__(self, parent, **kargs):
                super(Context, self).__init__(parent, **kargs)
                self.prefix = getattr(self.parent, 'prefix', None)
                self.names = [None]
                self.__counter_memo = {}

            def gen_next_name(self, operator):
                index = self.counter(type(operator))
                return self.add_prefix(operator.format_name(index=index))

            def gen_next_dropout_name(self):
                last_name = host.get_last_name()
                return last_name + '_drop' if last_name else None

            def add_prefix(self, name):
                return self.prefix + '_' + name if self.prefix else name

            def counter(self, operator):
                if operator in self.__counter_memo:
                    self.__counter_memo[operator] += 1
                else:
                    self.__counter_memo[operator] = 1
                return self.__counter_memo[operator]

        return Context
