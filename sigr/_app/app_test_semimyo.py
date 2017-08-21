from __future__ import division
import numpy as np
import mxnet as mx
from logbook import Logger
from .. import Context, utils
from ..get_module import get_module
from ..get_dataset import get_dataset
from ..get_evaluation import get_evaluation
from ..evaluation import Exp
from . import app, Cmd, d, options


name = utils.appname(__name__)
cmd = Cmd(name)
logger = Logger(name)


base_args = dict(
    evaluation='semimyo_v20161210',
    module='semimyo',
    symbol='semimyo',
    symbol_args={'shared_net': 'bn:zscore,conv64x2,lc64x2?,(fc512?)x2'})
db1_args = dict(dataset='ninapro.db1.sdata_semg',
                tag=utils.Append(['db1']),
                symbol_args=utils.Append({'gesture_net': 'fc128,$52'}))
dba_args = dict(dataset='capgmyo.dba.semg',
                tag=utils.Append(['dba']),
                symbol_args=utils.Append({'gesture_net': 'fc128,$8'}))
dbb_args = dict(dataset='capgmyo.dbb.semg',
                tag=utils.Append(['dbb']),
                symbol_args=utils.Append({'gesture_net': 'fc128,$8'}))
dbc_args = dict(dataset='capgmyo.dbc.semg',
                tag=utils.Append(['dbc']),
                symbol_args=utils.Append({'gesture_net': 'fc128,$12'}))
csl_semi_args = dict(
    dataset='csl.semg_prev_semi',
    dataset_args={'step': 205, 'expand': 512},
    tag=utils.Append(['csl']),
    symbol_args=utils.Append({
        'shared_net': 'fix(bn:zscore,conv64x2),lc64x2?,(fc512?)x2',
        'gesture_net': 'fc128,$27'}))
csl_args = dict(
    dataset='csl.semg',
    tag=utils.Append(['csl']),
    symbol_args=utils.Append({
        'shared_net': 'bn:zscore,conv64x2,lc64x2?,(fc512?)x2',
        'gesture_net': 'fc128,$27'}))
intra_subject_args = dict(crossval='intra_subject',
                          tag=utils.Append(['intra_subject']))
intra_session_args = dict(crossval='intra_session',
                          tag=utils.Append(['intra_session']))
adabn_args = dict(symbol_args=utils.Append({'adabn': 1, 'num_adabn_epoch': 10}))
baseline = lambda G: dict(
    symbol_args=utils.Append(
        {'shortnet': 'bn:zscore,conv64x2,lc64x2?,(fc512?)x2,fc128,$%d' % G}),
    tag=utils.Append(['baseline'])
)
db1_order_args = dict(symbol_args=utils.Append({
    'shared_net': 'bn:zscore,conv64x2,lc64x2',
    'gesture_net': '?(fc512?)x2,fc128,$52'}))
down = utils.Append('down')
paper_format = dict(format='%.3f')
paper_base_args = [base_args, paper_format, intra_subject_args]


@app.command()
@options.gpu_option
@options.deprecated_option
@options.option('--log-level', default='info')
@options.option('--max-num-fold', type=int)
@options.option('--format')
@options.option('--cmd')
@utils.packargs
@utils.rename(name)
def main(args, cmd):
    if cmd:
        utils.invoke(globals()[cmd])


@d(cmd)
def all(args):
    utils.invoke(table_1)
    utils.invoke(table_4)


@d(cmd)
def table_4(args):
    '''
    [2017-02-19 06:53:37.136457] INFO: test_semimyo: db1,intra_subject,per-frame: 0.779
    [2017-02-19 06:54:14.589587] INFO: test_semimyo: db1,intra_subject,vote:20: 0.794
    [2017-02-19 06:54:58.562410] INFO: test_semimyo: db1,down,intra_subject,per-frame: 0.760
    [2017-02-19 06:55:27.314958] INFO: test_semimyo: db1,down,intra_subject,vote:20: 0.776
    [2017-02-19 06:56:14.292195] INFO: test_semimyo: baseline,db1,down,intra_subject,per-frame: 0.738
    [2017-02-19 06:56:50.895035] INFO: test_semimyo: baseline,db1,down,intra_subject,vote:20: 0.757
    [2017-02-19 06:57:01.661054] INFO: test_semimyo: dba,intra_subject,per-frame: 0.895
    [2017-02-19 06:57:18.785741] INFO: test_semimyo: dba,intra_subject,vote:150: 0.996
    [2017-02-19 06:57:36.179645] INFO: test_semimyo: dba,down,intra_subject,per-frame: 0.797
    [2017-02-19 06:57:50.024678] INFO: test_semimyo: dba,down,intra_subject,vote:150: 0.988
    [2017-02-19 06:58:02.863253] INFO: test_semimyo: baseline,dba,down,intra_subject,per-frame: 0.764
    [2017-02-19 06:58:14.743855] INFO: test_semimyo: baseline,dba,down,intra_subject,vote:150: 0.981
    [2017-02-19 06:58:22.169628] INFO: test_semimyo: dbb,intra_subject,per-frame: 0.859
    [2017-02-19 06:58:26.865821] INFO: test_semimyo: dbb,intra_subject,vote:150: 0.987
    [2017-02-19 06:58:32.471061] INFO: test_semimyo: dbb,down,intra_subject,per-frame: 0.762
    [2017-02-19 06:58:37.836269] INFO: test_semimyo: dbb,down,intra_subject,vote:150: 0.972
    [2017-02-19 06:58:45.822310] INFO: test_semimyo: baseline,dbb,down,intra_subject,per-frame: 0.732
    [2017-02-19 06:58:48.981175] INFO: test_semimyo: baseline,dbb,down,intra_subject,vote:150: 0.970
    [2017-02-19 06:58:51.441297] INFO: test_semimyo: dbc,intra_subject,per-frame: 0.850
    [2017-02-19 06:59:06.620618] INFO: test_semimyo: dbc,intra_subject,vote:150: 0.992
    [2017-02-19 06:59:14.576277] INFO: test_semimyo: dbc,down,intra_subject,per-frame: 0.731
    [2017-02-19 06:59:21.406167] INFO: test_semimyo: dbc,down,intra_subject,vote:150: 0.985
    [2017-02-19 06:59:29.309061] INFO: test_semimyo: baseline,dbc,down,intra_subject,per-frame: 0.680
    [2017-02-19 06:59:37.551972] INFO: test_semimyo: baseline,dbc,down,intra_subject,vote:150: 0.979
    [2017-02-19 07:08:27.631564] INFO: test_semimyo: csl,intra_session,per-frame: 0.560
    [2017-02-19 07:15:02.606978] INFO: test_semimyo: csl,intra_session,vote:all: 0.969
    [2017-02-19 07:20:07.704494] INFO: test_semimyo: csl,down,intra_session,per-frame: 0.491
    [2017-02-19 07:24:47.235968] INFO: test_semimyo: csl,down,intra_session,vote:all: 0.943
    [2017-02-19 07:29:41.877527] INFO: test_semimyo: baseline,csl,down,intra_session,per-frame: 0.482
    [2017-02-19 07:32:16.275785] INFO: test_semimyo: baseline,csl,down,intra_session,vote:all: 0.929
    '''
    utils.invoke(table_4_db1)
    utils.invoke(table_4_dba)
    utils.invoke(table_4_dbb)
    utils.invoke(table_4_dbc)
    utils.invoke(table_4_csl)


@d(cmd)
def table_4_db1(args):
    with utils.use(base_args, paper_format, intra_subject_args,
                   db1_args, fold=list(range(27))):
        with utils.use(db1_order_args):
            utils.invoke(ninapro_db1_report, version='db1-full')
            utils.invoke(ninapro_db1_report, version='db1', tag=down)
        with utils.use(baseline(52)):
            utils.invoke(ninapro_db1_report, version='db1-baseline', tag=down)


@d(cmd)
def table_4_dba(args):
    with utils.use(base_args, paper_format, intra_subject_args,
                   dba_args, fold=list(range(18))):
        utils.invoke(capgmyo_report, version='dba-full')
        utils.invoke(capgmyo_report, version='dba', tag=down)
        with utils.use(baseline(8)):
            utils.invoke(capgmyo_report, version='dba-baseline', tag=down)


@d(cmd)
def table_4_dbb(args):
    with utils.use(base_args, paper_format, intra_subject_args,
                   dbb_args, fold=list(range(10))):
        utils.invoke(capgmyo_report, version='dbb-full')
        utils.invoke(capgmyo_report, version='dbb', tag=down)
        with utils.use(baseline(8)):
            utils.invoke(capgmyo_report, version='dbb-baseline', tag=down)


@d(cmd)
def table_4_dbc(args):
    with utils.use(base_args, paper_format, intra_subject_args,
                   dbc_args, fold=list(range(10))):
        utils.invoke(capgmyo_report, version='dbc-full')
        utils.invoke(capgmyo_report, version='dbc', tag=down)
        with utils.use(baseline(12)):
            utils.invoke(capgmyo_report, version='dbc-baseline', tag=down)


@d(cmd)
def table_4_csl(args):
    with utils.use(base_args, paper_format):
        with utils.use(csl_semi_args, adabn_args, intra_session_args,
                       fold=list(range(250))):
            utils.invoke(csl_report, version='csl-full')
        with utils.use(csl_args, intra_session_args, fold=list(range(250))):
            utils.invoke(csl_report, version='csl', tag=down)
            with utils.use(baseline(27)):
                utils.invoke(csl_report, version='csl-baseline', tag=down)


@d(cmd)
def table_1(args):
    with utils.use(base_args, paper_format, intra_subject_args,
                   db1_args, fold=list(range(27))):
        with utils.use(db1_order_args):
            utils.invoke(ninapro_db1_report, version='t2',
                         tag=utils.Append('T2'))
        utils.invoke(ninapro_db1_report, version='t3', tag=utils.Append('T3'))
        utils.invoke(ninapro_db1_report, version='t23', tag=utils.Append('T23'))


@d(cmd.option('--version', required=True))
def capgmyo_report(args):
    utils.invoke(report, window=150)


@d(cmd.option('--version', required=True))
def csl_report(args):
    utils.invoke(report, window=-1, balance=1)


@d(cmd.option('--version', required=True))
def ninapro_db1_report(args):
    utils.invoke(report, window=[20], balance=1)


@d(cmd.option('--version', required=True))
def caputo_report(args):
    utils.invoke(report, window=[40])


@d(cmd
   .option('--version', required=True)
   .option('--window', multiple=True, required=True))
def report(args, window):
    with utils.appendargs(tag=['per-frame']):
        utils.invoke(core)

    windows = window if isinstance(window, (list, tuple)) else [window]
    for win in windows:
        with utils.appendargs(tag=['vote:{}'.format('all' if win < 0 else win)]):
            utils.invoke(core, window=win)


@app.command()
@options.option('--evaluation', required=True)
@options.option('--crossval', required=True)
@options.option('--dataset', required=True)
@options.option('--module', required=True)
@options.option('--symbol', required=True)
@options.option('--version', required=True)
@options.option('--fold', type=int, multiple=True, required=True)
@options.option('--params',
                default='.cache/semimyo-fold-{fold}-v%s/model-%04d.params')
@options.json_option('--dataset-args', default={})
@options.json_option('--dataiter-args', default={})
@options.json_option('--symbol-args', default={})
@options.batch_size_option
@options.epoch_option
@options.gpu_option
@options.deprecated_option
@options.option('--balance', is_flag=True)
@options.option('--window', type=int)
@options.option('--tag', multiple=True)
@options.option('--log-level', default='info')
@options.option('--max-num-fold', type=int)
@options.option('--format')
@utils.packargs
@utils.subname(name)
def core(args):
    if 'deprecated' in args.tag and not args.deprecated:
        return

    with Context(parallel=True,
                 level=args.log_level.upper(),
                 mxnet_context=mx.Context(mx.gpu(args.gpu[0]))):
        utils.logargs(logger, args, 'DEBUG')
        eva = get_evaluation(
            args.evaluation,
            crossval=args.crossval,
            batch_size=args.batch_size,
            context=[mx.gpu(args.gpu[0])]
        )
        dataset = get_dataset(args.dataset, **args.dataset_args)
        symbol_args = dict(batch_size=args.batch_size,
                           num_gesture=dataset.num_gesture,
                           num_semg_row=dataset.num_semg_row,
                           num_semg_col=dataset.num_semg_col,
                           dropout=0,
                           shortnet_args={})
        symbol_args.update(args.symbol_args)

        if not args.window:
            if args.balance:
                get_accuracies = utils.F(eva.vote_accuracy_curves,
                                         balance=True,
                                         windows=[1])
            else:
                get_accuracies = eva.accuracies
        else:
            get_accuracies = utils.F(eva.vote_accuracy_curves,
                                     balance=args.balance,
                                     windows=[args.window])

        fold = np.array(args.fold)
        if args.max_num_fold is not None:
            fold = fold[:args.max_num_fold]

        acc = get_accuracies(
            [Exp(dataset=dataset,
                 dataset_args=args.dataiter_args,
                 Mod=Mod(args.module,
                         for_training=False,
                         network=args.symbol,
                         symbol_kargs=symbol_args,
                         params=str(args.params) % (args.version, args.epoch)))],
            folds=fold)
        acc = acc.mean()

        tag = sorted(args.tag, key=tag_sort_key)
        logger.info('{}: %s' % parse_format(args.format),
                    ','.join(tag) if tag else args.version, acc)


def parse_format(fmt):
    if not fmt:
        return '{}'
    if fmt.startswith('{') and fmt.endswith('}'):
        return fmt
    if fmt.startswith('%'):
        fmt = fmt[1:]
    return '{:%s}' % fmt


def tag_sort_key(tag):
    level = {'deprecated': 10}.get(tag, 0)
    return (level, tag)


def Mod(name, **kargs):
    return utils.F(
        utils.format_args(
            utils.ignore_args(
                utils.F(get_module, name, runtime=True),
                'fold'
            ),
            'params'
        ),
        **kargs
    )
