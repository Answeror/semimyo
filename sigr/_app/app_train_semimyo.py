from __future__ import division
import click
import mxnet as mx
from logbook import Logger
from pprint import pformat
import os
import json
from ..utils import packargs
from ..data import Preprocess
from .. import Context, constant
from ..get_module import get_module
from ..get_dataset import get_dataset
from . import app


logger = Logger('semimyo')


@app.command()
@click.option('--force', is_flag=True)
@click.option('--dataiter-args', default='{}', callback=lambda ctx, param, value: json.loads(value))
@click.option('--symbol-args', default='{}', callback=lambda ctx, param, value: json.loads(value))
@click.option('--shortnet-args', default='{}', callback=lambda ctx, param, value: json.loads(value))
@click.option('--monitor-pattern')
@click.option('--monitor-interval', type=int, default=1000)
@click.option('--module', required=True)
@click.option('--cudnn-tune', type=click.Choice(['off', 'limited_workspace', 'fastest']), default='fastest')
@click.option('--symbol', required=True)
@click.option('--num-epoch', type=int, default=constant.NUM_EPOCH, help='Maximum epoches')
@click.option('--lr-step', type=int, multiple=True, default=constant.LR_STEP, help='Epoch numbers to decay learning rate')
@click.option('--lr-factor', type=float, multiple=True)
@click.option('--batch-size', type=int, default=constant.BATCH_SIZE, help='Batch size')
@click.option('--lr', type=float, default=constant.LR, help='Base learning rate')
@click.option('--wd', type=float, default=constant.WD, help='Weight decay')
@click.option('--gpu', type=int, multiple=True, default=[0])
@click.option('--log', type=click.Path(), help='Path of the logging file')
@click.option('--snapshot', type=click.Path(), help='Snapshot prefix')
@click.option('--root', type=click.Path(), help='Root path of the experiment, auto create if not exists')
@click.option('--params', type=click.Path(exists=True), help='Inital weights')
@click.option('--ignore-params', multiple=True, help='Ignore params in --params with regex')
@click.option('--adabn', is_flag=True, help='AdaBN for model adaptation, must be used with --num-mini-batch')
@click.option('--num-adabn-epoch', type=int, default=constant.NUM_ADABN_EPOCH)
@click.option('--num-mini-batch', type=int, default=constant.NUM_MINI_BATCH, help='Split data into mini-batches')
@click.option('--num-eval-epoch', type=int, default=[1], multiple=True)
@click.option('--snapshot-period', type=int, default=constant.SNAPSHOT_PERIOD)
@click.option('--fix-params', multiple=True)
@click.option('--decay-all/--no-decay-all', default=constant.DECAY_ALL)
@click.option('--preprocess', callback=lambda ctx, param, value: Preprocess.parse(value))
@click.option('--dataset', required=True)
@click.option('--dataset-args', default='{}', callback=lambda ctx, param, value: json.loads(value))
#  @click.option('--balance-gesture', type=float, default=0)
@click.option('--amplitude-weighting', is_flag=True)
@click.option('--fold', type=int, required=True, help='Fold number of the crossval experiment')
@click.option('--crossval-type', required=True)
#  @click.option('--crossval-type', type=click.Choice(['intra-session',
                                                    #  'universal-intra-session',
                                                    #  'inter-session',
                                                    #  'universal-inter-session',
                                                    #  'intra-subject',
                                                    #  'universal-intra-subject',
                                                    #  'inter-subject',
                                                    #  'one-fold-intra-subject',
                                                    #  'universal-one-fold-intra-subject']), required=True)
@packargs
def train_semimyo(args):
    if args.root:
        if args.log:
            args.log = os.path.join(args.root, args.log)
        if args.snapshot:
            args.snapshot = os.path.join(args.root, args.snapshot)

    if '20170102.21' in args.root or '20170215.1.32' in args.root or '20170215.2.32' in args.root:
        click.echo('Deprecated: ' + args.root)
        return

    #  if ('universal' in args.crossval_type or 'inter' in args.crossval_type) and os.path.exists(args.log):
    if not args.force and os.path.exists(args.log):
        click.echo('Found log {}, exit'.format(args.log))
        return

    with Context(args.log, parallel=True, mxnet_context=mx.Context(mx.gpu(args.gpu[0]))):
        logger.info('Args:\n{}', pformat(args))
        for i in range(args.num_epoch):
            path = args.snapshot + '-%04d.params' % (i + 1)
            if not args.force and os.path.exists(path):
                logger.info('Found snapshot {}, exit', path)
                return

        dataset = get_dataset(args.dataset, **args.dataset_args)
        get_crossval_data = getattr(dataset, 'get_%s_data' % args.crossval_type.replace('-', '_'))
        train, val = get_crossval_data(
            batch_size=args.batch_size,
            fold=args.fold,
            preprocess=args.preprocess,
            num_mini_batch=args.num_mini_batch,
            #  balance_gesture=args.balance_gesture,
            amplitude_weighting=args.amplitude_weighting,
            **args.dataiter_args
        )
        logger.info('Train samples: {}', train.num_sample)
        logger.info('Val samples: {}', val.num_sample)
        symbol_kargs = dict(
            batch_size=args.batch_size,
            num_gesture=dataset.num_gesture,
            num_semg_row=dataset.num_semg_row,
            num_semg_col=dataset.num_semg_col,
            num_mini_batch=args.num_mini_batch,
            cudnn_tune=args.cudnn_tune,
            shortnet_args=args.shortnet_args
        )
        symbol_kargs.update(args.symbol_args)
        mod = get_module(
            args.module,
            network=args.symbol,
            adabn=args.adabn,
            num_adabn_epoch=args.num_adabn_epoch,
            for_training=True,
            num_eval_epoch=args.num_eval_epoch,
            snapshot_period=args.snapshot_period,
            symbol_kargs=symbol_kargs,
            context=[mx.gpu(i) for i in args.gpu]
        )
        mod.fit(
            monitor_pattern=args.monitor_pattern,
            monitor_interval=args.monitor_interval,
            train_data=train,
            eval_data=val,
            num_epoch=args.num_epoch,
            num_train=train.num_sample,
            batch_size=args.batch_size,
            lr_step=args.lr_step,
            lr_factor=args.lr_factor,
            lr=args.lr,
            wd=args.wd,
            snapshot=args.snapshot,
            params=args.params,
            ignore_params=args.ignore_params,
            fix_params=args.fix_params,
            decay_all=args.decay_all
        )
