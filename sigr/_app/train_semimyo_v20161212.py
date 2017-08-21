from __future__ import division
import click
import mxnet as mx
from logbook import Logger
from pprint import pformat
import os
from .utils import packargs, Bunch
from .data import Preprocess, Dataset
from . import Context, constant, get_module


logger = Logger('semimyo')


@click.group()
def cli():
    pass


@cli.command()
@click.option('--module', required=True)
@click.option('--cudnn-tune', type=click.Choice(['off', 'limited_workspace', 'fastest']), default='fastest')
@click.option('--symbol', required=True)
@click.option('--shared-net')
@click.option('--gesture-net')
@click.option('--latent-net')
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
@click.option('--dropout', type=float, default=constant.DROPOUT, help='Dropout ratio')
@click.option('--gesture-loss-weight', type=float)
@click.option('--latent-loss-weight', type=float)
@click.option('--num-mini-batch', type=int, default=constant.NUM_MINI_BATCH, help='Split data into mini-batches')
@click.option('--num-eval-epoch', type=int, default=1)
@click.option('--snapshot-period', type=int, default=constant.SNAPSHOT_PERIOD)
@click.option('--fix-params', multiple=True)
@click.option('--decay-all/--no-decay-all', default=constant.DECAY_ALL)
@click.option('--preprocess', callback=lambda ctx, param, value: Preprocess.parse(value))
@click.option('--dataset', type=click.Choice(['s21', 'csl',
                                              'dba', 'dbb', 'dbc',
                                              'ninapro-db1-matlab-lowpass',
                                              'ninapro-db1/caputo',
                                              'ninapro-db1',
                                              'ninapro-db1-raw/semg-glove',
                                              'ninapro-db1-raw/semg-spot',
                                              'ninapro-db1-raw/semg-libhand',
                                              'ninapro-db1-raw/semg-latent-128',
                                              'ninapro-db1-raw/semg-latent-8',
                                              'ninapro-db1-raw/semg-latent-3',
                                              'ninapro-db1/g53',
                                              'ninapro-db1/g5',
                                              'ninapro-db1/g8',
                                              'ninapro-db1/g12']), required=True)
@click.option('--balance-gesture', type=float, default=0)
@click.option('--amplitude-weighting', is_flag=True)
@click.option('--fold', type=int, required=True, help='Fold number of the crossval experiment')
@click.option('--crossval-type', type=click.Choice(['intra-session',
                                                    'universal-intra-session',
                                                    'inter-session',
                                                    'universal-inter-session',
                                                    'intra-subject',
                                                    'universal-intra-subject',
                                                    'inter-subject',
                                                    'one-fold-intra-subject',
                                                    'universal-one-fold-intra-subject']), required=True)
@packargs
def crossval(args):
    if args.root:
        if args.log:
            args.log = os.path.join(args.root, args.log)
        if args.snapshot:
            args.snapshot = os.path.join(args.root, args.snapshot)

    with Context(args.log, parallel=False):
        logger.info('Args:\n{}', pformat(args))
        for i in range(args.num_epoch):
            path = args.snapshot + '-%04d.params' % (i + 1)
            if os.path.exists(path):
                logger.info('Found snapshot {}, exit', path)
                return

        dataset = Dataset.from_name(args.dataset)
        get_crossval_data = getattr(dataset, 'get_%s_data' % args.crossval_type.replace('-', '_'))
        train, val = get_crossval_data(
            batch_size=args.batch_size,
            fold=args.fold,
            preprocess=args.preprocess,
            num_mini_batch=args.num_mini_batch,
            balance_gesture=args.balance_gesture,
            amplitude_weighting=args.amplitude_weighting
        )
        logger.info('Train samples: {}', train.num_sample)
        logger.info('Val samples: {}', val.num_sample)
        mod = get_module(
            args.module,
            network=args.symbol,
            adabn=args.adabn,
            num_adabn_epoch=args.num_adabn_epoch,
            for_training=True,
            num_eval_epoch=args.num_eval_epoch,
            snapshot_period=args.snapshot_period,
            symbol_kargs=dict(
                num_latent=dataset.num_latent,
                shared_net=args.shared_net,
                gesture_net=args.gesture_net,
                latent_net=args.latent_net,
                num_gesture=dataset.num_gesture,
                num_semg_row=dataset.num_semg_row,
                num_semg_col=dataset.num_semg_col,
                dropout=args.dropout,
                num_mini_batch=args.num_mini_batch,
                gesture_loss_weight=args.gesture_loss_weight,
                latent_loss_weight=args.latent_loss_weight,
                cudnn_tune=args.cudnn_tune
            ),
            context=[mx.gpu(i) for i in args.gpu]
        )
        mod.fit(
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


if __name__ == '__main__':
    cli(obj=Bunch())
