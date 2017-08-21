#!/usr/bin/env bash

ver=db1-full
scripts/app train_semimyo \
  --log log --snapshot model \
  --root .cache/semimyo-v$ver \
  --batch-size 1000 --dataset "ninapro.db1.raw_semg_prev" --dataset-args '{"step":10}' \
  --balance-gesture 1 \
  --preprocess '{ninapro-lowpass,identity,identity}' \
  --symbol 'semimyo_order' \
  --module 'semimyo_order' \
  --symbol-args '{"shared_net":"bn:zscore,conv64x2,lc64x2","gesture_net":"?(fc512?)x2,fc128,$52","order_net":"fc128,$2","gesture_loss_weight":1,"order_loss_weight":1}' \
  --crossval-type universal-one-fold-intra-subject --fold 0
for i in $(seq 0 26 | shuf); do
  scripts/app train_semimyo \
    --log log --snapshot model \
    --root .cache/semimyo-fold-$i-v$ver \
    --params .cache/semimyo-v$ver/model-0028.params \
    --batch-size 1000 --dataset "ninapro.db1.raw_semg_prev" --dataset-args '{"step":10}' \
    --balance-gesture 1 \
    --preprocess '{ninapro-lowpass,identity,identity}' \
    --symbol 'semimyo_order' \
    --module 'semimyo_order' \
    --symbol-args '{"shared_net":"fix(bn:zscore,conv64x2),lc64x2","gesture_net":"?(fc512?)x2,fc128,$52","order_net":"fc128,$2","gesture_loss_weight":1,"order_loss_weight":1}' \
    --crossval-type one-fold-intra-subject --fold $i
done

ver=db1
scripts/app train_semimyo \
  --log log --snapshot model \
  --root .cache/semimyo-v$ver \
  --batch-size 1000 --dataset "ninapro.db1.sdata_semg_prev_semi_downsample" --dataset-args '{"step":10,"balance_gesture":"1/2","balance_gesture_ignore_ratio":"7/8","downsample":16}' \
  --symbol 'semimyo_order' \
  --module 'semimyo_order' \
  --symbol-args '{"shared_net":"bn:zscore,conv64x2,lc64x2","gesture_net":"?(fc512?)x2,fc128,$52","order_net":"fc128,$2","gesture_loss_weight":8,"order_loss_weight":1}' \
  --crossval-type universal-one-fold-intra-subject --fold 0
for i in $(seq 0 26 | shuf); do
  scripts/app train_semimyo \
    --log log --snapshot model \
    --root .cache/semimyo-fold-$i-v$ver \
    --params .cache/semimyo-v$ver/model-0028.params \
    --batch-size 1000 --dataset "ninapro.db1.sdata_semg_prev_semi_downsample" --dataset-args '{"step":10,"balance_gesture":"1/2","balance_gesture_ignore_ratio":"7/8","downsample":16}' \
    --symbol 'semimyo_order' \
    --module 'semimyo_order' \
    --symbol-args '{"shared_net":"fix(bn:zscore,conv64x2),lc64x2","gesture_net":"?(fc512?)x2,fc128,$52","order_net":"fc128,$2","gesture_loss_weight":8,"order_loss_weight":1}' \
    --crossval-type one-fold-intra-subject --fold $i
done

ver=db1-baseline
scripts/app train_semimyo \
  --log log --snapshot model \
  --root .cache/semimyo-v$ver \
  --batch-size 1000 --dataset "ninapro.db1.sdata_semg_downsample" --dataset-args '{"downsample":16,"balance_gesture":1}' \
  --symbol 'semimyo' \
  --module 'semimyo' \
  --symbol-args '{"shortnet":"bn:zscore,conv64x2,lc64x2?,(fc512?)x2,fc128,$52"}' \
  --crossval-type universal-one-fold-intra-subject --fold 0
for i in $(seq 0 26 | shuf); do
  scripts/app train_semimyo \
    --log log --snapshot model \
    --root .cache/semimyo-fold-$i-v$ver \
    --params .cache/semimyo-v$ver/model-0028.params \
    --batch-size 1000 --dataset "ninapro.db1.sdata_semg_downsample" --dataset-args '{"downsample":16,"balance_gesture":1}' \
    --symbol 'semimyo' \
    --module 'semimyo' \
    --symbol-args '{"shortnet":"bn:zscore,conv64x2,lc64x2?,(fc512?)x2,fc128,$52"}' \
    --crossval-type one-fold-intra-subject --fold $i
done
