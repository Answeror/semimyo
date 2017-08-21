#!/usr/bin/env bash

ver=dbb-full
scripts/app train_semimyo \
  --log log --snapshot model \
  --root .cache/semimyo-v$ver \
  --batch-size 1000 --dataset "capgmyo.dbb.semg_prev_semi" --dataset-args '{"step":100}' \
  --symbol 'semimyo_order' \
  --module 'semimyo_order' \
  --symbol-args '{"shared_net":"bn:zscore,conv64x2,lc64x2?,(fc512?)x2","gesture_net":"fc128,$8","order_net":"fc128,$2","gesture_loss_weight":1,"order_loss_weight":1,"loss_normalization":"valid"}' \
  --crossval-type universal-intra-subject --fold 0
for i in $(seq 0 9 | shuf); do
  scripts/app train_semimyo \
    --log log --snapshot model \
    --root .cache/semimyo-fold-$i-v$ver \
    --params .cache/semimyo-v$ver/model-0028.params \
    --batch-size 1000 --dataset "capgmyo.dbb.semg_prev_semi" --dataset-args '{"step":100}' \
    --symbol 'semimyo_order' \
    --module 'semimyo_order' \
    --symbol-args '{"shared_net":"bn:zscore,conv64x2,lc64x2?,(fc512?)x2","gesture_net":"fc128,$8","order_net":"fc128,$2","gesture_loss_weight":1,"order_loss_weight":1,"loss_normalization":"valid"}' \
    --crossval-type intra-subject --fold $i
done

ver=dbb
scripts/app train_semimyo \
  --log log --snapshot model \
  --root .cache/semimyo-v$ver \
  --batch-size 1000 --dataset "capgmyo.dbb.semg_prev_semi_downsample" --dataset-args "{\"step\":100,\"downsample\":16}" \
  --symbol 'semimyo_order' \
  --module 'semimyo_order' \
  --symbol-args '{"shared_net":"bn:zscore,conv64x2,lc64x2?,(fc512?)x2","gesture_net":"fc128,$8","order_net":"fc128,$2","gesture_loss_weight":1,"order_loss_weight":1,"loss_normalization":"valid"}' \
  --crossval-type universal-intra-subject --fold 0
for i in $(seq 0 9 | shuf); do
  scripts/app train_semimyo \
    --log log --snapshot model \
    --root .cache/semimyo-fold-$i-v$ver \
    --params .cache/semimyo-v$ver/model-0028.params \
    --batch-size 1000 --dataset "capgmyo.dbb.semg_prev_semi_downsample" --dataset-args "{\"step\":100,\"downsample\":16}" \
    --symbol 'semimyo_order' \
    --module 'semimyo_order' \
    --symbol-args '{"shared_net":"bn:zscore,conv64x2,lc64x2?,(fc512?)x2","gesture_net":"fc128,$8","order_net":"fc128,$2","gesture_loss_weight":1,"order_loss_weight":1,"loss_normalization":"valid"}' \
    --crossval-type intra-subject --fold $i
done

ver=dbb-baseline
scripts/app train_semimyo \
  --log log --snapshot model \
  --root .cache/semimyo-v$ver \
  --batch-size 1000 --dataset "capgmyo.dbb.semg_downsample" --dataset-args "{\"downsample\":16}" \
  --symbol 'semimyo' \
  --module 'semimyo' \
  --symbol-args '{"shortnet":"bn:zscore,conv64x2,lc64x2?,(fc512?)x2,fc128,$8","loss_normalization":"valid"}' \
  --crossval-type universal-intra-subject --fold 0
for i in $(seq 0 9 | shuf); do
  scripts/app train_semimyo \
    --log log --snapshot model \
    --root .cache/semimyo-fold-$i-v$ver \
    --params .cache/semimyo-v$ver/model-0028.params \
    --batch-size 1000 --dataset "capgmyo.dbb.semg_downsample" --dataset-args "{\"downsample\":16}" \
    --symbol 'semimyo' \
    --module 'semimyo' \
    --symbol-args '{"shortnet":"bn:zscore,conv64x2,lc64x2?,(fc512?)x2,fc128,$8","loss_normalization":"valid"}' \
    --crossval-type intra-subject --fold $i
done
