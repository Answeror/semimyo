#!/usr/bin/env bash

ver=csl-full
for i in $(seq 0 9 | shuf); do
  scripts/app train_semimyo \
    --log log --snapshot model \
    --root .cache/semimyo-fold-$i-v$ver.0 \
    --batch-size 1000 --dataset "csl.semg_prev_semi" --dataset-args '{"step":205,"expand":512,"balance_gesture":1,"balance_gesture_ignore_ratio":0.5}' \
    --preprocess 'downsample-5' \
    --symbol 'semimyo_order' \
    --module 'semimyo_order' \
    --symbol-args '{"shared_net":"bn:zscore,conv64x2,lc64x2?,(fc512?)x2","gesture_net":"fc128,$27","order_net":"fc128,$2","gesture_loss_weight":2,"order_loss_weight":1}' \
    --crossval-type universal-intra-session --fold $i
done
for i in $(seq 0 249 | shuf); do
  scripts/app train_semimyo \
    --log log --snapshot model \
    --root .cache/semimyo-fold-$i-v$ver \
    --params .cache/semimyo-fold-$(($i % 10))-v$ver.0/model-0028.params \
    --batch-size 1000 --dataset "csl.semg_prev_semi" --dataset-args '{"step":205,"expand":512,"balance_gesture":1,"balance_gesture_ignore_ratio":0.5}' \
    --symbol 'semimyo_order' \
    --module 'semimyo_order' \
    --symbol-args '{"shared_net":"fix(bn:zscore,conv64x2),lc64x2?,(fc512?)x2","gesture_net":"fc128,$27","order_net":"fc128,$2","gesture_loss_weight":2,"order_loss_weight":1}' \
    --crossval-type intra-session --fold $i
done

ver=csl
for i in $(seq 0 249 | shuf); do
  scripts/app train_semimyo \
    --log log --snapshot model \
    --root .cache/semimyo-fold-$i-v$ver \
    --batch-size 1000 --dataset "csl.semg_prev_semi_downsample" --dataset-args "{\"downsample\":16,\"expand\":0,\"step\":205,\"balance_gesture\":1,\"balance_gesture_ignore_downsample\":\"1/8\"}" \
    --symbol 'semimyo_order' \
    --module 'semimyo_order' \
    --symbol-args '{"shared_net":"bn:zscore,conv64x2,lc64x2?,(fc512?)x2","gesture_net":"fc128,$27","order_net":"fc128,$2","gesture_loss_weight":1,"order_loss_weight":1,"loss_normalization":"valid"}' \
    --crossval-type intra-session --fold $i
done

ver=csl-baseline
for i in $(seq 0 249 | shuf); do
  scripts/app train_semimyo \
    --log log --snapshot model \
    --root .cache/semimyo-fold-$i-v$ver \
    --batch-size 1000 --dataset "csl.semg_downsample" --dataset-args '{"balance_gesture":1,"downsample":16}' \
    --symbol 'semimyo' \
    --module 'semimyo' \
    --symbol-args '{"shortnet":"bn:zscore,conv64x2,lc64x2?,(fc512?)x2,fc128,$27"}' \
    --crossval-type intra-session --fold $i
done
