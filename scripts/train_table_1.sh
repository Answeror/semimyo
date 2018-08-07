#!/usr/bin/env bash

ver=t2
scripts/app train_semimyo \
  --log log --snapshot model \
  --root .cache/semimyo-v$ver \
  --batch-size 1000 --dataset "ninapro.db1.raw_semg_prev" --dataset-args '{"step":10}' \
  --dataiter-args '{"norest":1,"balance_gesture":1}' \
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
    --dataiter-args '{"norest":1,"balance_gesture":1}' \
    --preprocess '{ninapro-lowpass,identity,identity}' \
    --symbol 'semimyo_order' \
    --module 'semimyo_order' \
    --symbol-args '{"shared_net":"fix(bn:zscore,conv64x2),lc64x2","gesture_net":"?(fc512?)x2,fc128,$52","order_net":"fc128,$2","gesture_loss_weight":1,"order_loss_weight":1}' \
    --crossval-type one-fold-intra-subject --fold $i
done

ver=t3
scripts/app train_semimyo_v20161213 \
  --log log --snapshot model \
  --root .cache/semimyo-v$ver \
  --batch-size 1000 --dataset "ninapro.db1.raw_semg_pose" --dataset-args '{"num_pose":512}' \
  --balance-gesture 1 \
  --preprocess '{ninapro-lowpass,identity,identity}' \
  --symbol 'semimyo_pose_shortnet' \
  --module 'semimyo_v20161213' \
  --shared-net 'bn:zscore,conv64x2,lc64x2?,(fc512?)x2' \
  --gesture-net 'fc128,$52' \
  --pose-net 'fc128,$512' \
  --gesture-loss-weight 1 \
  --pose-loss-weight 1 \
  --crossval-type universal-one-fold-intra-subject --fold 0
for i in $(seq 0 26 | shuf); do
  scripts/app train_semimyo_v20161213 \
    --log log --snapshot model \
    --root .cache/semimyo-fold-$i-v$ver \
    --params .cache/semimyo-v$ver/model-0028.params \
    --batch-size 1000 --dataset "ninapro.db1.raw_semg_pose" --dataset-args '{"num_pose":512}' \
    --balance-gesture 1 \
    --preprocess '{ninapro-lowpass,identity,identity}' \
    --symbol 'semimyo_pose_shortnet' \
    --module 'semimyo_v20161213' \
    --shared-net 'fix(bn:zscore,conv64x2),lc64x2?,(fc512?)x2' \
    --gesture-net 'fc128,$52' \
    --pose-net 'fc128,$512' \
    --gesture-loss-weight 1 \
    --pose-loss-weight 1 \
    --crossval-type one-fold-intra-subject --fold $i
done

ver=t23
scripts/app train_semimyo \
  --log log --snapshot model \
  --root .cache/semimyo-v$ver \
  --batch-size 1000 --dataset "ninapro.db1.raw_semg_pose_prev" --dataset-args '{"num_pose":512,"step":10}' \
  --dataiter-args '{"norest":1,"balance_gesture":1}' \
  --preprocess '{ninapro-lowpass}' \
  --symbol 'semimyo_pose_order' \
  --module 'semimyo_pose_order' \
  --symbol-args '{"shared_net":"bn:zscore,conv64x2,lc64x2;?(fc512?)x2","gesture_net":"fc128,$52","pose_net":"fc128,$512","order_net":"fc128,$2","gesture_loss_weight":1,"pose_loss_weight":1,"order_loss_weight":1}' \
  --crossval-type universal-one-fold-intra-subject --fold 0
for i in $(seq 0 26 | shuf); do
  scripts/app train_semimyo \
    --log log --snapshot model \
    --root .cache/semimyo-fold-$i-v$ver \
    --params .cache/semimyo-v$ver/model-0028.params \
    --batch-size 1000 --dataset "ninapro.db1.raw_semg_pose_prev" --dataset-args '{"num_pose":512,"step":10}' \
    --dataiter-args '{"norest":1,"balance_gesture":1}' \
    --preprocess '{ninapro-lowpass}' \
    --symbol 'semimyo_pose_order' \
    --module 'semimyo_pose_order' \
    --symbol-args '{"shared_net":"fix(bn:zscore,conv64x2),lc64x2;?(fc512?)x2","gesture_net":"fc128,$52","pose_net":"fc128,$512","order_net":"fc128,$2","gesture_loss_weight":1,"pose_loss_weight":1,"order_loss_weight":1}' \
    --crossval-type one-fold-intra-subject --fold $i
done
