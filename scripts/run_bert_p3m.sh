#!/bin/sh

seed=66

python train.py --data_dir ./dataset/docred \
    --transformer_type bert \
    --model_name_or_path ../../pretrain/bert-base-cased \
    --train_file train_annotated.json \
    --dev_file dev.json \
    --test_file test_revised.json \
    --train_batch_size 4 \
    --test_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-5 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.06 \
    --num_train_epochs 10.0 \
    --seed ${seed} \
    --num_class 97 \
    --la 10 \
    --e 3.0 \
    --aug pos_aug \
    --dropout_rate 0.2 \
    --use_mixup 1 \
    --mixup_alpha 1.0 \
    --mixup_rate 0.05 \
    --mixup_type r_emb
