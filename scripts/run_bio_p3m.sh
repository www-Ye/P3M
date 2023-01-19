#!/bin/sh

seed=66

python train_bio.py --data_dir ./dataset/chemdisgene \
    --transformer_type bert \
    --model_name_or_path ../../pretrain/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --train_file train.json \
    --dev_file valid.json \
    --test_file test.anno_all.json \
    --train_batch_size 8 \
    --test_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.06 \
    --num_train_epochs 10.0 \
    --seed ${seed} \
    --num_class 15 \
    --la 10 \
    --e 3.0 \
    --aug pos_aug \
    --dropout_rate 0.2 \
    --use_mixup 1 \
    --mixup_alpha 1.0 \
    --mixup_rate 0.05 \
    --mixup_type r_emb
