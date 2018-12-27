#!/usr/bin/env bash

python train_wsdm_pl.py reproduce_bert_pseudo_labels \
                --data_dir pl_data \
                --bert_model bert-base-chinese \
                --task_name wsdm_pseudo \
                --output_dir ../zake7749/data/bert/reproduce_bert_pseudo_labels \
                --max_seq_length 128 \
                --do_train \
                --train_batch_size 32 \
                --eval_batch_size 8 \
                --learning_rate 5e-5 \
                --num_train_epochs 3.0 \
                --warmup_proportion 0.1 \
                --seed 42 \
                --gradient_accumulation_steps 1 \
                --do_test \
                #--subset 320 \
                #--dev_subset 80 \
                #--no_cuda
                #--do_eval \
                #--fp16 \
                #--loss_scale 128 \
                #--optimize_on_cpu \
                #--local_rank -1 \
