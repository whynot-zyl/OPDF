#!/bin/bash
# nohup bash glue_script/script_qqp.sh &
# QQP
# nohup bash -c 'CUDA_VISIBLE_DEVICES=2 python run_glue_MPO_losslayer.py --model_name_or_path /mnt/zhanyuliang/data/nlp_data/theseus/BertFineTrain/download/qqp/bert-base-uncased-QQP/ --task_name qqp --do_train --do_eval --do_lower_case --data_dir /mnt/zhanyuliang/data/nlp_data/GLUE/QQP --max_seq_length 128 --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 --learning_rate 5e-6 --save_steps 50 --num_train_epochs 3 --output_dir /mnt/zhanyuliang/data/checkpoint/nlp/theseus/save_successor/ --evaluate_during_training --replacing_rate 0.3 --scheduler_type linear --scheduler_linear_k 0.0001 --logging_steps=500 --overwrite_output_dir' >log/qqp_mpo.log >&1 &
# wait
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 taskset -c 20-29 python run_glue_MPO_losslayer.py --model_name_or_path /mnt/zhanyuliang/data/nlp_data/theseus/BertFineTrain/download/qqp/bert-base-uncased-QQP/ --task_name qqp --do_train --do_eval --do_lower_case --data_dir /mnt/zhanyuliang/data/nlp_data/GLUE/QQP --max_seq_length 128 --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 --learning_rate 1e-6 --save_steps 50 --num_train_epochs 5 --output_dir /mnt/zhanyuliang/data/checkpoint/nlp/theseus/save_successor/ --evaluate_during_training --replacing_rate 0.3 --scheduler_type linear --scheduler_linear_k 0.0003 --logging_steps=500 --overwrite_output_dir' >log/qqp_mpo_2.log >&1 &
wait
nohup bash -c 'CUDA_VISIBLE_DEVICES=1 taskset -c 30-39 python run_glue_MPO_losslayer.py --model_name_or_path /mnt/zhanyuliang/data/nlp_data/theseus/BertFineTrain/download/qqp/bert-base-uncased-QQP/ --task_name qqp --do_train --do_eval --do_lower_case --data_dir /mnt/zhanyuliang/data/nlp_data/GLUE/QQP --max_seq_length 128 --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 --learning_rate 1e-6 --save_steps 50 --num_train_epochs 5 --output_dir /mnt/zhanyuliang/data/checkpoint/nlp/theseus/save_successor/ --evaluate_during_training --replacing_rate 0.3 --scheduler_type linear --scheduler_linear_k 0.0006 --logging_steps=500 --overwrite_output_dir' >log/qqp_mpo_3.log >&1 &
wait