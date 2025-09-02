#!/bin/bash

# MovieCORE Test Script for HERMES
# Usage: bash run_scripts/moviecore/test.sh [checkpoint_path]

export CUDA_VISIBLE_DEVICES=5

checkpoint_path=/home/phd/12/josmyfaure/vlms/HERMES/checkpoint_best.pth

torchrun --nproc_per_node=1 \
    --master_port=34659 \
    train.py \
    --cfg-path lavis/projects/hermes/qa_moviecore.yaml \
    --options \
    model.arch blip2_vicuna_instruct \
    model.model_type vicuna7b \
    model.load_finetuned False \
    model.load_pretrained True \
    model.num_query_token 32 \
    model.vit_precision fp16 \
    model.freeze_vit True \
    model.memory_bank_length 20 \
    model.num_frames 100 \
    model.window_size 10 \
    model.num_frames_global 20 \
    model.trail_percentage 0.02 \
    model.max_txt_len 512 \
    model.max_output_txt_len 512 \
    model.is_zero_shot False \
    run.init_lr 1e-4 \
    run.max_epoch 5 \
    run.num_beams 1 \
    run.batch_size_train 3 \
    run.batch_size_eval 1 \
    run.accum_grad_iters 1 \
    run.num_workers 12 \
    run.seed 42 \
    run.evaluate True \
    run.valid_splits "['test']" \
    run.report_metric True \
    run.prefix moviecore \
    run.resume_ckpt_path ${checkpoint_path}
