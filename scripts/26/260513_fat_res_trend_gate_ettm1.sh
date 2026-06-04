#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=2

MODEL=FAT_res_trend_gate_new
EXP_NAME=260513_fat_res_trend_gate_ettm1_sl336
TRANSFER_EXP_NAME=${EXP_NAME}
SEQ_LEN=336

ROOT_PATH=./dataset/ETT-small_old/
DATA_PATH=ETTm1.csv
DATASET=ETTm1
ENC_IN=7
DEC_IN=7
C_OUT=7

COMMON_ARGS="\
    --root_path ${ROOT_PATH} \
    --data_path ${DATA_PATH} \
    --model ${MODEL} \
    --data ${DATASET} \
    --pretrain_data ${DATASET} \
    --features M \
    --d_model 8 \
    --n_heads 8 \
    --e_layers 2 \
    --d_ff 32 \
    --seq_len ${SEQ_LEN} \
    --batch_size 16 \
    --enc_in ${ENC_IN} \
    --dec_in ${DEC_IN} \
    --c_out ${C_OUT} \
    --decomp_kernel 13 \
    --n_knlg 32 \
    --struct_dropout 0.4 \
    --memory_size 64 \
    --top_k 5 \
    --use_time_index 1 \
    --time_feature_dim 6 \
    --embed timeF \
    --freq t"

python -u run.py \
    --exp_name ${EXP_NAME} \
    --task_name pretrain \
    --pretrain_mode residual \
    --learning_rate 0.0008 \
    --pretrain_epochs 50 \
    --mask_rate 0.5 \
    --lm 3 \
    --positive_nums 3 \
    --negative_nums 1 \
    ${COMMON_ARGS}

for FORECAST_MODE in freq unfreq; do
  for PRED_LEN in 96 192 336 720; do
    python -u run.py \
        --task_type reg \
        --pretrain_mode residual \
        --task_name finetune \
        --transfer_expname ${TRANSFER_EXP_NAME} \
        --freeze 1 \
        --learning_rate 0.0001 \
        --train_epochs 20 \
        --dropout 0.1 \
        --head_dropout 0.1 \
        --exp_name ${EXP_NAME}_${FORECAST_MODE} \
        --pred_len ${PRED_LEN} \
        --patience 5 \
        --forcastMode ${FORECAST_MODE} \
        ${COMMON_ARGS}
  done
done
