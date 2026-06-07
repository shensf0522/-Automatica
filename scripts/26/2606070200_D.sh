#!/bin/bash
# ==============================================================================
# 实验版本 D: 双Loss限制重建 (Double Loss Target)
# - 增强策略: denoise (时域多视角去噪: 平滑 + 中值 + 可预测性滤波)
# - 损失设计: double (同时优化 loss_clean 和 loss_faithful)
# - 超参数: res_double_beta = 0.3
# ==============================================================================
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
if [ ! -d "./logs/pretrain" ]; then
    mkdir ./logs/pretrain
fi
set -e

export CUDA_VISIBLE_DEVICES=0

MODEL=FAT_res_trend_gate_new
EXP_NAME=2606070200_D
TRANSFER_EXP_NAME=${EXP_NAME}
SEQ_LEN=336

ROOT_PATH=./dataset/ETT-small_old/
DATA_PATH=ETTh1.csv
DATASET=ETTh1
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
    --freq h"

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
    --res_aug_version denoise \
    --res_recon_target double \
    --res_double_beta 0.3 \
    ${COMMON_ARGS} >logs/pretrain/$EXP_NAME'_'$MODEL'_'$DATASET'_'$SEQ_LEN.log

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
        --res_aug_version denoise \
        --res_recon_target double \
        --res_double_beta 0.3 \
        ${COMMON_ARGS} >logs/LongForecasting/$EXP_NAME'_'$MODEL'_'$DATASET'_'$SEQ_LEN'_'$PRED_LEN'_'$FORECAST_MODE.log
  done
done
