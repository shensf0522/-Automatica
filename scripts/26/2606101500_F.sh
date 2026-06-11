#!/bin/bash
# ==============================================================================
# 实验版本 F: 多类型噪声注入增强 (Multi-Noise Injection Augmentation)
# - 增强策略: noise_inject (在 mask 位置注入 3 种不同噪声:
#     View1=高斯白噪声, View2=拉普拉斯重尾噪声, View3=有色漂移噪声)
# - 重建目标: raw (从含噪视图重建原始干净信号，学会去噪能力)
# - CL分支: 含噪视图作为原始信号的正样本，仍使用KL散度对齐
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
EXP_NAME=2606101500_F
TRANSFER_EXP_NAME=${EXP_NAME}
SEQ_LEN=96

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
    --res_aug_version noise_inject \
    --res_recon_target raw \
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
        --res_aug_version noise_inject \
        --res_recon_target raw \
        ${COMMON_ARGS} >logs/LongForecasting/$EXP_NAME'_'$MODEL'_'$DATASET'_'$SEQ_LEN'_'$PRED_LEN'_'$FORECAST_MODE.log
  done
done
