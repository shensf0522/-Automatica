#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
MODEL=FAT_VLM_Cross_att
BASE_EXP_NAME=260330_0342

# ==================== 第一阶段：先跑 336 seq_len + 16 batch_size ====================
echo "========== 显卡4：【第一阶段】开始跑 Traffic seq_len=336 batch_size=16 =========="
SEQ_LEN=336
batch_size=1

DATASET=Traffic
ROOT_PATH=./dataset/traffic/
DATA_PATH=traffic.csv
ENC_IN=862
DEC_IN=862
C_OUT=862
EXP_NAME=${BASE_EXP_NAME}_${DATASET}_seq${SEQ_LEN}_bs${batch_size}

# 预训练
python -u run.py \
    --exp_name $EXP_NAME \
    --task_name pretrain \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --model $MODEL \
    --data Traffic \
    --pretrain_data Traffic \
    --features M \
    --pretrain_mode residual \
    --d_model 8 \
    --n_heads 8 \
    --e_layers 2 \
    --d_ff 32 \
    --learning_rate 0.0008 \
    --seq_len $SEQ_LEN \
    --batch_size $batch_size \
    --pretrain_epochs 50 \
    --mask_rate 0.5 \
    --lm 3 \
    --enc_in $ENC_IN \
    --dec_in $DEC_IN \
    --c_out $C_OUT \
    --positive_nums 3 \
    --negative_nums 1 \
    --decomp_kernel 13 \
    --n_knlg 32 \
    --struct_dropout 0.4

# 微调
echo "========== 完成 Traffic 预训练，开始微调 =========="
for PRED_LEN in 96 192 336 720; do
    python -u run.py \
        --task_type reg \
        --pretrain_mode residual \
        --task_name finetune \
        --root_path $ROOT_PATH \
        --data_path $DATA_PATH \
        --model $MODEL \
        --pretrain_data Traffic \
        --transfer_expname $EXP_NAME \
        --data Traffic \
        --features M \
        --freeze 1 \
        --d_model 8 \
        --n_heads 8 \
        --e_layers 2 \
        --d_ff 32 \
        --learning_rate 0.0001 \
        --batch_size $batch_size \
        --train_epochs 20 \
        --dropout 0.1 \
        --head_dropout 0.1 \
        --seq_len $SEQ_LEN \
        --enc_in $ENC_IN \
        --dec_in $DEC_IN \
        --c_out $C_OUT \
        --exp_name $EXP_NAME \
        --pred_len $PRED_LEN \
        --patience 5 \
        --decomp_kernel 13 \
        --n_knlg 32
done
echo "========== 完成 Traffic seq_len=${SEQ_LEN} batch_size=${batch_size} 全流程 =========="
echo "========== 显卡4：【第一阶段】336/16配置完成 =========="

# ==================== 第二阶段：再跑 512 seq_len + 8 batch_size ====================
echo "========== 显卡4：【第二阶段】开始跑 Traffic seq_len=512 batch_size=8 =========="
SEQ_LEN=512
batch_size=1

DATASET=Traffic
ROOT_PATH=./dataset/traffic/
DATA_PATH=traffic.csv
ENC_IN=862
DEC_IN=862
C_OUT=862
EXP_NAME=${BASE_EXP_NAME}_${DATASET}_seq${SEQ_LEN}_bs${batch_size}

# 预训练
python -u run.py \
    --exp_name $EXP_NAME \
    --task_name pretrain \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --model $MODEL \
    --data Traffic \
    --pretrain_data Traffic \
    --features M \
    --pretrain_mode residual \
    --d_model 8 \
    --n_heads 8 \
    --e_layers 2 \
    --d_ff 32 \
    --learning_rate 0.0008 \
    --seq_len $SEQ_LEN \
    --batch_size $batch_size \
    --pretrain_epochs 50 \
    --mask_rate 0.5 \
    --lm 3 \
    --enc_in $ENC_IN \
    --dec_in $DEC_IN \
    --c_out $C_OUT \
    --positive_nums 3 \
    --negative_nums 1 \
    --decomp_kernel 13 \
    --n_knlg 32 \
    --struct_dropout 0.4

# 微调
echo "========== 完成 Traffic 预训练，开始微调 =========="
for PRED_LEN in 96 192 336 720; do
    python -u run.py \
        --task_type reg \
        --pretrain_mode residual \
        --task_name finetune \
        --root_path $ROOT_PATH \
        --data_path $DATA_PATH \
        --model $MODEL \
        --pretrain_data custom \
        --transfer_expname $EXP_NAME \
        --data custom \
        --features M \
        --freeze 1 \
        --d_model 8 \
        --n_heads 8 \
        --e_layers 2 \
        --d_ff 32 \
        --learning_rate 0.0001 \
        --batch_size $batch_size \
        --train_epochs 20 \
        --dropout 0.1 \
        --head_dropout 0.1 \
        --seq_len $SEQ_LEN \
        --enc_in $ENC_IN \
        --dec_in $DEC_IN \
        --c_out $C_OUT \
        --exp_name $EXP_NAME \
        --pred_len $PRED_LEN \
        --patience 5 \
        --decomp_kernel 13 \
        --n_knlg 32
done
echo "========== 完成 Traffic seq_len=${SEQ_LEN} batch_size=${batch_size} 全流程 =========="
echo "========== 显卡4：【第二阶段】512/8配置完成 =========="
echo "========== 显卡4所有任务全部结束 =========="