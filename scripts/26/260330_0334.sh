#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
MODEL=FAT_VLM_Cross_att
BASE_EXP_NAME=260330_0334

# ==================== 第一阶段：先跑 336 seq_len + 16 batch_size ====================
echo "========== 显卡2：【第一阶段】开始跑 Weather seq_len=336 batch_size=16 =========="
SEQ_LEN=336
batch_size=4

DATASET=Weather
ROOT_PATH=./dataset/weather/
DATA_PATH=weather.csv
ENC_IN=21
DEC_IN=21
C_OUT=21
EXP_NAME=${BASE_EXP_NAME}_${DATASET}_seq${SEQ_LEN}_bs${batch_size}

# 预训练
python -u run.py \
    --exp_name $EXP_NAME \
    --task_name pretrain \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --model $MODEL \
    --data weather \
    --pretrain_data weather \
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
echo "========== 完成 Weather 预训练，开始微调 =========="
for PRED_LEN in 96 192 336 720; do
    python -u run.py \
        --task_type reg \
        --pretrain_mode residual \
        --task_name finetune \
        --root_path $ROOT_PATH \
        --data_path $DATA_PATH \
        --model $MODEL \
        --pretrain_data weather \
        --transfer_expname $EXP_NAME \
        --data weather \
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
echo "========== 完成 Weather seq_len=${SEQ_LEN} batch_size=${batch_size} 全流程 =========="
echo "========== 显卡2：【第一阶段】336/16配置完成 =========="

# ==================== 第二阶段：再跑 512 seq_len + 8 batch_size ====================
echo "========== 显卡2：【第二阶段】开始跑 Weather seq_len=512 batch_size=8 =========="
SEQ_LEN=512
batch_size=2

DATASET=Weather
ROOT_PATH=./dataset/weather/
DATA_PATH=weather.csv
ENC_IN=21
DEC_IN=21
C_OUT=21
EXP_NAME=${BASE_EXP_NAME}_${DATASET}_seq${SEQ_LEN}_bs${batch_size}

# 预训练
python -u run.py \
    --exp_name $EXP_NAME \
    --task_name pretrain \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --model $MODEL \
    --data weather \
    --pretrain_data weather \
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
echo "========== 完成 Weather 预训练，开始微调 =========="
for PRED_LEN in 96 192 336 720; do
    python -u run.py \
        --task_type reg \
        --pretrain_mode residual \
        --task_name finetune \
        --root_path $ROOT_PATH \
        --data_path $DATA_PATH \
        --model $MODEL \
        --pretrain_data weather \
        --transfer_expname $EXP_NAME \
        --data weather \
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
echo "========== 完成 Weather seq_len=${SEQ_LEN} batch_size=${batch_size} 全流程 =========="
echo "========== 显卡2：【第二阶段】512/8配置完成 =========="
echo "========== 显卡2所有任务全部结束 =========="
脚本 4：显卡 3 - 先跑 Electricity 的 336/16 → 再跑 Electricity 的 512/8（保存为 run_gpu3_electricity.sh）
bash
运行
#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
MODEL=FAT_VLM_Cross_att
BASE_EXP_NAME=260330_0250

# ==================== 第一阶段：先跑 336 seq_len + 16 batch_size ====================
echo "========== 显卡3：【第一阶段】开始跑 Electricity seq_len=336 batch_size=16 =========="
SEQ_LEN=336
batch_size=1

DATASET=Electricity
ROOT_PATH=./dataset/electricity/
DATA_PATH=electricity.csv
ENC_IN=321
DEC_IN=321
C_OUT=321
EXP_NAME=${BASE_EXP_NAME}_${DATASET}_seq${SEQ_LEN}_bs${batch_size}

# 预训练
python -u run.py \
    --exp_name $EXP_NAME \
    --task_name pretrain \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --model $MODEL \
    --data electricity \
    --pretrain_data electricity \
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
echo "========== 完成 Electricity 预训练，开始微调 =========="
for PRED_LEN in 96 192 336 720; do
    python -u run.py \
        --task_type reg \
        --pretrain_mode residual \
        --task_name finetune \
        --root_path $ROOT_PATH \
        --data_path $DATA_PATH \
        --model $MODEL \
        --pretrain_data electricity \
        --transfer_expname $EXP_NAME \
        --data electricity \
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
echo "========== 完成 Electricity seq_len=${SEQ_LEN} batch_size=${batch_size} 全流程 =========="
echo "========== 显卡3：【第一阶段】336/16配置完成 =========="

# ==================== 第二阶段：再跑 512 seq_len + 8 batch_size ====================
echo "========== 显卡3：【第二阶段】开始跑 Electricity seq_len=512 batch_size=8 =========="
SEQ_LEN=512
batch_size=1

DATASET=Electricity
ROOT_PATH=./dataset/electricity/
DATA_PATH=electricity.csv
ENC_IN=321
DEC_IN=321
C_OUT=321
EXP_NAME=${BASE_EXP_NAME}_${DATASET}_seq${SEQ_LEN}_bs${batch_size}

# 预训练
python -u run.py \
    --exp_name $EXP_NAME \
    --task_name pretrain \
    --root_path $ROOT_PATH \
    --data_path $DATA_PATH \
    --model $MODEL \
    --data electricity \
    --pretrain_data electricity \
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
echo "========== 完成 Electricity 预训练，开始微调 =========="
for PRED_LEN in 96 192 336 720; do
    python -u run.py \
        --task_type reg \
        --pretrain_mode residual \
        --task_name finetune \
        --root_path $ROOT_PATH \
        --data_path $DATA_PATH \
        --model $MODEL \
        --pretrain_data electricity \
        --transfer_expname $EXP_NAME \
        --data electricity \
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
echo "========== 完成 Electricity seq_len=${SEQ_LEN} batch_size=${batch_size} 全流程 =========="
echo "========== 显卡3：【第二阶段】512/8配置完成 =========="
echo "========== 显卡3所有任务全部结束 =========="