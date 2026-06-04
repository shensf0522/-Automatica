#!/bin/bash
# ============================================================
# FAT_interPDN_v2 消融实验 —— 【多GPU并行+单卡串行】最终版
# 分配规则：
# GPU0: ABC  |  GPU1: A_only  |  GPU2: B_only  |  GPU3: C_only
# 每张卡内部：依次跑 96 → 192 → 336 → 720 预测长度
# ============================================================
mkdir -p logs

# 公共参数
COMMON="--task_name finetune \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1 \
    --data ETTh1 \
    --pretrain_data ETTh1 \
    --features M \
    --seq_len 336 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --n_heads 8 \
    --d_model 16 \
    --d_ff 32 \
    --positive_nums 3 \
    --mask_rate 0.5 \
    --learning_rate 0.0001 \
    --batch_size 16 \
    --train_epochs 20 \
    --patience 5 \
    --dropout 0.1 \
    --pretrain_epochs 50 \
    --lambda_prob 0.05 --lambda_scale 0.05 --alpha_con 0.1"

# 预测长度列表
PREDS=(96 192 336 720)

# ============================================================
# GPU 0 👉 跑 ABC 全组件（后台并行）
# ============================================================
(
for PRED_LEN in "${PREDS[@]}"; do
    echo "🚀 GPU0 | ABC | pred_len=$PRED_LEN 开始运行"
    CUDA_VISIBLE_DEVICES=1 python -u run.py \
        --model FAT_interPDN_v2 \
        --exp_name interPDN_v2_ABC_$PRED_LEN \
        --transfer_expname interPDN_v2_ABC \
        --pred_len $PRED_LEN \
        --freeze 1 \
        --use_comp_a 1 --use_comp_b 1 --use_comp_c 1 \
        $COMMON > logs/interPDN_v2_ABC_$PRED_LEN.log 2>&1
done
echo "✅ GPU0 所有任务完成！"
) &

# ============================================================
# GPU 1 👉 跑 A_only 组件（后台并行）
# ============================================================
(
for PRED_LEN in "${PREDS[@]}"; do
    echo "🚀 GPU1 | A_only | pred_len=$PRED_LEN 开始运行"
    CUDA_VISIBLE_DEVICES=2 python -u run.py \
        --model FAT_interPDN_v2 \
        --exp_name interPDN_v2_A_only_$PRED_LEN \
        --transfer_expname interPDN_v2_A_only \
        --pred_len $PRED_LEN \
        --freeze 1 \
        --use_comp_a 1 --use_comp_b 0 --use_comp_c 0 \
        $COMMON > logs/interPDN_v2_A_only_$PRED_LEN.log 2>&1
done
echo "✅ GPU1 所有任务完成！"
) &

# ============================================================
# GPU 2 👉 跑 B_only 组件（后台并行）
# ============================================================
(
for PRED_LEN in "${PREDS[@]}"; do
    echo "🚀 GPU2 | B_only | pred_len=$PRED_LEN 开始运行"
    CUDA_VISIBLE_DEVICES=3 python -u run.py \
        --model FAT_interPDN_v2 \
        --exp_name interPDN_v2_B_only_$PRED_LEN \
        --transfer_expname interPDN_v2_B_only \
        --pred_len $PRED_LEN \
        --freeze 1 \
        --use_comp_a 0 --use_comp_b 1 --use_comp_c 0 \
        $COMMON > logs/interPDN_v2_B_only_$PRED_LEN.log 2>&1
done
echo "✅ GPU2 所有任务完成！"
) &

# ============================================================
# GPU 3 👉 跑 C_only 组件（后台并行）
# ============================================================
(
for PRED_LEN in "${PREDS[@]}"; do
    echo "🚀 GPU3 | C_only | pred_len=$PRED_LEN 开始运行"
    CUDA_VISIBLE_DEVICES=4 python -u run.py \
        --model FAT_interPDN_v2 \
        --exp_name interPDN_v2_C_only_$PRED_LEN \
        --transfer_expname interPDN_v2_C_only \
        --pred_len $PRED_LEN \
        --freeze 1 \
        --use_comp_a 0 --use_comp_b 0 --use_comp_c 1 \
        $COMMON > logs/interPDN_v2_C_only_$PRED_LEN.log 2>&1
done
echo "✅ GPU3 所有任务完成！"
) &

# 等待所有GPU任务全部结束
wait

echo "============================================="
echo "🎉 所有实验全部运行完成！"
echo "📊 日志文件：logs/ 目录下"
echo "🧠 模型权重：results/checkpoints/ 目录下（按pred_len隔离）"
echo "============================================="