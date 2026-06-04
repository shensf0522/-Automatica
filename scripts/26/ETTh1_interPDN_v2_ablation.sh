#!/bin/bash
# ============================================================
# FAT_interPDN_v2 消融实验脚本 — ETTh1 预训练 (多GPU并行)
# 运行方式: bash scripts/pretrain/ETT_script/ETTh1_interPDN_v2_ablation.sh
# ============================================================
mkdir -p logs

# 公共参数
COMMON="--task_name pretrain \
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
    --learning_rate 0.001 \
    --batch_size 16 \
    --pretrain_epochs 50"

# ============================================================
# 第一轮: 4个实验并行 (GPU 0/1/2/3)
# ============================================================

echo "========================================="
echo "第一轮: 4个实验并行启动"
echo "========================================="

echo "[GPU 0] Exp 1/5: FAT_interPDN_v2 全组件 (A+B+C)"
CUDA_VISIBLE_DEVICES=1 python -u run.py \
    --model FAT_interPDN_v2 \
    --exp_name interPDN_v2_ABC \
    --use_comp_a 1 --use_comp_b 1 --use_comp_c 1 \
    --lambda_prob 0.05 --lambda_scale 0.05 --alpha_con 0.1 \
    $COMMON > logs/interPDN_v2_ABC.log 2>&1 &

echo "[GPU 1] Exp 2/5: 仅组件 A (双视角概率正则化)"
CUDA_VISIBLE_DEVICES=2 python -u run.py \
    --model FAT_interPDN_v2 \
    --exp_name interPDN_v2_A_only \
    --use_comp_a 1 --use_comp_b 0 --use_comp_c 0 \
    --lambda_prob 0.05 --lambda_scale 0.05 --alpha_con 0.1 \
    $COMMON > logs/interPDN_v2_A_only.log 2>&1 &

echo "[GPU 2] Exp 3/5: 仅组件 B (时-频双视角重建)"
CUDA_VISIBLE_DEVICES=3 python -u run.py \
    --model FAT_interPDN_v2 \
    --exp_name interPDN_v2_B_only \
    --use_comp_a 0 --use_comp_b 1 --use_comp_c 0 \
    --lambda_prob 0.05 --lambda_scale 0.05 --alpha_con 0.1 \
    $COMMON > logs/interPDN_v2_B_only.log 2>&1 &

echo "[GPU 3] Exp 4/5: 仅组件 C (跨尺度一致性)"
CUDA_VISIBLE_DEVICES=4 python -u run.py \
    --model FAT_interPDN_v2 \
    --exp_name interPDN_v2_C_only \
    --use_comp_a 0 --use_comp_b 0 --use_comp_c 1 \
    --lambda_prob 0.05 --lambda_scale 0.05 --alpha_con 0.1 \
    $COMMON > logs/interPDN_v2_C_only.log 2>&1 &

echo "等待第一轮 4 个实验完成..."
wait
echo "第一轮完成!"

# ============================================================
# 第二轮: 最后1个实验
# ============================================================

# echo "========================================="
# echo "[GPU 0] Exp 5/5: Baseline FAT (无新组件)"
# echo "========================================="
# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --model FAT_interPDN_v2 \
#     --exp_name interPDN_v2_baseline \
#     --use_comp_a 0 --use_comp_b 0 --use_comp_c 0 \
#     --lambda_prob 0.05 --lambda_scale 0.05 --alpha_con 0.1 \
#     $COMMON > logs/interPDN_v2_baseline.log 2>&1

# echo "========================================="
# echo "所有消融实验完成! 日志保存在 logs/ 目录下"
# echo "========================================="
