#!/bin/bash
# ==============================================================================
# 并行消融实验调度脚本 (80GB GPU, 同时运行5个实验)
# 实验列表:
#   B - 去噪共识重建 (consensus target)
#   C - 混合目标重建 (mix target, alpha=0.5)
#   D - 双Loss限制   (double loss, beta=0.3)
#   E - 噪声惩罚项   (noise penalty, gamma=0.1)
#   F - 多噪声注入   (noise_inject augmentation, raw target)
# ==============================================================================
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# ==============================================================================
# 【已注释】旧的并行消融实验
# ==============================================================================
# echo "=========================================="
# echo "Starting all 5 experiments in parallel..."
# echo "=========================================="
# 
# bash scripts/26/2606070200_B.sh &
# PID_B=$!
# echo "Option B running with PID: $PID_B"
# 
# bash scripts/26/2606070200_C.sh &
# PID_C=$!
# echo "Option C running with PID: $PID_C"
# 
# bash scripts/26/2606070200_D.sh &
# PID_D=$!
# echo "Option D running with PID: $PID_D"
# 
# bash scripts/26/2606070200_E.sh &
# PID_E=$!
# echo "Option E running with PID: $PID_E"
# 
# bash scripts/26/2606070200_F.sh &
# PID_F=$!
# echo "Option F running with PID: $PID_F"
# 
# echo "=========================================="
# echo "Waiting for all experiments to complete..."
# echo "=========================================="
# 
# wait $PID_B
# echo "[DONE] Option B has completed."
# 
# wait $PID_C
# echo "[DONE] Option C has completed."
# 
# wait $PID_D
# echo "[DONE] Option D has completed."
# 
# wait $PID_E
# echo "[DONE] Option E has completed."
# 
# wait $PID_F
# echo "[DONE] Option F has completed."
# 
# echo "=========================================="
# echo "All ablation experiments (B, C, D, E, F) have completed!"
# echo "=========================================="


# ==============================================================================
# 【当前运行】新生成的四个长度与RevIN消融实验
# ==============================================================================
# ==============================================================================
# 【当前运行】新生成的四个长度与RevIN消融实验
# 运行策略：
#   第一阶段：先并行运行前 3 个实验
#   第二阶段：前 3 个全部结束后，再运行最后 1 个实验，避免 OOM
# ==============================================================================
echo "=========================================="
echo "Starting first 3 experiments in parallel..."
echo "=========================================="

bash scripts/26/2606090034_F_seq512_bs8.sh &
PID_1=$!
echo "[RUN] seq=512, bs=8, with revin running with PID: $PID_1"

bash scripts/26/2606090034_F_seq512_bs8_no_revin.sh &
PID_2=$!
echo "[RUN] seq=512, bs=8, no revin running with PID: $PID_2"

bash scripts/26/2606090034_F_seq720_bs4.sh &
PID_3=$!
echo "[RUN] seq=720, bs=4, with revin running with PID: $PID_3"

echo "=========================================="
echo "Waiting for first 3 experiments to complete..."
echo "=========================================="

wait $PID_1
echo "[DONE] seq=512, bs=8, with revin has completed."

wait $PID_2
echo "[DONE] seq=512, bs=8, no revin has completed."

wait $PID_3
echo "[DONE] seq=720, bs=4, with revin has completed."

echo "=========================================="
echo "First 3 experiments have completed."
echo "Now starting the last experiment..."
echo "=========================================="

bash scripts/26/2606090034_F_seq720_bs4_no_revin.sh
echo "[DONE] seq=720, bs=4, no revin has completed."

echo "=========================================="
echo "All 4 new experiments have completed!"
echo "=========================================="
