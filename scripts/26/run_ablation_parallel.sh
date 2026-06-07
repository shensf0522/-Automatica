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

echo "=========================================="
echo "Starting all 5 experiments in parallel..."
echo "=========================================="

bash scripts/26/2606070200_B.sh &
PID_B=$!
echo "Option B running with PID: $PID_B"

bash scripts/26/2606070200_C.sh &
PID_C=$!
echo "Option C running with PID: $PID_C"

bash scripts/26/2606070200_D.sh &
PID_D=$!
echo "Option D running with PID: $PID_D"

bash scripts/26/2606070200_E.sh &
PID_E=$!
echo "Option E running with PID: $PID_E"

bash scripts/26/2606070200_F.sh &
PID_F=$!
echo "Option F running with PID: $PID_F"

echo "=========================================="
echo "Waiting for all experiments to complete..."
echo "=========================================="

wait $PID_B
echo "[DONE] Option B has completed."

wait $PID_C
echo "[DONE] Option C has completed."

wait $PID_D
echo "[DONE] Option D has completed."

wait $PID_E
echo "[DONE] Option E has completed."

wait $PID_F
echo "[DONE] Option F has completed."

echo "=========================================="
echo "All ablation experiments (B, C, D, E, F) have completed!"
echo "=========================================="
