import os
import random
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.stats import linregress
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score
import matplotlib.pyplot as plt
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 移除硬编码的VAR_NUM和DATA_COLS，改为从数据中动态获取
# 数据配置（仅保留路径和窗口参数，其他动态生成）
DATA_URL = "./dataset/weather/weather.csv"  # 改为你的数据集路径（如weather.csv）
SEQ_LEN = 96  # 可根据数据集调整
PRED_LEN = 336
MASK_RATIO = 0.15

# 模型和训练配置不变
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3
DROPOUT = 0.1
BATCH_SIZE = 16
PRETRAIN_EPOCHS = 1
FINETUNE_EPOCHS = 1
LR = 1e-4


# -------------------------- 2. 多变量数据预处理（核心：单变量拆解+单独处理） --------------------------
class MultivarETTDataset(Dataset):
    def __init__(self, data, explicit_list, seq_len, pred_len=0, is_pretrain=True):
        self.data = data  # [total_len, var_num]（var_num动态）
        self.explicit_list = explicit_list  # [var_num, total_len, 1]
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.is_pretrain = is_pretrain
        self.total_len = len(data)
        self.var_num = data.shape[1]  # 动态获取特征数

    def __len__(self):
        return self.total_len - self.seq_len - (self.pred_len if not self.is_pretrain else 0) # 预训练的时候返回0，微调的时候问号

    def __getitem__(self, idx):
        '''
        预训练的时候，返回seq_x,微调的时候，返回
        '''
        if self.is_pretrain:
            # 预训练：返回多变量窗口（形状：[seq_len, var_num]）
            seq_x = self.data[idx:idx + self.seq_len]  # [seq_len, var_num]
            return torch.tensor(seq_x, dtype=torch.float32).unsqueeze(-1)  # [seq_len, var_num, 1]（适配单变量编码器）
        else:
            # 下游任务：返回「多变量输入窗口 + 每个变量的显式特征 + 多变量标签」
            seq_x_data = self.data[idx:idx + self.seq_len]  # [seq_len, var_num]
            # 每个变量的显式特征（取标签段的显式特征）
            seq_x_explicit = []
            for var_idx in range(self.var_num):
                # 对每一个特征channel[i],截取一段长度（包括序列到预测的长度）
                exp = self.explicit_list[var_idx][idx + self.seq_len:idx + self.seq_len + self.pred_len] # 从未来的pred_len上去提取显示特征
                seq_x_explicit.append(exp) # 将提取到的显示特征进行存储
            seq_x_explicit = np.stack(seq_x_explicit, axis=1)  # [pred_len, var_num, 1]
            # 多变量标签
            seq_y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]  # [pred_len, var_num], 返回的是真实的pred_len序列

            return (torch.tensor(seq_x_data, dtype=torch.float32).unsqueeze(-1),  # [seq_len, var_num, 1]
                    torch.tensor(seq_x_explicit, dtype=torch.float32),  # [pred_len, var_num, 1]
                    torch.tensor(seq_y, dtype=torch.float32).unsqueeze(-1))  # [pred_len, var_num, 1]


def extract_multivar_explicit(raw_data):
    """动态为每个特征提取显式特征（无需修改，已兼容任意var_num）"""
    total_len, var_num = raw_data.shape  # 从输入数据获取特征数
    explicit_list = []
    for var_idx in range(var_num):
        # 原有逻辑不变（循环次数由var_num动态决定）
        var_raw = raw_data[:, var_idx].reshape(-1, 1)  # 这里的raw_data有没有去除掉时间特征维度？
        explicit = np.zeros_like(var_raw)
        # 1. 周期特征（每个变量共享ETTh1日周期24）
        T = 24 # 定义周期？ 应该自己提取?
        t = np.arange(total_len)
        A = np.std(var_raw) * 0.6  # 每个变量单独计算振幅
        B = np.mean(var_raw)
        phi = np.pi / 4
        periodic = A * np.sin(2 * np.pi * t / T + phi) + B
        explicit += periodic

        # 2. 趋势特征（每个变量单独拟合）
        window_size = T * 3
        trend = np.zeros_like(var_raw)
        for i in range(total_len):
            win_start = max(0, i - window_size)
            win_t = np.arange(win_start, i)
            win_data = var_raw[win_start:i]
            if len(win_t) < 10:
                trend[i] = B
                continue
            slope, intercept, _, _, _ = linregress(win_t, win_data.reshape(-1))
            trend[i] = slope * i + intercept
        explicit += trend
        explicit_list.append(explicit)
    explicit_list = np.stack(explicit_list, axis=0)
    print(f"多变量显式特征提取完成 | 变量数={var_num} | 周期=24")  # 周期可改为动态获取（见扩展建议）
    return explicit_list


def prepare_multivar_data(data_url):
    """修改为从输入路径加载数据，并动态获取特征数"""
    # 1. 加载数据（自动获取列名和特征数）
    df = pd.read_csv(data_url)
    # 假设第一列是时间戳（如有），则剔除；否则直接使用所有列
    if 'date' in df.columns or 'time' in df.columns:
        data_cols = [col for col in df.columns if col not in ['date', 'time']]
    else:
        data_cols = df.columns.tolist()  # 动态获取特征列名
    raw_data = df[data_cols].values  # [total_len, var_num]，var_num = len(data_cols)
    var_num = raw_data.shape[1]  # 动态获取特征数
    print(f"数据集加载完成 | 特征数={var_num} | 样本数={len(raw_data)} | 特征列={data_cols}")

    # 2. 显式特征和残差提取（已兼容动态var_num）
    explicit_list = extract_multivar_explicit(raw_data)  # [var_num, total_len, 1]
    residual_list = []
    for var_idx in range(var_num):
        var_raw = raw_data[:, var_idx].reshape(-1, 1)
        var_explicit = explicit_list[var_idx]
        residual_list.append(var_raw - var_explicit)
    residual_list = np.stack(residual_list, axis=0)  # [var_num, total_len, 1] # 这里按照axis堆叠的含义是什么？

    # 3. 每个特征单独归一化（动态循环var_num次）
    scaler_dict = {}
    raw_norm = np.zeros_like(raw_data)
    explicit_norm_list = np.zeros_like(explicit_list)
    residual_norm_list = np.zeros_like(residual_list)
    for var_idx in range(var_num):
        # 原有逻辑不变（每个特征单独归一化）
        scaler_raw = MinMaxScaler(feature_range=(0, 1))
        raw_norm[:, var_idx] = scaler_raw.fit_transform(raw_data[:, var_idx].reshape(-1, 1)).reshape(-1)
        scaler_exp = MinMaxScaler(feature_range=(0, 1))
        explicit_norm_list[var_idx] = scaler_exp.fit_transform(explicit_list[var_idx])
        scaler_res = MinMaxScaler(feature_range=(0, 1))
        residual_norm_list[var_idx] = scaler_res.fit_transform(residual_list[var_idx])
        scaler_dict[var_idx] = {"raw": scaler_raw, "explicit": scaler_exp, "residual": scaler_res}

    # 4. 调整显式/残差形状（兼容动态var_num）
    explicit_norm = np.transpose(explicit_norm_list, (1, 0, 2))  # [total_len, var_num, 1]
    residual_norm = np.transpose(residual_norm_list, (1, 0, 2))  # [total_len, var_num, 1]

    # 5. 划分训练/测试集（8:2）
    train_len = int(len(raw_norm) * 0.8)
    # 生成数据集（动态传入var_num相关参数）
    datasets = {
        "pretrain_raw": MultivarETTDataset(
            raw_norm[:train_len], explicit_norm_list, SEQ_LEN, is_pretrain=True
        ),
        "pretrain_residual": MultivarETTDataset(
            residual_norm[:train_len, :, 0], residual_norm_list, SEQ_LEN, is_pretrain=True
        ),
        "test_raw": MultivarETTDataset(
            raw_norm, explicit_norm_list, SEQ_LEN, PRED_LEN, is_pretrain=False
        ),
        "test_residual": MultivarETTDataset(
            residual_norm[:, :, 0], residual_norm_list, SEQ_LEN, PRED_LEN, is_pretrain=False
        ),
        "test_fusion": MultivarETTDataset(
            residual_norm[:, :, 0], residual_norm_list, SEQ_LEN, PRED_LEN, is_pretrain=False
        )
    }

    # 6. 生成DataLoader
    dataloaders = {
        "pretrain_raw": DataLoader(datasets["pretrain_raw"], BATCH_SIZE, shuffle=True),
        "pretrain_residual": DataLoader(datasets["pretrain_residual"], BATCH_SIZE, shuffle=True),
        "test_raw": DataLoader(datasets["test_raw"], BATCH_SIZE, shuffle=False),
        "test_residual": DataLoader(datasets["test_residual"], BATCH_SIZE, shuffle=False),
        "test_fusion": DataLoader(datasets["test_fusion"], BATCH_SIZE, shuffle=False)
    }

    return dataloaders, scaler_dict, (explicit_norm, residual_norm, var_num, data_cols)
    # 返回var_num和data_cols，供后续模型使用


# -------------------------- 3. 多变量适配的SimMTM编码器（核心：单变量并行处理） --------------------------
class MultivarSimMTMEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout, var_num):
        super().__init__()
        self.var_num = var_num  # 动态特征数
        self.encoders = nn.ModuleList([
            SimMTMEncoder(d_model, n_heads, n_layers, dropout) for _ in range(var_num)  # 按特征数创建编码器
        ])

    def forward(self, x, mask_ratio=MASK_RATIO):
        """
        多变量编码：对每个变量单独编码，再拼接结果
        - x: [batch, seq_len, var_num, 1]（多变量输入，最后一维为特征维）
        - return: recon_list（每个变量的重建结果）, mask_list（每个变量的掩码）
        """
        recon_list = []
        mask_list = []
        # 对每个变量单独编码
        for var_idx in range(self.var_num):
            # 提取单个变量的输入：[batch, seq_len, 1]
            x_var = x[:, :, var_idx, :]
            # 单个变量编码
            recon_var, mask_var = self.encoders[var_idx](x_var, mask_ratio)
            recon_list.append(recon_var)  # [batch, seq_len, 1]
            mask_list.append(mask_var)  # [batch, seq_len, 1]
        # 拼接所有变量的结果：[batch, seq_len, var_num, 1]
        recon = torch.stack(recon_list, dim=2)  # [batch, seq_len, var_num, 1]
        mask = torch.stack(mask_list, dim=2)  # [batch, seq_len, var_num, 1]
        return recon, mask


# 复用单变量编码器（无需修改）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, device=DEVICE)
        position = torch.arange(0, max_len, dtype=torch.float, device=DEVICE).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)).to(DEVICE)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]


class SimMTMEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.decoder = nn.Linear(d_model, 1)

    def generate_mask(self, x, mask_ratio):
        batch_size, seq_len, _ = x.shape
        mask = torch.ones(batch_size, seq_len, 1, device=DEVICE)
        mask_num = int(seq_len * mask_ratio)
        for i in range(batch_size):
            mask_idx = random.sample(range(seq_len), mask_num)
            mask[i, mask_idx] = 0
        return mask

    def forward(self, x, mask_ratio=MASK_RATIO):
        x_emb = self.embedding(x)  # [batch, seq_len, d_model]
        x_emb = self.pos_enc(x_emb)
        mask = self.generate_mask(x, mask_ratio)
        x_masked = x_emb * mask
        enc_out = self.encoder(x_masked)
        recon = self.decoder(enc_out)  # [batch, seq_len, 1]
        return recon, mask


def pretrain_multivar_model(model, dataloader, epochs, lr, save_path):
    """多变量预训练：对每个变量的重建损失求和"""
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            # batch形状：[batch, seq_len, var_num, 1]
            batch = batch.to(DEVICE)
            recon, mask = model(batch)

            # 计算每个变量的掩码损失，求和（多变量总损失）
            loss = 0.0
            for var_idx in range(model.var_num):
                # 提取单个变量的重建结果和掩码
                recon_var = recon[:, :, var_idx, :]
                mask_var = mask[:, :, var_idx, :]
                batch_var = batch[:, :, var_idx, :]
                # 累加单个变量的损失
                loss += criterion(recon_var * (1 - mask_var), batch_var * (1 - mask_var))
            # 平均到每个变量（避免损失过大）
            loss /= model.var_num

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"多变量预训练Epoch {epoch + 1}/{epochs} | 平均损失: {avg_loss:.6f}")
    torch.save(model.state_dict(), save_path)
    print(f"多变量预训练模型保存至: {save_path}\n")
    return model


# -------------------------- 4. 多变量下游预测（核心：单变量预测后重组维度） --------------------------
class MultivarPredictor(nn.Module):
    def __init__(self, encoder, d_model, pred_len, var_num):
        super().__init__()
        self.encoder = encoder
        self.var_num = var_num  # 动态特征数
        self.pred_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, pred_len)
            ) for _ in range(var_num)  # 按特征数创建预测头
        ])

    def forward(self, x):
        """
        多变量预测：每个变量单独预测，再重组维度
        - x: [batch, seq_len, var_num, 1]
        - return: [batch, var_num, pred_len]（按你需求的维度）
        """
        pred_list = []
        for var_idx in range(self.var_num):
            x_var = x[:, :, var_idx, :]  # [batch, seq_len, 1]
            # 仅编码器部分冻结（缩小no_grad范围）
            with torch.no_grad():
                x_emb = self.encoder.encoders[var_idx].embedding(x_var)
                x_emb = self.encoder.encoders[var_idx].pos_enc(x_emb)
                enc_out = self.encoder.encoders[var_idx].encoder(x_emb)
                feat_var = enc_out[:, -1, :]  # [batch, d_model]（无需梯度）

            # 预测头计算移出no_grad，确保梯度传播
            pred_var = self.pred_heads[var_idx](feat_var)  # [batch, pred_len]
            pred_list.append(pred_var)

        # 重组维度
        pred = torch.stack(pred_list, dim=1)  # [batch, var_num, pred_len]
        return pred


class MultivarFusionPredictor(nn.Module):
    def __init__(self, encoder, d_model, pred_len, var_num):
        super().__init__()
        self.encoder = encoder
        self.var_num = var_num  # 动态特征数
        self.fusion_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model + d_model // 2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, pred_len)
            ) for _ in range(var_num)  # 按特征数创建融合块
        ])
        self.explicit_projs = nn.ModuleList([
            nn.Linear(1, d_model // 2) for _ in range(var_num)  # 按特征数创建投影层
        ])

    def forward(self, x_residual, x_explicit):
        pred_list = []
        for var_idx in range(self.var_num):
            # 1. 提取残差特征（仅编码器部分冻结）
            with torch.no_grad():  # 缩小no_grad范围至编码器操作
                x_res_var = x_residual[:, :, var_idx, :]  # [batch, seq_len, 1]
                x_emb = self.encoder.encoders[var_idx].embedding(x_res_var)
                x_emb = self.encoder.encoders[var_idx].pos_enc(x_emb)
                enc_out = self.encoder.encoders[var_idx].encoder(x_emb)
                res_feat = enc_out[:, -1, :]  # [batch, d_model]（无需梯度）

            # 2. 处理显式特征（投影层需要梯度，移出no_grad上下文）
            x_exp_var = x_explicit[:, :, var_idx, :]  # [batch, pred_len, 1]
            exp_feat = self.explicit_projs[var_idx](x_exp_var.mean(dim=1))  # 现在有梯度了

            # 3. 融合+预测（融合头需要梯度，移出no_grad上下文）
            fusion_feat = torch.cat([res_feat, exp_feat], dim=1)
            pred_var = self.fusion_blocks[var_idx](fusion_feat)  # 现在有梯度了
            pred_list.append(pred_var)

        # 重组维度
        pred = torch.stack(pred_list, dim=1)
        return pred


def finetune_multivar(dataloader_train, dataloader_test, encoder_path, scaler_dict, var_num, is_fusion=False):
    # 1. 加载编码器（传入动态var_num）
    encoder = MultivarSimMTMEncoder(D_MODEL, N_HEADS, N_LAYERS, DROPOUT, var_num).to(DEVICE)
    encoder.load_state_dict(torch.load(encoder_path))

    # 2. 初始化预测器（传入动态var_num）
    if is_fusion:
        model = MultivarFusionPredictor(encoder, D_MODEL, PRED_LEN, var_num).to(DEVICE)
        optimizer = optim.Adam([
            {'params': model.fusion_blocks.parameters()},
            {'params': model.explicit_projs.parameters()}
        ], lr=LR)
    else:
        model = MultivarPredictor(encoder, D_MODEL, PRED_LEN, var_num).to(DEVICE)
        pred_head_params = []
        for head in model.pred_heads:
            pred_head_params.extend(head.parameters())
        optimizer = optim.Adam(pred_head_params, lr=LR)

    criterion = nn.MSELoss()
    model.train()

    # 3. 微调
    for epoch in range(FINETUNE_EPOCHS):
        total_loss = 0.0
        for batch in dataloader_train:
            if is_fusion:
                x_res, x_exp, y = batch
                x_res, x_exp, y = x_res.to(DEVICE), x_exp.to(DEVICE), y.to(DEVICE)
                pred = model(x_res, x_exp)
                # 融合组：用x_res的size获取批次大小
                batch_size = x_res.size(0)
            else:
                x, _, y = batch
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                # 非融合组：用x的size获取批次大小
                batch_size = x.size(0)

            # 标签形状调整
            y = y.squeeze(-1).transpose(1, 2)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 用统一的batch_size累加损失
            total_loss += loss.item() * batch_size

    # 4. 评估（每个变量单独反归一化，计算均值MAE/RMSE）
    model.eval()
    all_pred = []
    all_true = []
    with torch.no_grad():
        for batch in dataloader_test:
            if is_fusion:
                x_res, x_exp, y = batch
                x_res, x_exp, y = x_res.to(DEVICE), x_exp.to(DEVICE), y.to(DEVICE)
                pred = model(x_res, x_exp)
            else:
                x, _, y = batch
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)

            all_pred.append(pred.cpu().numpy())  # [batch, var_num, pred_len]
            # 标签调整：[batch, pred_len, var_num, 1] → [batch, var_num, pred_len]
            y = y.squeeze(-1).transpose(1, 2).cpu().numpy()
            all_true.append(y)

    # 拼接所有批次
    all_pred = np.concatenate(all_pred, axis=0)  # [total_test, var_num, pred_len]
    all_true = np.concatenate(all_true, axis=0)  # [total_test, var_num, pred_len]

    # 每个变量单独反归一化并计算指标
    mae_list = []
    rmse_list = []
    for var_idx in range(var_num):
        # 反归一化（用每个变量自己的原始数据归一化器）
        scaler = scaler_dict[var_idx]["raw"]
        # 调整形状：[total_test×pred_len, 1]（适配MinMaxScaler）
        pred_var = all_pred[:, var_idx, :].reshape(-1, 1)
        true_var = all_true[:, var_idx, :].reshape(-1, 1)
        pred_var_inv = scaler.inverse_transform(pred_var)
        true_var_inv = scaler.inverse_transform(true_var)
        # 计算指标
        mae = mean_absolute_error(true_var_inv, pred_var_inv)
        rmse = np.sqrt(mean_squared_error(true_var_inv, pred_var_inv))
        mae_list.append(mae)
        rmse_list.append(rmse)

    # 返回所有变量的均值（整体性能）
    avg_mae = np.mean(mae_list)
    avg_rmse = np.mean(rmse_list)
    print(f"该组所有变量平均 | MAE: {avg_mae:.2f}, RMSE: {avg_rmse:.2f}")
    return avg_mae, avg_rmse


# -------------------------- 5. 多变量弱异常检测（简化，验证融合特征有效性） --------------------------
def add_multivar_weak_anomaly(data):
    """多变量弱异常插入：每个变量独立插入异常"""
    data = data.copy()  # [total_len, var_num]
    total_len, var_num = data.shape
    anomaly_ratio = 0.03
    anomaly_num = int(total_len * anomaly_ratio)
    label = np.zeros((total_len, var_num))  # [total_len, var_num]（每个变量的异常标签）

    for var_idx in range(var_num):
        # 每个变量单独插入异常
        anomaly_starts = random.sample(range(total_len - 5), anomaly_num)
        for start in anomaly_starts:
            end = start + 5
            # 弱异常：微小波动
            noise = np.random.normal(0, 0.03, size=5)
            data[start:end, var_idx] = data[start:end, var_idx] * (1.05 + noise) + noise
            data[start:end, var_idx] = np.clip(data[start:end, var_idx], 0, 1)
            label[start:end, var_idx] = 1  # 标记异常

    return data, label


def multivar_anomaly_eval(encoder_raw_path, encoder_res_path, raw_norm, residual_norm, explicit_norm, scaler_dict):
    """多变量异常检测：对比纯原始特征 vs 残差+显式融合特征"""
    # 1. 生成带异常的多变量数据
    data_anomaly, label = add_multivar_weak_anomaly(raw_norm)  # [total_len, var_num]
    print(f"多变量异常数据形状: {data_anomaly.shape}, 异常标签形状: {label.shape}")

    # 2. 提取两种多变量特征（纯原始 / 残差+显式融合）
    def extract_multivar_feat(encoder_path, data_input, explicit_input=None, is_fusion=False):
        """提取多变量特征：每个变量单独提取后拼接"""
        encoder = MultivarSimMTMEncoder(D_MODEL, N_HEADS, N_LAYERS, DROPOUT, VAR_NUM).to(DEVICE)
        encoder.load_state_dict(torch.load(encoder_path))
        encoder.eval()

        feats = []
        total_len = data_input.shape[0]
        with torch.no_grad():
            for i in range(total_len - SEQ_LEN):
                # 输入窗口：[1, seq_len, var_num, 1]（batch=1）
                seq = data_input[i:i + SEQ_LEN].reshape(1, SEQ_LEN, VAR_NUM, 1)
                seq = torch.tensor(seq, dtype=torch.float32).to(DEVICE)

                # 提取每个变量的特征
                var_feats = []
                for var_idx in range(VAR_NUM):
                    x_var = seq[:, :, var_idx, :]
                    x_emb = encoder.encoders[var_idx].embedding(x_var)
                    x_emb = encoder.encoders[var_idx].pos_enc(x_emb)
                    enc_out = encoder.encoders[var_idx].encoder(x_emb)
                    feat_var = enc_out[:, -1, :].cpu().numpy()  # [1, d_model]

                    # 融合显式特征（若需要）
                    if is_fusion:
                        exp_var = explicit_input[i:i + SEQ_LEN, var_idx, :].reshape(1, SEQ_LEN, 1)
                        exp_var = torch.tensor(exp_var, dtype=torch.float32).to(DEVICE)
                        exp_feat = torch.mean(exp_var, dim=1).cpu().numpy()  # [1, 1]
                        feat_var = np.concatenate([feat_var, exp_feat], axis=1)  # [1, d_model+1]

                    var_feats.append(feat_var)

                # 拼接所有变量的特征：[1, var_num×feat_dim]
                window_feat = np.concatenate(var_feats, axis=1)
                feats.append(window_feat)

        return np.concatenate(feats, axis=0)  # [total_test_window, var_num×feat_dim]

    # 提取两组特征
    feat_raw = extract_multivar_feat(encoder_raw_path, data_anomaly)
    feat_fusion = extract_multivar_feat(encoder_res_path, residual_norm[:, :, 0], explicit_norm, is_fusion=True)

    # 3. 生成窗口级异常标签（任一变量异常则窗口异常）
    window_labels = []
    for i in range(len(data_anomaly) - SEQ_LEN):
        window_label = 1 if np.any(label[i:i + SEQ_LEN] == 1) else 0
        window_labels.append(window_label)
    window_labels = np.array(window_labels)

    # 4. 训练分类器评估
    X_train, X_test, y_train, y_test = train_test_split(feat_raw, window_labels, test_size=0.2, random_state=42)
    clf_raw = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf_raw.fit(X_train, y_train)
    f1_raw = f1_score(y_test, clf_raw.predict(X_test))

    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(feat_fusion, window_labels, test_size=0.2,
                                                                random_state=42)
    clf_fusion = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf_fusion.fit(X_train_f, y_train_f)
    f1_fusion = f1_score(y_test_f, clf_fusion.predict(X_test_f))

    print(f"纯原始特征组 | F1-Score: {f1_raw:.4f}")
    print(f"残差+显式融合特征组 | F1-Score: {f1_fusion:.4f}")
    return f1_raw, f1_fusion


# -------------------------- 6. 主函数（多变量完整流程） --------------------------
if __name__ == "__main__":
    # 1. 准备数据（传入数据集路径，动态获取特征数）
    dataloaders, scaler_dict, (explicit_norm, residual_norm, var_num, data_cols) = prepare_multivar_data(
        data_url=DATA_URL  # 可改为任意数据集路径（如"/data/weather.csv"）
    )

    # 2. 预训练（传入动态var_num）
    print("=== 开始多变量预训练（纯原始数据组）===")
    model_raw = MultivarSimMTMEncoder(D_MODEL, N_HEADS, N_LAYERS, DROPOUT, var_num).to(DEVICE)
    model_raw = pretrain_multivar_model(
        model_raw, dataloaders["pretrain_raw"], PRETRAIN_EPOCHS, LR,
        save_path="multivar_simmtm_raw.pth"
    )

    print("=== 开始多变量预训练（纯残差数据组）===")
    model_residual = MultivarSimMTMEncoder(D_MODEL, N_HEADS, N_LAYERS, DROPOUT, var_num).to(DEVICE)
    model_residual = pretrain_multivar_model(
        model_residual, dataloaders["pretrain_residual"], PRETRAIN_EPOCHS, LR,
        save_path="multivar_simmtm_residual.pth"
    )

    # 3. 下游预测（传入动态var_num）
    print("\n=== 下游任务1：多变量长期预测评估 ===")
    print("\n【1. 纯原始数据组】")
    mae_raw, rmse_raw = finetune_multivar(
        dataloaders["test_raw"], dataloaders["test_raw"],
        encoder_path="multivar_simmtm_raw.pth",
        scaler_dict=scaler_dict,
        var_num=var_num,  # 动态传入
        is_fusion=False
    )

    print("\n【2. 纯残差数据组】")
    mae_res, rmse_res = finetune_multivar(
        dataloaders["test_residual"], dataloaders["test_residual"],
        encoder_path="multivar_simmtm_residual.pth",
        scaler_dict=scaler_dict,
        var_num=var_num,  # 动态传入
        is_fusion=False
    )

    print("\n【3. 残差+显式融合组】")
    mae_fusion, rmse_fusion = finetune_multivar(
        dataloaders["test_fusion"], dataloaders["test_fusion"],
        encoder_path="multivar_simmtm_residual.pth",
        scaler_dict=scaler_dict,
        var_num=var_num,  # 动态传入
        is_fusion=True
    )

    # # 4. 异常检测（传入动态var_num）
    # print("\n=== 下游任务2：多变量弱异常检测评估 ===")
    # f1_raw, f1_fusion = multivar_anomaly_eval(
    #     encoder_raw_path="multivar_simmtm_raw.pth",
    #     encoder_res_path="multivar_simmtm_residual.pth",
    #     raw_norm=dataloaders["test_raw"].dataset.data,
    #     residual_norm=residual_norm,
    #     explicit_norm=explicit_norm,
    #     scaler_dict=scaler_dict,
    #     var_num=var_num  # 动态传入
    # )

    # 5. 多变量实验结论
    print("\n=== 多变量实验结论 ===")
    fusion_better_pred = mae_fusion < mae_raw and rmse_fusion < rmse_raw
    # fusion_better_anomaly = f1_fusion > f1_raw
    residual_worse = mae_res > mae_raw and rmse_res > rmse_raw

    if fusion_better_pred and residual_worse:
        print("✅ 多变量核心观点验证成功！")
        print(f"1. 融合组平均MAE {mae_fusion:.2f} < 原始组 {mae_raw:.2f}，预测更优；")
        # print(f"2. 融合组F1 {f1_fusion:.4f} > 原始组 {f1_raw:.4f}，异常检测更优；")
        print(f"3. 纯残差组MAE {mae_res:.2f} > 原始组，证明需结合显式特征；")
        print("4. 多变量按单变量并行处理，符合需求且效果可控。")
    else:
        print("❌ 部分结论未验证，建议调整：")
        if not fusion_better_pred:
            print("- 增大融合头中显式特征的权重（如调整Linear维度）；")
        if not fusion_better_anomaly:
            print("- 降低异常波动幅度（如1.02倍而非1.05倍）；")
        if not residual_worse:
            print("- 检查显式特征提取是否不足（如增大周期振幅）。")