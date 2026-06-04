import torch
import torch.nn as nn
from utils.augmentations import augment_positive_test
from utils.tools import ContrastiveWeight, AggregationRebuild, generate_CLLabels, FFT_sim
from utils.losses import AutomaticWeightedLoss
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding
import torch.nn.functional as F
import math

class MDMF(nn.Module):
    """Frequency-enhanced Multi-Scale Decomposable Mixing block."""
    def __init__(self, input_shape, patch=12, k=3, c=2, layernorm=True):
        # 输入包括 input_shape、patch（滑动窗口长度）、k（多尺度层数）、c（层间比例系数）以及是否使用 layernorm
        super(MDMF, self).__init__()
        self.seq_len = input_shape[0]
        self.patch = patch
        self.T = self.seq_len - patch + 1   # 计算滑窗展开后的 patch 数量

        self.k = k

        self.time_linear = nn.Linear(self.patch, 1)    # 用于处理时间域 patch
        self.freq_linear = nn.Linear(self.patch, 1)    # 用于处理频域（FFT 实部）patch
        self.fuse_linear = nn.Linear(2, 1)   # 将时间与频率特征拼接后的二维向量映射为单值

        if self.k > 0:
            self.k_list = [c ** i for i in range(k, 0, -1)]
            self.avg_pools = nn.ModuleList([
                nn.AvgPool1d(kernel_size=k, stride=k) for k in self.k_list
            ])
            self.linears = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.T // k, self.T // k),
                        nn.GELU(),
                        nn.Linear(self.T // k, self.T * c // k),
                    )
                    for k in self.k_list
                ]
            )

        self.layernorm = layernorm
        if self.layernorm:
            self.norm = nn.BatchNorm1d(input_shape[0] * input_shape[-1])

        weight = torch.ones(1, 1, self.patch) / self.patch
        self.register_buffer("fold_weight", weight)

    def forward(self, x):
        if self.layernorm:
            x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)
        if self.k == 0:
            return x

        b, c, _ = x.shape # batch，channel，seq_len
        patches = x.unfold(dimension=2, size=self.patch, step=1)
        # 沿着seq_lens的维度，从起始点开始，每次取size=patch的窗口，每次移动的距离为1，所以unfold之后的维度变为(batch_size, channels, T, patch)，这里是重叠的，并且unfold返回的Tensor
        # 在内存上不连续的，通过下方的contiguous函数强制在内存中重新复制一份连续的Tensor，便于后续的shape操作

        patches = patches.contiguous().reshape(-1, self.patch) # reshape，要在不复制数据的前提下调整Tensor的view

        # 分别计算时间域特征（线性层）、频域特征（FFT 后取实部再线性层），拼接后二次映射为融合特征，并还原成 [b, c, T]
        time_feat = self.time_linear(patches)
        freq_feat = self.freq_linear(torch.fft.fft(patches, dim=1).real)  # 在dim=1的维度上，也就是对每一行做一维的FFT，即对每个patch的序列做频域变换，数据维度不变，shape：仍然是 (B * C * T, patch)
        fused = self.fuse_linear(torch.cat([time_feat, freq_feat], dim=1))
        fused = fused.view(b, c, self.T)

        # 对融合特征按多尺度池化，逐层用对应线性层上采样并做残差相加，得到最终的 patch 表示 out_patch
        sample_x = [pool(fused) for pool in self.avg_pools]
        sample_x.append(fused)
        for i in range(len(self.k_list)):
            tmp = self.linears[i](sample_x[i])
            sample_x[i + 1] = sample_x[i + 1] + tmp
        out_patch = sample_x[-1]   # 这里的dimension[2]不是seq_len,而是T，out_patch.shape = (b, c, T)

        # 将 patch 表示折叠回原序列
        out_patch = out_patch.reshape(b * c, 1, self.T)
        out_seq = F.conv_transpose1d(out_patch, self.fold_weight, stride=1)   # 使用 conv_transpose1d 与平均权重卷积，实现重叠部分的均值合成，这里的矩阵对应之前的patch的提取
        out_seq = out_seq[:, :, : self.seq_len]
        out_seq = out_seq.view(b, c, self.seq_len)  # 截取到原序列长度 seq_len 并 reshape 为 (b, c, seq_len)
        return out_seq

class CAFIBlock(nn.Module):
    def __init__(self, feature_dim, seq_len, rank=4, dropout=0.1):
        super(CAFIBlock, self).__init__()
        self.rank = rank
        self.seq_len = seq_len

        self.query = nn.Linear(seq_len, rank)
        self.key = nn.Linear(seq_len, rank)
        self.value = nn.Linear(seq_len, seq_len)

        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm([seq_len, feature_dim])

        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # x: [B, L, C]
        x_norm = self.layernorm(x)
        x_t = x_norm.transpose(1, 2)  # [B, C, L]

        Q = self.query(x_t)  # [B, C, r]
        K = self.key(x_t)    # [B, C, r]
        A = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.rank)
        A = torch.softmax(A, dim=-1)
        V = self.value(x_t)  # [B, C, L]

        out = torch.matmul(A, V)  # [B, C, L]
        out = x_t + self.alpha * self.dropout(out)

        out_time = V + self.beta * self.dropout(V)  # 时间维残差增强
        out = out + out_time

        return out.transpose(1, 2)  # [B, L, C]

class CAFI(nn.Module):
    def __init__(self, input_shape, dropout=0.2, patch=12, rank=4, layernorm=True):
        super(CAFI, self).__init__()
        self.seq_len, self.feature_dim = input_shape
        self.patch = patch
        self.rank = rank
        self.layernorm = layernorm

        if self.layernorm:
            self.norm = nn.BatchNorm1d(self.seq_len * self.feature_dim)
        self.norm1 = nn.BatchNorm1d(patch * self.feature_dim)
        self.norm2 = nn.BatchNorm1d(patch * self.feature_dim)

        self.agg = nn.Sequential(
            nn.Linear(self.patch, self.patch),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.block = CAFIBlock(
            feature_dim=self.feature_dim,
            seq_len=self.patch,
            rank=self.rank,
            dropout=dropout
        )

    def forward(self, x):
        # [B, C, L]
        B, C, L = x.shape
        if self.layernorm:
            x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)

        output = torch.zeros_like(x)
        output[:, :, :self.patch] = x[:, :, :self.patch].clone()

        for i in range(self.patch, self.seq_len, self.patch):
            chunk = output[:, :, i - self.patch: i]  # [B, C, patch]
            chunk = self.norm1(torch.flatten(chunk, 1, -1)).reshape(chunk.shape)
            chunk = self.agg(chunk)  # [B, C, patch]
            tmp = chunk + x[:, :, i: i + self.patch]  # 时间维交互
            res = tmp

            tmp = self.norm2(torch.flatten(tmp, 1, -1)).reshape(tmp.shape)
            tmp = tmp.transpose(1, 2)  # [B, patch, C]
            tmp = self.block(tmp)      # CAFI 替代 fc_block
            tmp = tmp.transpose(1, 2)  # [B, C, patch]

            output[:, :, i: i + self.patch] = res + tmp

        return output

# 融合MDMF和CAFI的结果送入到encoder中进行编码
class MDMF_CAFI_fuse(nn.Module):
    def __init__(self, input_shape, patch=12, dropout=0.1, gating_hidden=128):  # input_shape:(seq_len,n_vars)
        super().__init__()
        self.mdmf = MDMF(input_shape, patch=patch)
        self.cafi = CAFI(input_shape, dropout=dropout, patch=patch)

        seq_len, feature_dim = input_shape
        self.gating = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * feature_dim, gating_hidden),
            nn.ReLU(),
            nn.Linear(gating_hidden, 2),   # 两个专家
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        b, c, l = x.shape

        # 两个专家表征
        feat_mdmf = self.mdmf(x)     # (B, C, L)
        feat_cafi = self.cafi(x)     # (B, C, L)

        # Gating 得到两个专家的权重
        gate_input = x.flatten(start_dim=1)  # (B, C*L)
        gates = self.gating(gate_input)      # (B, 2)

        gate_mdmf = gates[:, 0].view(b, 1, 1)  # expand for broadcasting
        gate_cafi = gates[:, 1].view(b, 1, 1)

        # 融合两个专家
        fused_feat = gate_mdmf * feat_mdmf + gate_cafi * feat_cafi    # (B, C, L)

        return fused_feat, gates


class ComplexLinear(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.Linear_real = torch.nn.Linear(in_features, out_features, bias=bias)
        self.Linear_img = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        real_real = self.Linear_real(input.real)
        img_real = self.Linear_img(input.real)
        real_img = self.Linear_real(input.imag)
        img_img = self.Linear_img(input.imag)
        return real_real - img_img + 1j * (real_img + img_real)


class Complex_Dropout(nn.Module):
    def __init__(self, p, inplace=False, size=None, device='cuda'):
        super().__init__()
        self.size = size
        self.device = device
        if self.size is not None:
            self.ones = torch.ones(size)
            if self.device is not None:
                self.ones = self.ones.to(self.device)
        self.real_dropout = nn.Dropout(p=p, inplace=inplace)

    def forward(self, input):
        if self.size is not None:
            return input * self.real_dropout(self.ones)
        else:
            if self.device is not None:
                return input * self.real_dropout(torch.ones(input.size()).to(self.device))
            return input * self.real_dropout(torch.ones(input.size()))


class Flatten_Head(nn.Module):
    def __init__(self, seq_len, d_model, pred_len, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(seq_len * d_model, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # [bs x n_vars x seq_len x d_model]
        x = self.flatten(x)  # [bs x n_vars x (seq_len * d_model)]
        x = self.linear(x)  # [bs x n_vars x seq_len]
        x = self.dropout(x)  # [bs x n_vars x seq_len]
        return x


class Pooler_Head(nn.Module):

    def __init__(self, seq_len, d_model, head_dropout=0):
        super().__init__()
        pn = seq_len * d_model
        dimension = 128
        self.pooler = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(pn, pn // 2),
            nn.BatchNorm1d(pn // 2),
            nn.ReLU(),
            nn.Linear(pn // 2, dimension),
            nn.Dropout(head_dropout),
        )

    def forward(self, x):  # [(bs * n_vars) x seq_len x d_model]
        x = self.pooler(x)  # [(bs * n_vars) x dimension]
        return x


def circular_convolution(self, x, w):
    x = torch.fft.rfft(x, dim=2, norm='ortho')
    w = torch.fft.rfft(w, dim=1, norm='ortho')
    y = x * w
    out = torch.fft.irfft(y, n=self.embed_size, dim=2, norm="ortho")
    return out


# class FreNormLaryer(nn.Module):
#     def __init__(self, scale, embed_size):
#         super(FreNormLaryer, self).__init__()
#         self.embed_size = embed_size
#         self.w = nn.Parameter(scale * torch.randn(1, embed_size))
#
#     def forward(self, x):
#         x = torch.fft.rfft(x, dim=-1, norm='ortho')
#         w = torch.fft.rfft(self.w, dim=1, norm='ortho')
#         y = x * w
#         out = torch.fft.irfft(y, n=self.embed_size, dim=-1, norm="ortho")
#         return out
#
#
# class FreNormLaryer_KB(nn.Module):
#     def __init__(self, n_knlg, input_len, bias=True):
#         super(FreNormLaryer_KB, self).__init__()
#         self.embed_dim = input_len // 2 + 1
#         self.n_knlg = n_knlg  # test: 8 16 32 64
#         self.kb = nn.Parameter(torch.randn(n_knlg, self.embed_dim, dtype=torch.cfloat))
#         self.scaling = self.embed_dim ** -0.5
#         self.in_proj_q = ComplexLinear(self.embed_dim, self.embed_dim, bias=bias)
#         self.in_proj_k = ComplexLinear(self.embed_dim, self.embed_dim, bias=bias)
#         self.in_proj_v = ComplexLinear(self.embed_dim, self.embed_dim, bias=bias)
#         self.out_proj = ComplexLinear(self.embed_dim, self.embed_dim, bias=bias)
#         self.out_dim = input_len
#         # self.cdropout = Complex_Dropout(attn_dropout)
#         # self.w =  nn.Parameter(scale * torch.randn(1, embed_size))
#
#     def retrive_w(self, x):
#         q = self.in_proj_q(x)  # (448, 169)
#         k = self.in_proj_k(self.kb)  # (n_knlg, 169)
#         v = self.in_proj_v(self.kb)
#
#         attn_weights = torch.matmul(q, torch.conj_physical(k).T) * self.scaling  # (448, 8)
#         real = torch.real(attn_weights)
#         attn_weights = F.softmax(real, dim=-1).type(torch.complex64)
#         # attn_weights = self.cdropout(attn_weights)
#
#         w = torch.matmul(attn_weights, v)
#         # w = self.out_proj(w)
#
#         return w
#
#     def forward(self, x):
#         x = torch.fft.rfft(x, dim=-1, norm='ortho')
#         w = self.retrive_w(x)
#         y = x * w
#         out = torch.fft.irfft(y, n=self.out_dim, dim=-1, norm="ortho")
#         return out, w


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):

        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str, mask=None):  # (b, s, n)
        if mode == 'norm':
            if mask:
                self.means = torch.sum(x, dim=1) / torch.sum(mask == 1, dim=1).unsqueeze(1).detach()
                x = x - self.means
                x = x.masked_fill(mask == 0, 0)
                self.stdev = torch.sqrt(torch.sum(x * x, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5).unsqueeze(
                    1).detach()
                x /= self.stdev
                if self.affine:
                    x = x * self.affine_weight
                    x = x + self.affine_bias
                    x = x.masked_fill(mask == 0, 0)
            else:
                self._get_statistics(x)
                x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_type = configs.task_type
        self.task_name = configs.task_name
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.scale = 0.02
        self.revin_layer_encoder = RevIN(configs.enc_in, affine=True, subtract_last=False)
        self.fuse_extractor = MDMF_CAFI_fuse((configs.seq_len, configs.enc_in), patch=configs.patch, dropout=configs.dropout, gating_hidden=128)
        self.embed_size = self.seq_len
        self.hidden_size = configs.hidden_size

        # self.fre_norm_encoder = FreNormLaryer_KB(configs.n_knlg, configs.seq_len)

        self.enc_embedding = DataEmbedding(1, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)  # (b*n*4, seq, 1)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        # inference weight
        self.infer_projection = Pooler_Head(configs.seq_len, configs.d_model, head_dropout=configs.head_dropout)
        self.head_Inference_Re = nn.Linear(128, configs.seq_len // 2 + 1, bias=True)
        self.head_Inference_Img = nn.Linear(128, configs.seq_len // 2 + 1, bias=True)
        # cl weight
        self.cl_projection = Pooler_Head(configs.seq_len, configs.d_model, head_dropout=configs.head_dropout)
        # reconstrution weight
        self.head_pretrain = Flatten_Head(configs.seq_len, configs.d_model, configs.seq_len,
                                          head_dropout=configs.head_dropout)
        # finetube weight

        if self.task_type == "c":
            configs.cls_num = get_cls_num(configs.data)
            self.head_clf = Flatten_Head(configs.seq_len, configs.d_model, configs.cls_num,
                                         head_dropout=configs.head_dropout)
        else:
            self.head_forecast = Flatten_Head(configs.seq_len, configs.d_model, configs.pred_len,
                                              head_dropout=configs.head_dropout)

        self.labels_cl = None
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')
        self.awl = AutomaticWeightedLoss(2)
        self.mse = torch.nn.MSELoss()

    def pretrainWithContrast(self, batch_x):
        bs, seq_len, n_vars = batch_x.shape  # (16,96,21)

        # 1)归一化
        z = self.revin_layer_encoder(batch_x, 'norm')  # 保留batch 和 n_vars，仅在seq 上计算均值，每一个batch的每一个n_vars 计算了一个均值
        x_raw = z.permute(0, 2, 1)  # (b,n,s) (16,21,96)

        # 2) MDMF+CAFI 融合特征
        fused, _ = self.fuse_extractor(x_raw)  # (B,N,S)
        fused_flat = fused.reshape(-1, seq_len)  # (B*N,S) #  （16*21,96）

        # 3)负样本索引
        f_norm = F.normalize(fused_flat.detach(), dim=1)
        sim_mat = torch.matmul(f_norm, f_norm.T)  # (B*N,B*N)
        diag = torch.arange(sim_mat.size(0), device=sim_mat.device)
        sim_mat[diag, diag] = -1.
        negative_index = torch.topk(sim_mat,
                                    k=self.configs.negative_nums,
                                    dim=1).indices

        # 4) 正样本增强 —— 仍复用原函数
        x_pos = augment_positive_test(
            fused_flat, self.configs.mask_rate, self.configs.lm,
            k=self.configs.positive_nums)
        x_pos = x_pos.reshape(-1, seq_len)
        x_raw = x_raw.reshape(-1, seq_len)
        # 5) 拼接全部样本 → Encoder
        x_all = torch.cat([x_raw, x_pos], dim=0)  # ((1+K)B*N,S)

        enc_out = self.enc_embedding(x_all.unsqueeze(-1))
        enc_out, _ = self.encoder(enc_out)

        # 6) 对比损失
        s_enc_out = self.cl_projection(enc_out)
        s_enc_out_norm = F.normalize(s_enc_out, dim=1)  # ((1+K)B*N,128)
        q = s_enc_out_norm[: bs * n_vars]
        k = s_enc_out_norm[bs * n_vars:].reshape(bs * n_vars,
                                       self.configs.positive_nums, -1)

        if self.labels_cl is None:
            self.labels_cl = generate_CLLabels(
                x_raw, self.configs.positive_nums, self.configs.negative_nums)

        pos_sim = torch.matmul(q.unsqueeze(1), k.permute(0, 2, 1)).squeeze()
        if self.configs.negative_nums == 1:
            neg_sim = torch.matmul(q.unsqueeze(1), k[:, 0, :][negative_index].permute(0, 2, 1)).squeeze(-1)
        else:
            neg_sim = torch.matmul(q.unsqueeze(1), k[:, 0, :][negative_index].permute(0, 2, 1)).squeeze()
        sim_mat = torch.cat([pos_sim, neg_sim], dim=-1)
        loss_cl = self.kl(self.log_softmax(sim_mat / self.configs.temperature),
                          self.labels_cl)

        # 7) 重建损失（与原逻辑一致，只是输入编码器不同）
        pos_enc = enc_out[bs * n_vars:].reshape(bs * n_vars,
                                                self.configs.positive_nums, -1)
        hard_neg = pos_enc[:, 0, :][negative_index]
        w_mat = self.softmax(sim_mat / self.configs.temperature)

        rebuild = (w_mat[:, :self.configs.positive_nums].unsqueeze(1) @ pos_enc +
                   w_mat[:, self.configs.positive_nums:].unsqueeze(1) @ hard_neg)
        rebuild = rebuild.reshape(bs, n_vars, seq_len, -1)

        pred = self.head_pretrain(rebuild).reshape(bs, n_vars, -1).permute(0, 2, 1)
        pred = self.revin_layer_encoder(pred, 'denorm')
        loss_rb = self.mse(batch_x, pred)

        loss = self.awl(loss_cl, loss_rb)
        return loss, loss_cl, loss_rb, None, None, None


    def forecast(self, x):
        bs, seq_len, n_vars = x.shape
        z = self.revin_layer_encoder(x, 'norm')
        x_raw = z.permute(0, 2, 1)
        x_raw = x_raw.reshape(-1, seq_len)

        enc_out = self.enc_embedding(x_raw.unsqueeze(-1))
        enc_out, _ = self.encoder(enc_out)

        # Forecast head
        x = self.head_forecast(enc_out)
        x = x.reshape(bs, n_vars, -1).permute(0, 2, 1)
        z = self.revin_layer_encoder(x, 'denorm')
        x = z
        return x

    def forward(self, batch_x):

        if self.task_name == 'pretrain':
            if self.configs.pretrain_mode == "1":
                return self.pretrainWithContrast(batch_x)
            elif self.configs.pretrain_mode == "0":
                return self.pretrain(batch_x)
            else:
                print("ERROR")

        if self.task_name == 'finetune':
            if self.task_type == 'c':
                dec_out = self.clf(batch_x)
                return dec_out
            elif self.task_type == 'r':
                dec_out = self.forecast(batch_x)
                return dec_out
            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")

