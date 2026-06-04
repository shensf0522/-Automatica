import torch
import torch.nn as nn
from utils.augmentations import augment_positive_test
from utils.tools import generate_CLLabels, FFT_sim
from utils.losses import AutomaticWeightedLoss
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding
import torch.nn.functional as F

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
        self.linear = nn.Linear(seq_len*d_model, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # [bs x n_vars x seq_len x d_model]
        x = self.flatten(x) # [bs x n_vars x (seq_len * d_model)]
        x = self.linear(x) # [bs x n_vars x seq_len]
        x = self.dropout(x) # [bs x n_vars x seq_len]
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
        x = self.pooler(x) # [(bs * n_vars) x dimension]
        return x

class FreNormLaryer(nn.Module):
    def __init__(self, scale, embed_size):
        super(FreNormLaryer, self).__init__()
        self.embed_size = embed_size
        self.w =  nn.Parameter(scale * torch.randn(1, embed_size))

    def forward(self, x):
        x = torch.fft.rfft(x, dim=-1, norm='ortho')
        w = torch.fft.rfft(self.w, dim=1, norm='ortho')
        y = x * w
        out = torch.fft.irfft(y, n=self.embed_size, dim=-1, norm="ortho")
        return out

class FreNormLayer_KB(nn.Module):
    def __init__(self, n_knlg, input_len, bias=True):
        super(FreNormLayer_KB, self).__init__()
        self.embed_dim = input_len // 2 + 1
        self.n_knlg = n_knlg  # test: 8 16 32 64
        self.kb = nn.Parameter(torch.randn(n_knlg, self.embed_dim, dtype=torch.cfloat))
        self.scaling = self.embed_dim ** -0.5
        self.in_proj_q = ComplexLinear(self.embed_dim, self.embed_dim, bias=bias)
        self.in_proj_k = ComplexLinear(self.embed_dim, self.embed_dim, bias=bias)
        self.in_proj_v = ComplexLinear(self.embed_dim, self.embed_dim, bias=bias)
        self.out_proj = ComplexLinear(self.embed_dim, self.embed_dim, bias=bias)
        self.out_dim = input_len
        # self.cdropout = Complex_Dropout(attn_dropout)
        # self.w =  nn.Parameter(scale * torch.randn(1, embed_size))

    def retrive_w(self, x):
        q = self.in_proj_q(x) # (448, 169)
        k = self.in_proj_k(self.kb) # (n_knlg, 169)
        v = self.in_proj_v(self.kb)

        attn_weights = torch.matmul(q, torch.conj_physical(k).T) * self.scaling #(448, 8)
        real = torch.real(attn_weights)
        attn_weights = F.softmax(real, dim=-1).type(torch.complex64)
        # attn_weights = self.cdropout(attn_weights)
        w = torch.matmul(attn_weights, v) # x从知识库v根据频域分量检索出来的向量
        # w = self.out_proj(w)
        return w

    def forward(self, x):
        x = torch.fft.rfft(x, dim=-1, norm='ortho')
        w = self.retrive_w(x)
        y = x * w
        out = torch.fft.irfft(y, n=self.out_dim, dim=-1, norm="ortho")
        return out, y  # y 表示的是频域增强后的过滤部分

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):

        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str, mask = None):  # (b, s, n)
        if mode == 'norm':
            if mask:
                self.means = torch.sum(x, dim=1) / torch.sum(mask == 1, dim=1).unsqueeze(1).detach()
                x = x - self.means
                x = x.masked_fill(mask == 0, 0)
                self.stdev = torch.sqrt(torch.sum(x * x, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5).unsqueeze(1).detach()
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
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
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
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.scale = 0.02
        self.revin_layer_encoder = RevIN(configs.enc_in, affine=True, subtract_last=False)

        self.embed_size = self.seq_len
        self.hidden_size = configs.hidden_size
        if self.configs.use_kb:
            self.KnowledgeGuide_encoder = FreNormLayer_KB(configs.n_knlg, configs.seq_len)
        else:
            self.KnowledgeGuide_encoder = nn.Identity()  # 直接透传，不做任何处理
        self.enc_embedding = DataEmbedding(1, configs.d_model, configs.embed, configs.freq, configs.dropout)   # (b*n*4, seq, 1)

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
        self.infer_projection = Pooler_Head(configs.seq_len,configs.d_model,head_dropout=configs.head_dropout)
        self.head_Inference_Re = nn.Linear(128, configs.seq_len  // 2 + 1, bias=True)
        self.head_Inference_Img = nn.Linear(128, configs.seq_len // 2 + 1, bias=True)
        # cl weight
        self.cl_projection = Pooler_Head(configs.seq_len,configs.d_model,head_dropout=configs.head_dropout)
        # reconstrution weight
        self.head_pretrain = Flatten_Head(configs.seq_len, configs.d_model, configs.seq_len, head_dropout=configs.head_dropout)
        # finetube weight
        self.head_forecast = Flatten_Head(configs.seq_len, configs.d_model, configs.pred_len, head_dropout=configs.head_dropout)
        self.labels_cl = None
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')
        self.awl = AutomaticWeightedLoss(2)
        self.mse = torch.nn.MSELoss()

    def get_random_negatives(self, batch_size_n, k, device):
        """
        Baseline 策略: 随机选择负样本索引
        """
        # 生成一个随机权重矩阵，对角线置0（不选自己）
        weights = torch.ones(batch_size_n, batch_size_n, device=device)
        weights.fill_diagonal_(0)
        # 无放回采样 k 个索引
        negative_index = torch.multinomial(weights, k, replacement=False)
        return negative_index

    def get_fft_negatives(self, x, k, fft_var):
        """
        策略 B: FFT 相似度负样本 (融合优化版)
        x_normed: (bs, n_vars, seq_len)
        """
        bs, n_vars, seq_len = x.shape

        # 1. [修正] 正确的展平方式，不打乱时间步
        # 输入已经是 (bs, n, seq)，直接展平前两维即可
        if fft_var:
            x_flat = x.reshape(bs * n_vars, seq_len)
        else:
            x_flat = x.permute(0, 2, 1).reshape(bs * n_vars, seq_len)

        # 2. FFT 计算
        fft_result = torch.fft.fft(x_flat)
        magnitude = torch.abs(fft_result)

        # 3. 归一化 (L2 Norm)
        freq_normalized = torch.nn.functional.normalize(magnitude, p=2, dim=1)

        # 4. 计算相似度矩阵 (bs*n, bs*n) - 全正数 [0, 1]
        sim_matrix = torch.matmul(freq_normalized, freq_normalized.T)

        # 5. [采纳] 块级掩码策略 (Block Masking)
        # 我们不仅要屏蔽自己，还要屏蔽同一个 Sample 下的其他变量
        # 这样负样本一定来自"其他的时间片段"，更具区分度

        # 创建一个对角块为 1 的 Mask
        # 也就是 mask[i, j] = 1 表示 i 和 j 属于同一个 batch sample
        mask = torch.zeros((bs * n_vars, bs * n_vars), device=x.device).bool()
        for i in range(bs):
            start = i * n_vars
            end = (i + 1) * n_vars
            mask[start:end, start:end] = True

        # 6. [优化] 使用 -inf 进行掩码，确保 Top-K 绝对不会选中块内元素
        sim_matrix.masked_fill_(mask, -float('inf'))

        # 7. 选最像的 k 个 (Hard Negatives)
        negative_index = torch.topk(sim_matrix, k=k, dim=1).indices

        return negative_index

    def pretrain(self, batch_x):
        bs, seq_len, n_vars = batch_x.shape

        # 1. RevIN Normalization
        z = batch_x  # (b, s, n)  n(=n'*f)*n(=n'*f)
        z = self.revin_layer_encoder(z, 'norm')
        # augmentation
        x_normed = z.permute(0, 2, 1)

        # 2. 负样本采样 (Baseline: Random)
        # 这里没有 FFT，没有 Knowledge Base，只有纯随机
        batch_size_n = bs * n_vars
        if self.configs.use_fft_sim:
            # Experiment 3 Logic
            negative_index = self.get_fft_negatives(x_normed, self.configs.negative_nums,self.configs.fft_var)
        else:
            # Baseline Logic
            negative_index = self.get_random_negatives(
                batch_size_n, self.configs.negative_nums, x_normed.device
            )
        x_normed_flat = x_normed.reshape(-1, seq_len)

        # -----------------------------------------------------------
        # Step A: Teacher (KB) 生成频域目标
        # -----------------------------------------------------------
        loss_distill = torch.tensor(0.0, device=batch_x.device)
        
        if self.configs.use_kb:
            with torch.no_grad():
                # 获取 KB 认为的"理想频域特征" target_freq
                # target_freq 是复数张量 (bs*n, seq//2+1)
                _, target_freq = self.KnowledgeGuide_encoder(x_normed_flat)
                target_freq = target_freq.detach()
            # x_normed_flat, _ = self.KnowledgeGuide_encoder(x_normed_flat)

        # -----------------------------------------------------------
        # Step B: Student (Encoder)
        # -----------------------------------------------------------
        # 采样 & 增强 (保持不变)

        x_positives = augment_positive_test(
            x_normed_flat,
            self.configs.mask_rate,
            self.configs.lm,
            k=self.configs.positive_nums
        )
        x_positives = x_positives.reshape(-1, seq_len)

        x_all = torch.cat([x_normed_flat, x_positives], dim=0)

        enc_out = self.enc_embedding(x_all.unsqueeze(-1))
        enc_out, _ = self.encoder(enc_out)

        s_enc_out = self.cl_projection(enc_out)
        s_enc_out = F.normalize(s_enc_out, dim=1)

        s_q = s_enc_out[:batch_size_n]
        s_k = s_enc_out[batch_size_n:].reshape(batch_size_n, self.configs.positive_nums, -1)

        # 提取随机选中的负样本特征
        neg_candidates = s_k[:, 0, :]
        selected_negatives = neg_candidates[negative_index]  # (bs*n, neg_nums, dim)

        # 计算相似度 Logits
        if self.configs.positive_nums == 1:
            pos_sim = torch.matmul(s_q.unsqueeze(1), s_k.permute(0, 2, 1)).squeeze(-1)
        else:
            pos_sim = torch.matmul(s_q.unsqueeze(1), s_k.permute(0, 2, 1)).squeeze()

        if self.configs.negative_nums == 1:
            neg_sim = torch.matmul(s_q.unsqueeze(1), selected_negatives.permute(0, 2, 1)).squeeze(-1)
        else:
            neg_sim = torch.matmul(s_q.unsqueeze(1), selected_negatives.permute(0, 2, 1)).squeeze()

        similarity_matrix = torch.cat([pos_sim, neg_sim], dim=-1)
        similarity_matrix = similarity_matrix / self.configs.temperature
        similarity_normed = self.log_softmax(similarity_matrix)

        if self.labels_cl is None:
            self.labels_cl = generate_CLLabels(x_normed_flat, self.configs.positive_nums, self.configs.negative_nums)
        loss_cl = self.kl(similarity_normed, self.labels_cl)

        # 6. Reconstruction Loss (Manifold Learning)
        # 使用随机负样本辅助重建
        positive_enc_out = enc_out[batch_size_n:].reshape(batch_size_n, self.configs.positive_nums, -1)
        negative_enc_out = positive_enc_out[:, 0, :][negative_index]

        # 计算重建权重
        rebuild_weight_matrix = self.softmax(similarity_matrix)
        pos_att = rebuild_weight_matrix[:, :self.configs.positive_nums].unsqueeze(1)
        neg_att = rebuild_weight_matrix[:, self.configs.positive_nums:].unsqueeze(1)

        rebuild_embed = torch.matmul(pos_att, positive_enc_out) + torch.matmul(neg_att, negative_enc_out)
        rebuild_embed = rebuild_embed.reshape(bs, n_vars, seq_len, -1)

        # 解码并计算 MSE
        pred_x = self.head_pretrain(rebuild_embed)
        pred_x = pred_x.reshape(bs, n_vars, -1).permute(0, 2, 1)
        pred_x = self.revin_layer_encoder(pred_x, 'denorm')

        loss_rb = self.mse(batch_x, pred_x)

        # -----------------------------------------------------------
        # Step D: [基于 FFT 变换的频域蒸馏]
        # -----------------------------------------------------------
        if self.configs.use_kb:
            # 1. 取出 Encoder 对原始样本的表征
            enc_out_origin = enc_out[:batch_size_n]  # (bs*n, seq, d)
            # 2. [桥接] 映射回时域 (使用预训练头)
            # 这步很关键：我们先还原成时域信号，这是物理上有意义的
            pred_time_signal = self.head_pretrain(enc_out_origin)  # (bs*n, seq)
            # 3. [变换] 显式 FFT 变换
            # 强迫模型生成的时域信号，必须具备正确的频域特性
            pred_freq_signal = torch.fft.rfft(pred_time_signal, dim=-1, norm='ortho')

            # 4. 计算复数距离 Loss
            # 我们可以计算幅度(Amplitude)损失和相位(Phase)损失，或者直接算复数距离

            # 方式一：直接复数 MSE (等价于 实部MSE + 虚部MSE)
            # loss_distill = torch.mean(torch.abs(pred_freq_signal - target_freq) ** 2)

            # 方式二 (更精细)：分别约束幅度和相位 (通常对时序任务更鲁棒)
            mag_loss = self.mse(torch.abs(pred_freq_signal), torch.abs(target_freq))
            loss_distill = mag_loss

        # 动态加权 Loss
        loss = self.awl(loss_cl, loss_rb) + self.configs.distill_weight * loss_distill

        return loss, loss_cl, loss_rb, loss_distill, None, None

    def forecast(self, x):
        bs, seq_len, n_vars = x.shape

        # 1. Norm
        x = self.revin_layer_encoder(x, 'norm')
        x = x.permute(0, 2, 1)
        if self.configs.ft_use_kb:
            x, _ = self.KnowledgeGuide_encoder(x)  # B, N, D
            x = x.reshape(-1, seq_len, 1)
        else:
            x = x.reshape(-1, seq_len, 1)

        # 2. Encoder
        x = self.enc_embedding(x)
        enc_out, _ = self.encoder(x)

        # 3. Forecast Head
        x = self.head_forecast(enc_out)
        x = x.reshape(bs, n_vars, -1).permute(0, 2, 1)

        # 4. Denorm
        x = self.revin_layer_encoder(x, 'denorm')
        return x

    def forward(self, batch_x):
        if self.task_name == 'pretrain':
            return self.pretrain(batch_x)
        elif self.task_name == 'finetune':
            return self.forecast(batch_x)
        return None