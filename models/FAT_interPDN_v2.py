import torch
import torch
import torch.nn as nn
from utils.augmentations import augment_positive_test
from utils.tools import generate_CLLabels, FFT_sim
from utils.losses import AutomaticWeightedLoss
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding
import torch.nn.functional as F


# ============================================================
# 组件 A: 频域感知的双视角概率正则化
# ============================================================
class DualViewProbRegularizer(nn.Module):
    """
    在 Encoder 输出的表征空间上施加概率分布正则化。
    时域视角和频域视角分别产生概率分布，通过对称 KL 散度约束其一致性，
    使 Encoder 学到结构化的、多视角一致的表征。
    """
    def __init__(self, d_model, n_bins=25):
        super().__init__()
        self.n_bins = n_bins
        # 时域视角投影
        self.proj_time = nn.Linear(d_model, n_bins)
        # 频域视角投影
        self.proj_freq = nn.Linear(d_model, n_bins)

    def forward(self, enc_out):
        """
        enc_out: (bs*n_vars, seq_len, d_model)
        return: loss_prob (标量)
        """
        # 时域视角: 直接在每个时间步上投影到 n_bins 维概率分布
        logits_time = self.proj_time(enc_out)               # (bs*n, seq_len, n_bins)
        p_time = F.softmax(logits_time, dim=-1) + 1e-8      # 避免 log(0)

        # 频域视角: 对 enc_out 做 FFT，取幅度谱作为特征
        enc_freq = torch.fft.rfft(enc_out, dim=1, norm='ortho')
        enc_freq_mag = enc_freq.abs()                        # (bs*n, freq_bins, d_model)
        # 由于 freq_bins 和 seq_len 维度不同，我们在 d_model 维度上投影
        # 然后对频率维度做 adaptive pooling 使其对齐到 seq_len
        logits_freq_raw = self.proj_freq(enc_freq_mag)       # (bs*n, freq_bins, n_bins)
        # Adaptive pooling 对齐时间维度
        logits_freq = F.adaptive_avg_pool1d(
            logits_freq_raw.permute(0, 2, 1),                # (bs*n, n_bins, freq_bins)
            enc_out.shape[1]                                  # 对齐到 seq_len
        ).permute(0, 2, 1)                                    # (bs*n, seq_len, n_bins)
        p_freq = F.softmax(logits_freq, dim=-1) + 1e-8

        # 对称 KL 散度
        kl_tf = F.kl_div(p_time.log(), p_freq, reduction='batchmean')
        kl_ft = F.kl_div(p_freq.log(), p_time, reduction='batchmean')
        loss_prob = (kl_tf + kl_ft) / 2.0

        return loss_prob


# ============================================================
# 组件 B: 频域重建头 (与原有时域 Flatten_Head 形成双视角)
# ============================================================
class FreqReconHead(nn.Module):
    """
    通过频域系数重建时域信号：
    将 Encoder 输出映射到频率系数 (实部+虚部)，再 IRFFT 回时域。
    与原有的 Flatten_Head(时域直接回归) 构成互补的双视角重建。
    """
    def __init__(self, seq_len, d_model, target_len, head_dropout=0):
        super().__init__()
        self.target_len = target_len
        self.freq_dim = target_len // 2 + 1  # RFFT 输出的频率分量数

        self.flatten = nn.Flatten(start_dim=-2)
        self.fc_real = nn.Linear(seq_len * d_model, self.freq_dim)
        self.fc_imag = nn.Linear(seq_len * d_model, self.freq_dim)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        x: (bs, n_vars, seq_len, d_model) 或 (bs*n_vars, seq_len, d_model)
        return: 时域重建信号
        """
        x = self.flatten(x)           # (..., seq_len * d_model)
        real = self.fc_real(x)         # (..., freq_dim)
        imag = self.fc_imag(x)         # (..., freq_dim)
        freq_coeff = torch.complex(real, imag)
        out = torch.fft.irfft(freq_coeff, n=self.target_len, dim=-1, norm='ortho')
        out = self.dropout(out)
        return out


# ============================================================
# 原始 FAT 组件 (保持不变)
# ============================================================
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
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


class FreNormLayer_KB(nn.Module):
    def __init__(self, n_knlg, input_len, bias=True):
        super(FreNormLayer_KB, self).__init__()
        self.embed_dim = input_len // 2 + 1
        self.n_knlg = n_knlg
        self.kb = nn.Parameter(torch.randn(n_knlg, self.embed_dim, dtype=torch.cfloat))
        self.scaling = self.embed_dim ** -0.5
        self.in_proj_q = ComplexLinear(self.embed_dim, self.embed_dim, bias=bias)
        self.in_proj_k = ComplexLinear(self.embed_dim, self.embed_dim, bias=bias)
        self.in_proj_v = ComplexLinear(self.embed_dim, self.embed_dim, bias=bias)
        self.out_proj = ComplexLinear(self.embed_dim, self.embed_dim, bias=bias)
        self.out_dim = input_len

    def retrive_w(self, x):
        q = self.in_proj_q(x)
        k = self.in_proj_k(self.kb)
        v = self.in_proj_v(self.kb)
        attn_weights = torch.matmul(q, torch.conj_physical(k).T) * self.scaling
        real = torch.real(attn_weights)
        attn_weights = F.softmax(real, dim=-1).type(torch.complex64)
        w = torch.matmul(attn_weights, v)
        return w

    def forward(self, x):
        x = torch.fft.rfft(x, dim=-1, norm='ortho')
        w = self.retrive_w(x)
        y = x * w
        out = torch.fft.irfft(y, n=self.out_dim, dim=-1, norm="ortho")
        return out, w


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


# ============================================================
# 主模型: FAT_interPDN_v2
# ============================================================
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

        self.KnowledgeGuide_encoder = FreNormLayer_KB(configs.n_knlg, configs.seq_len)
        self.enc_embedding = DataEmbedding(1, configs.d_model, configs.embed, configs.freq, configs.dropout)

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
        self.head_Inference_Re = nn.Linear(128, configs.seq_len  // 2 + 1, bias=True)
        self.head_Inference_Img = nn.Linear(128, configs.seq_len // 2 + 1, bias=True)
        # cl weight
        self.cl_projection = Pooler_Head(configs.seq_len, configs.d_model, head_dropout=configs.head_dropout)
        # reconstrution weight - 时域重建 (Branch 1, 原始 Flatten_Head)
        self.head_pretrain = Flatten_Head(configs.seq_len, configs.d_model, configs.seq_len, head_dropout=configs.head_dropout)
        # reconstrution weight - 频域重建 (Branch 2, 组件B)
        self.head_pretrain_freq = FreqReconHead(configs.seq_len, configs.d_model, configs.seq_len, head_dropout=configs.head_dropout)
        # finetune weight
        self.head_forecast = Flatten_Head(configs.seq_len, configs.d_model, configs.pred_len, head_dropout=configs.head_dropout)

        self.labels_cl = None
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')
        self.awl = AutomaticWeightedLoss(2)
        self.mse = torch.nn.MSELoss()

        # ============ 新增组件 ============

        # 组件 A: 双视角概率正则化器
        self.prob_regularizer = DualViewProbRegularizer(configs.d_model, n_bins=25)

        # 组件 C: 跨尺度一致性 - 使用独立投影层将粗尺度表征映射回同一空间
        self.scale_factor = 4
        self.coarse_proj = nn.Linear(configs.d_model, configs.d_model)

        # ============ 损失权重 (可通过 configs 外部控制) ============
        self.lambda_prob = getattr(configs, 'lambda_prob', 0.05)
        self.lambda_scale = getattr(configs, 'lambda_scale', 0.05)
        self.alpha_con = getattr(configs, 'alpha_con', 0.1)

        # ============ 消融控制开关 ============
        # 通过 configs 控制各组件开关, 默认全开 (argparse 传入 int 0/1)
        self.use_comp_a = bool(getattr(configs, 'use_comp_a', 1))
        self.use_comp_b = bool(getattr(configs, 'use_comp_b', 1))
        self.use_comp_c = bool(getattr(configs, 'use_comp_c', 1))

    def pretrain(self, batch_x):
        bs, seq_len, n_vars = batch_x.shape
        z = batch_x  # (b, s, n)
        z = self.revin_layer_encoder(z, 'norm')
        # augmentation
        x_normed = z
        x_normed = x_normed.permute(0, 2, 1)  
        sim_matrix = FFT_sim(x_normed)  # (b*n, b*n)
        x_normed = x_normed.reshape(-1, seq_len)  # (b*n, s)
        negative_index = torch.topk(sim_matrix, k=self.configs.negative_nums, dim=1).indices
        # Knowledge_guide
        x_reformed, _ = self.KnowledgeGuide_encoder(x_normed)  # 过滤噪声
        x_positives = augment_positive_test(x_reformed, self.configs.mask_rate, self.configs.lm, k=self.configs.positive_nums)
        x_positives = x_positives.reshape(-1, seq_len)
        x_all = torch.cat([x_normed, x_positives], dim=0)

        # Encoder
        enc_out = self.enc_embedding(x_all.unsqueeze(-1))
        enc_out, _ = self.encoder(enc_out)

        # ============================================================
        # 组件 A: 在原始样本的 Encoder 输出上施加概率分布正则化
        # ============================================================
        if self.use_comp_a:
            enc_out_orig = enc_out[:bs * n_vars]  # 只取原始样本的表征
            loss_prob = self.prob_regularizer(enc_out_orig)
        else:
            loss_prob = torch.tensor(0.0, device=enc_out.device)

        # ============================================================
        # 组件 C: 跨尺度表征一致性
        # ============================================================
        if self.use_comp_c:
            enc_out_orig = enc_out[:bs * n_vars]  # (bs*n, seq_len, d_model)
            # Fine-scale: 对时间维度做 AvgPool 下采样
            fine_for_pool = enc_out_orig.permute(0, 2, 1)  # (bs*n, d_model, seq_len)
            fine_down = F.avg_pool1d(fine_for_pool, kernel_size=self.scale_factor,
                                     stride=self.scale_factor)  # (bs*n, d_model, seq_len//4)
            fine_down = fine_down.permute(0, 2, 1)  # (bs*n, seq_len//4, d_model)
            # Coarse-scale: 独立投影
            coarse = self.coarse_proj(fine_down.detach())  # detach 防止退化
            loss_scale = self.mse(fine_down, coarse)
        else:
            loss_scale = torch.tensor(0.0, device=enc_out.device)

        # ============================================================
        # Contrastive Learning (与原始 FAT 完全一致)
        # ============================================================
        s_enc_out = self.cl_projection(enc_out)
        s_enc_out = F.normalize(s_enc_out, dim=1)
        s_q = s_enc_out[: bs * n_vars]
        s_k = s_enc_out[bs * n_vars:].reshape(bs * n_vars, self.configs.positive_nums, -1)
        if self.labels_cl is None:
            self.labels_cl = generate_CLLabels(x_normed, self.configs.positive_nums, self.configs.negative_nums)
        if self.configs.positive_nums == 1:
            positive_similarity_matrix = torch.matmul(s_q.unsqueeze(1), s_k.permute(0, 2, 1)).squeeze(-1)
        else:
            positive_similarity_matrix = torch.matmul(s_q.unsqueeze(1), s_k.permute(0, 2, 1)).squeeze()
        if self.configs.negative_nums == 1:
            negative_similarity_matrix = torch.matmul(s_q.unsqueeze(1),
                                                      s_k[:, 0, :][negative_index].permute(0, 2, 1)).squeeze(-1)
        else:
            negative_similarity_matrix = torch.matmul(s_q.unsqueeze(1),
                                                      s_k[:, 0, :][negative_index].permute(0, 2, 1)).squeeze()
        similarity_matrix = torch.cat([positive_similarity_matrix, negative_similarity_matrix], dim=-1)
        similarity_matrix = similarity_matrix / self.configs.temperature
        similarity_normed = self.log_softmax(similarity_matrix)
        loss_cl = self.kl(similarity_normed, self.labels_cl)

        # ============================================================
        # Reconstruction: 相似度加权聚合 (与原始 FAT 一致)
        # ============================================================
        positive_enc_out = enc_out[bs * n_vars:].reshape(bs * n_vars, self.configs.positive_nums, -1)
        negative_enc_out = positive_enc_out[:, 0, :][negative_index]
        rebuild_weight_matrix = self.softmax(similarity_matrix)
        pos_att = rebuild_weight_matrix[:, :self.configs.positive_nums].unsqueeze(1)
        neg_att = rebuild_weight_matrix[:, self.configs.positive_nums:].unsqueeze(1)
        rebuild_embed = torch.matmul(pos_att, positive_enc_out) + torch.matmul(neg_att, negative_enc_out)
        rebuild_embed = rebuild_embed.reshape(bs, n_vars, seq_len, -1)

        # Branch 1: 时域重建 (原始 Flatten_Head)
        pred_x_time = self.head_pretrain(rebuild_embed)
        pred_x_time = pred_x_time.reshape(bs, n_vars, -1).permute(0, 2, 1)
        pred_x_time = self.revin_layer_encoder(pred_x_time, 'denorm')
        loss_rb_time = self.mse(batch_x, pred_x_time)

        # ============================================================
        # 组件 B: 频域重建 + 双视角一致性
        # ============================================================
        if self.use_comp_b:
            pred_x_freq = self.head_pretrain_freq(rebuild_embed)
            pred_x_freq = pred_x_freq.reshape(bs, n_vars, -1).permute(0, 2, 1)
            pred_x_freq_denorm = self.revin_layer_encoder(pred_x_freq, 'denorm')
            loss_rb_freq = self.mse(batch_x, pred_x_freq_denorm)
            # 一致性约束: 时域重建和频域重建结果应一致 (detach 其中一个防止退化)
            loss_con = self.mse(pred_x_time, pred_x_freq_denorm.detach())
            loss_rb = loss_rb_time + loss_rb_freq + self.alpha_con * loss_con
        else:
            loss_rb = loss_rb_time

        # ============================================================
        # 总损失
        # ============================================================
        loss_base = self.awl(loss_cl, loss_rb)
        loss = loss_base + self.lambda_prob * loss_prob + self.lambda_scale * loss_scale

        return loss, loss_cl, loss_rb, None, None, None

    def forecast(self, x):
        """微调预测路径，与原始 FAT 完全一致，不受新组件影响"""
        bs, seq_len, n_vars = x.shape
        z = x
        z = self.revin_layer_encoder(z, 'norm')
        x = z
        x = x.permute(0, 2, 1)
        if self.configs.forcastMode == "freq":
            x, _ = self.KnowledgeGuide_encoder(x)
            x = x.reshape(-1, seq_len, 1)
        else:
            x = x.reshape(-1, seq_len, 1)
        x = self.enc_embedding(x)
        x, _ = self.encoder(x)

        x = self.head_forecast(x)
        x = x.reshape(bs, n_vars, -1).permute(0, 2, 1)
        z = x
        z = self.revin_layer_encoder(z, 'denorm')
        x = z
        return x

    def forward(self, batch_x):

        if self.task_name == 'pretrain':
            if self.configs.pretrain_mode == "1":
                return self.pretrain(batch_x)
            elif self.configs.pretrain_mode == "0":
                return self.pretrain(batch_x)
            else:
                return self.pretrain(batch_x)

        if self.task_name == 'finetune':
            dec_out = self.forecast(batch_x)
            return dec_out
        return None
