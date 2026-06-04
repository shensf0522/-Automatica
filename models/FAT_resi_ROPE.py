"""
FAT_resi_ROPE.py
基于 FAT_resi.py，整合 RotaryEmbedding (RoPE) 来增强相位信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.augmentations import augment_positive_test
from utils.tools import ContrastiveWeight, AggregationRebuild, generate_CLLabels, FFT_sim
from utils.losses import AutomaticWeightedLoss
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos

import math


# ============== RotaryEmbedding (从 GFMixer 移植) ==============

class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    通过旋转编码将位置信息编码为相位信息
    """

    def __init__(
        self,
        config,
        dim: int = None,
        n_channels: int = None,
        prefix: str = "attn",
        freq_distribution: str = None,
        freq_learnable: bool = None,
        use_rope_cache: bool = True,
        include_neg_freq: bool = None,
        floor_freq_ratio: float = None,
        clamp_floor_freq: bool = None,
        clamp_floor_to_zero: bool = None,
        upper_freq_ratio: float = None,
        clamp_upper_freq: float = None,
        clamp_upper_to_zero: bool = None,
        zero_freq_ratio: float = None,
        clamp_to_linear: bool = None,
        clamp_to_linear_mode: str = None,
        init_upper_freq: float = None,
        init_floor_freq: float = None,
    ):
        super().__init__()
        self.config = config

        self.prefix = prefix
        self.suffix = "rope"
        self.use_rope_cache = use_rope_cache

        self.freq_distribution = freq_distribution if freq_distribution is not None else getattr(config, 'rope_init_distribution', 'exponential')
        if self.freq_distribution not in ["constant", "linear", "uniform", "gaussian", "exponential"]:
            self.freq_distribution = "exponential"

        self.freq_learnable = freq_learnable if freq_learnable is not None else getattr(config, 'rope_learnable', False)
        self.include_neg_freq = include_neg_freq if include_neg_freq is not None else getattr(config, 'rope_include_neg_freq', False)

        if dim is not None:
            self.dim = dim
        elif hasattr(config, 'd_model'):
            self.dim = config.d_model // config.n_heads
        else:
            self.dim = 64

        if n_channels is not None:
            self.n_channels = n_channels
        elif prefix == "attn" and self.freq_learnable and getattr(config, 'rope_no_repetition', False):
            self.n_channels = config.n_heads
        else:
            self.n_channels = 1

        self.init_floor_freq = init_floor_freq if init_floor_freq is not None else getattr(config, 'rope_init_floor_freq', 0.0)
        self.init_upper_freq = init_upper_freq if init_upper_freq is not None else getattr(config, 'rope_init_upper_freq', 1.0)

        self.clamp_floor_freq = clamp_floor_freq if clamp_floor_freq is not None else getattr(config, 'rope_clamp_floor_freq', True)
        if self.clamp_floor_freq:
            self.floor_freq_ratio = floor_freq_ratio if floor_freq_ratio is not None else getattr(config, 'rope_floor_freq_ratio', 0.1)
            max_seq_len = getattr(config, 'max_sequence_length', 1024)
            self.floor_freq = 2 * math.pi / max_seq_len * self.floor_freq_ratio

            self.clamp_floor_to_zero = clamp_floor_to_zero if clamp_floor_to_zero is not None else getattr(config, 'rope_clamp_floor_to_zero', True)
            self.clamp_floor_value = 0.0 if self.clamp_floor_to_zero else 2 * math.pi / max_seq_len * self.floor_freq_ratio
        else:
            self.floor_freq = 0.0

        self.clamp_upper_freq = clamp_upper_freq if clamp_upper_freq is not None else getattr(config, 'rope_clamp_upper_freq', False)
        if self.clamp_upper_freq:
            self.upper_freq_ratio = upper_freq_ratio if upper_freq_ratio is not None else getattr(config, 'rope_upper_freq_ratio', 0.8)
            self.upper_freq = math.pi * self.upper_freq_ratio

            self.clamp_upper_to_zero = clamp_upper_to_zero if clamp_upper_to_zero is not None else getattr(config, 'rope_clamp_upper_to_zero', False)
            self.clamp_upper_value = 0.0 if self.clamp_upper_to_zero else math.pi * self.upper_freq_ratio
        else:
            self.upper_freq = 1.0

        if self.clamp_floor_freq or self.clamp_upper_freq:
            assert self.upper_freq >= self.floor_freq

        self.zero_freq_ratio = zero_freq_ratio if zero_freq_ratio is not None else getattr(config, 'rope_zero_freq_ratio', 0.0)

        self.clamp_to_linear = clamp_to_linear if clamp_to_linear is not None else getattr(config, 'rope_clamp_to_linear', False)
        self.clamp_to_linear_mode = clamp_to_linear_mode if clamp_to_linear_mode is not None else getattr(config, 'rope_clamp_to_linear_mode', 'floor')

        max_seq_len = getattr(config, 'max_sequence_length', 1024)
        if self.freq_learnable:
            self.inv_freq = nn.Parameter(
                self.get_inv_freq(self.dim, config),
                requires_grad=True
            )
        else:
            self.inv_freq = self.get_inv_freq(self.dim, config)

            if self.use_rope_cache:
                self.get_rotary_embedding(max_seq_len, config)

    def get_inv_freq(self, dim: int, config) -> torch.Tensor:
        # 使用 CPU 作为默认设备，与模型参数分离
        device = torch.device("cuda")
        max_seq_len = getattr(config, 'max_sequence_length', 1024)

        if self.freq_distribution == "constant":
            inv_freq = torch.ones(dim // 2, device=device, dtype=torch.float)
        elif self.freq_distribution == "linear":
            inv_freq = 2 * math.pi / max_seq_len * torch.arange(0, dim, 2, device=device, dtype=torch.float)
            inv_freq = inv_freq.flip(0)
        elif self.freq_distribution == "uniform":
            inv_freq = 1.0 * torch.rand(dim // 2, device=device, dtype=torch.float)
        elif self.freq_distribution == "gaussian":
            inv_freq = torch.randn(dim // 2, device=device, dtype=torch.float).abs()
            inv_freq = inv_freq / inv_freq.max()
        else:  # exponential (default)
            theta = getattr(self.config, 'rope_theta', 10000)
            inv_freq = 1.0 / (
                theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
            )

        inv_freq = self.init_floor_freq + inv_freq * (self.init_upper_freq - self.init_floor_freq)

        if self.clamp_floor_freq:
            inv_freq[inv_freq < self.floor_freq] = self.clamp_floor_value
        if self.clamp_upper_freq:
            inv_freq[inv_freq > self.upper_freq] = self.clamp_upper_value

        if self.include_neg_freq:
            inv_freq *= (-1) ** torch.arange(0, dim // 2, device=device, dtype=torch.float)

        if self.prefix == "embed":
            inv_freq = inv_freq
        elif self.prefix == "attn":
            if self.freq_learnable and getattr(self.config, 'rope_no_repetition', False):
                inv_freq = inv_freq.repeat(self.n_channels, 1)
            else:
                inv_freq = inv_freq[None, :]
        else:
            inv_freq = inv_freq[None, :]

        return inv_freq

    def get_rotary_embedding(self, seq_len: int, config, use_rope_cache: bool = None) -> tuple:
        use_rope_cache = use_rope_cache or ((not self.freq_learnable) and self.use_rope_cache)

        # 只有当 inv_freq 是可学习参数时才进行 clamp
        if self.freq_learnable and hasattr(self.inv_freq, 'data'):
            if self.clamp_floor_freq or self.clamp_upper_freq:
                sign = self.inv_freq.data.sign()
                max_seq_len = getattr(config, 'max_sequence_length', 1024)
                self.inv_freq.data.abs_().clamp_(
                    2 * math.pi / max_seq_len * self.floor_freq_ratio, math.pi * self.upper_freq_ratio
                ).mul_(sign)
            else:
                self.inv_freq.data.clamp_(-math.pi, math.pi)

        device = self.inv_freq.device
        seq = torch.arange(seq_len, device=device, dtype=torch.float)

        if self.prefix == "embed":
            freqs = torch.einsum("t, d -> td", seq, self.inv_freq)
        elif self.prefix == "attn":
            freqs = torch.einsum("t, hd -> htd", seq, self.inv_freq)
        else:
            freqs = torch.einsum("t, d -> td", seq, self.inv_freq.squeeze(0))

        positions = torch.cat((freqs, freqs), dim=-1).unsqueeze(0)
        pos_sin, pos_cos = positions.sin(), positions.cos()

        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        if self.prefix == "embed":
            B, T, hs = x.size()
            x = x.view(B, T, 2, hs // 2)
        elif self.prefix == "attn":
            B, nh, T, hs = x.size()
            x = x.view(B, nh, T, 2, hs // 2)
        else:
            B, T, hs = x.size()
            x = x.view(B, T, 2, hs // 2)

        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        if not inverse:
            return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)
        else:
            return ((t * pos_cos) - (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(self, x: torch.Tensor, all_len: int = None, layer_idx: int = None, inverse: bool = False) -> torch.Tensor:
        """
        对输入应用 RoPE 旋转位置编码
        x: [B, H, T, D] 或 [B, T, D]
        """
        if all_len is None:
            all_len = x.shape[-2]

        x_ = x.float()
        device = x.device

        # 尝试从缓存获取，或重新计算
        try:
            if not self.freq_learnable and self.use_rope_cache and hasattr(self, '_cached_pos'):
                pos_sin, pos_cos = self._cached_pos
            else:
                pos_sin, pos_cos = self.get_rotary_embedding(all_len, self.config)
                if not self.freq_learnable and self.use_rope_cache:
                    self._cached_pos = (pos_sin, pos_cos)
        except:
            pos_sin, pos_cos = self.get_rotary_embedding(all_len, self.config)

        pos_sin = pos_sin.to(device)
        pos_cos = pos_cos.to(device)

        x_len = x_.shape[-2]

        if self.prefix == "attn":
            x_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, all_len - x_len:all_len, :],
                pos_cos[:, :, all_len - x_len:all_len, :],
                x_,
                inverse
            )
        else:
            x_ = self.apply_rotary_pos_emb(
                pos_sin[:, all_len - x_len:all_len, :],
                pos_cos[:, all_len - x_len:all_len, :],
                x_,
                inverse
            )

        return x_.type_as(x)


# ============== 辅助函数 ==============

def moving_average(x, kernel_size):
    """
    x: [B, S, N]  (已经是 RevIN 归一化后的序列)
    在时间维度 S 上做滑动平均
    """
    if kernel_size <= 1:
        return x

    B, S, N = x.shape
    x_bn = x.permute(0, 2, 1).reshape(-1, 1, S)
    pad = kernel_size - 1
    x_pad = F.pad(x_bn, (pad, 0), mode='replicate')
    trend = F.avg_pool1d(x_pad, kernel_size, stride=1)
    trend = trend.reshape(B, N, S).permute(0, 2, 1)
    return trend


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


class Flatten_Head(nn.Module):
    def __init__(self, seq_len, d_model, pred_len, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(seq_len * d_model, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
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

    def forward(self, x):
        x = self.pooler(x)
        return x


class FreNormLaryer_KB(nn.Module):
    def __init__(self, n_knlg, input_len, bias=True):
        super(FreNormLaryer_KB, self).__init__()
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
        out = torch.fft.irfft(y, n=self.out_dim, dim=-1, norm='ortho')
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

    def forward(self, x, mode: str, mask=None):
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
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
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

        self.embed_size = self.seq_len
        self.hidden_size = configs.hidden_size

        # 知识库
        self.KnowledgeGuide_encoder = FreNormLaryer_KB(configs.n_knlg, configs.seq_len)
        # 使用 DataEmbedding_wo_pos 避免与 RoPE 冲突
        self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # ============== 添加 RoPE 配置 ==============
        self.use_rope = getattr(configs, 'use_rope', True)
        self.max_seq_len = configs.seq_len

        if self.use_rope:
            # 创建 RoPE 配置对象
            rope_config = type('RopeConfig', (), {
                'd_model': configs.d_model,
                'n_heads': configs.n_heads,
                'max_sequence_length': configs.seq_len,
                'rope_theta': getattr(configs, 'rope_theta', 10000),
                'rope_init_distribution': getattr(configs, 'rope_init_distribution', 'exponential'),
                'rope_learnable': getattr(configs, 'rope_learnable', False),
                'rope_include_neg_freq': getattr(configs, 'rope_include_neg_freq', False),
                'rope_init_upper_freq': getattr(configs, 'rope_init_upper_freq', 1.0),
                'rope_init_floor_freq': getattr(configs, 'rope_init_floor_freq', 0.0),
                'rope_floor_freq_ratio': getattr(configs, 'rope_floor_freq_ratio', 0.1),
                'rope_clamp_floor_freq': getattr(configs, 'rope_clamp_floor_freq', True),
                'rope_clamp_floor_to_zero': getattr(configs, 'rope_clamp_floor_to_zero', True),
            })()

            # 初始化 RotaryEmbedding
            self.rope = RotaryEmbedding(
                rope_config,
                dim=configs.d_model,
                prefix="embed",
                use_rope_cache=True
            )

        # Transformer Encoder
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
        self.head_pretrain = Flatten_Head(configs.seq_len, configs.d_model, configs.seq_len, head_dropout=configs.head_dropout)
        # finetube weight

        self.use_residual_pretrain = getattr(configs, 'use_residual_pretrain', False)
        self.decomp_kernel = getattr(configs, 'decomp_kernel', 25)
        self.residual_mask_rate = getattr(configs, 'residual_mask_rate', 0.5)

        self.use_amd_struct_head = getattr(configs, 'use_amd_struct_head', False)
        raw_kernels = getattr(configs, 'decomp_kernels', None)
        if raw_kernels is None:
            self.decomp_kernels = [self.decomp_kernel]
        elif isinstance(raw_kernels, (list, tuple)):
            self.decomp_kernels = list(raw_kernels)
        else:
            self.decomp_kernels = [raw_kernels]

        self.hierarchical_trend = getattr(configs, 'hierarchical_trend', True)

        self.head_forecast = Flatten_Head(configs.seq_len, configs.d_model, configs.pred_len, head_dropout=configs.head_dropout)
        self.labels_cl = None
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')
        self.awl = AutomaticWeightedLoss(2)
        self.mse = torch.nn.MSELoss()

        self.trend_projection = nn.Linear(1, configs.d_model)

    def compute_trend(self, x):
        trend_total, trends = multi_scale_moving_average(
            x,
            kernel_list=self.decomp_kernels,
            hierarchical=self.hierarchical_trend
        )
        return trend_total, trends

    def _apply_rope(self, x):
        """
        对输入应用 RoPE
        x: [B, T, D]
        """
        if not self.use_rope:
            return x

        # 将 x 转换为 [B, T, D] 形状
        if x.dim() == 3:
            # x: [B * N, T, D] 或 [B, T, D]
            return self.rope(x, all_len=self.max_seq_len)
        return x

    def _apply_rope_to_attn_input(self, q, k, v):
        """
        在 attention 计算前对 Q, K, V 应用 RoPE (适用于 multi-head attention)
        q, k, v: [B, H, T, D] 或 [B, T, D]
        """
        if not self.use_rope:
            return q, k, v

        # 对每个 head 应用 RoPE
        if q.dim() == 4:  # [B, H, T, D]
            q = self.rope(q.permute(0, 1, 2, 3), all_len=self.max_seq_len).permute(0, 1, 2, 3)
            k = self.rope(k.permute(0, 1, 2, 3), all_len=self.max_seq_len).permute(0, 1, 2, 3)
        elif q.dim() == 3:  # [B, T, D]
            q = self.rope(q, all_len=self.max_seq_len)
            k = self.rope(k, all_len=self.max_seq_len)

        return q, k, v

    def pretrain_residual(self, batch_x, batch_x_mark=None):
        """
        残差自监督预训练
        """
        bs, seq_len, n_vars = batch_x.shape

        # 1) 显式结构（趋势） + 残差
        trend = moving_average(batch_x, self.decomp_kernel)
        res_ts = batch_x - trend

        # 2) RevIN 归一化
        z = self.revin_layer_encoder(res_ts, 'norm')

        r_normed = z
        r_normed = r_normed.permute(0, 2, 1)
        sim_matrix = FFT_sim(r_normed)
        r_normed = r_normed.reshape(-1, seq_len)
        negative_index = torch.topk(sim_matrix, k=self.configs.negative_nums, dim=1).indices

        # Knowledge guide
        r_reformed, _ = self.KnowledgeGuide_encoder(r_normed)
        r_positives = augment_positive_test(r_reformed, self.configs.mask_rate, self.configs.lm,
                                            k=self.configs.positive_nums)
        r_positives = r_positives.reshape(-1, seq_len)
        r_all = torch.cat([r_normed, r_positives], dim=0)

        # ==================== 新增处理 x_mark 的逻辑 ====================
        if batch_x_mark is not None:
            mark_dim = batch_x_mark.shape[-1]
            # 适配独立通道 (CI): [bs, seq_len, F] -> [bs, n_vars, seq_len, F] -> [bs * n_vars, seq_len, F]
            x_mark_enc = batch_x_mark.unsqueeze(1).repeat(1, n_vars, 1, 1).reshape(-1, seq_len, mark_dim)
            # 适配对比学习的样本翻倍 (原样本 + 正样本) -> [2 * bs * n_vars, seq_len, F]
            x_mark_all = torch.cat([x_mark_enc, x_mark_enc], dim=0)
        else:
            x_mark_all = None
        # ================================================================

        # Encoder
        enc_out = self.enc_embedding(r_all.unsqueeze(-1),x_mark_all)

        # ============== 应用 RoPE ==============
        # 在 embedding 后、encoder 前应用 RoPE
        if self.use_rope:
            enc_out = self._apply_rope(enc_out)

        enc_out, _ = self.encoder(enc_out)

        # Contrastive Learning
        s_enc_out = self.cl_projection(enc_out)
        s_enc_out = F.normalize(s_enc_out, dim=1)
        s_q = s_enc_out[: bs * n_vars]
        s_k = s_enc_out[bs * n_vars:].reshape(bs * n_vars, self.configs.positive_nums, -1)
        if self.labels_cl is None:
            self.labels_cl = generate_CLLabels(r_normed, self.configs.positive_nums, self.configs.negative_nums)
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

        # rebuild origin
        positive_enc_out = enc_out[bs * n_vars:].reshape(bs * n_vars, self.configs.positive_nums, -1)
        negative_enc_out = positive_enc_out[:, 0, :][negative_index]
        rebuild_weight_matrix = self.softmax(similarity_matrix)
        pos_att = rebuild_weight_matrix[:, :self.configs.positive_nums].unsqueeze(1)
        neg_att = rebuild_weight_matrix[:, self.configs.positive_nums:].unsqueeze(1)
        rebuild_embed = torch.matmul(pos_att, positive_enc_out) + torch.matmul(neg_att, negative_enc_out)
        rebuild_embed = rebuild_embed.reshape(bs, n_vars, seq_len, -1)

        # 补充原始的特征
        trend_feat_in = trend.permute(0, 2, 1).reshape(bs, n_vars, seq_len, 1)

        fused_embed = rebuild_embed + trend_feat_in
        pred_x = self.head_pretrain(fused_embed)
        pred_x = pred_x.reshape(bs, n_vars, -1).permute(0, 2, 1)
        pred_x = self.revin_layer_encoder(pred_x, 'denorm')
        loss_rb = self.mse(batch_x, pred_x)

        loss = self.awl(loss_cl, loss_rb)

        return loss, loss_cl, loss_rb, None, None, None

    def pretrainWithContrast(self, batch_x, batch_x_mark):
        bs, seq_len, n_vars = batch_x.shape
        z = batch_x
        z = self.revin_layer_encoder(z, 'norm')

        x_raw = z
        x_raw = x_raw.permute(0, 2, 1)
        sim_matrix = FFT_sim(x_raw)
        x_raw = x_raw.reshape(-1, seq_len)
        negative_index = torch.topk(sim_matrix, k=self.configs.negative_nums, dim=1).indices

        # fre norm
        x_reformed, _ = self.KnowledgeGuide_encoder(x_raw)

        # positive_cases
        x_positives = augment_positive_test(x_reformed, self.configs.mask_rate, self.configs.lm, k=self.configs.positive_nums)
        x_positives = x_positives.reshape(-1, seq_len)
        x_all = torch.cat([x_raw, x_positives], dim=0)

        # Encode all samples
        enc_out = self.enc_embedding(x_all.unsqueeze(-1))

        # ============== 应用 RoPE ==============
        if self.use_rope:
            enc_out = self._apply_rope(enc_out)

        enc_out, _ = self.encoder(enc_out)

        # CL loss
        s_enc_out = self.cl_projection(enc_out)
        s_enc_out_norm = F.normalize(s_enc_out, dim=1)
        s_q = s_enc_out_norm[: bs * n_vars]
        s_k = s_enc_out_norm[bs * n_vars:].reshape(bs * n_vars, self.configs.positive_nums, -1)
        if self.labels_cl is None:
            self.labels_cl = generate_CLLabels(x_raw, self.configs.positive_nums, self.configs.negative_nums)
        if self.configs.positive_nums == 1:
            positive_similarity_matrix = torch.matmul(s_q.unsqueeze(1), s_k.permute(0, 2, 1)).squeeze(-1)
        else:
            positive_similarity_matrix = torch.matmul(s_q.unsqueeze(1), s_k.permute(0, 2, 1)).squeeze()
        if self.configs.negative_nums == 1:
            negative_similarity_matrix = torch.matmul(s_q.unsqueeze(1), s_k[:, 0, :][negative_index].permute(0, 2, 1)).squeeze(-1)
        else:
            negative_similarity_matrix = torch.matmul(s_q.unsqueeze(1), s_k[:, 0, :][negative_index].permute(0, 2, 1)).squeeze()
        similarity_matrix = torch.cat([positive_similarity_matrix, negative_similarity_matrix], dim=-1)
        similarity_matrix = similarity_matrix / self.configs.temperature
        similarity_normed = self.log_softmax(similarity_matrix)

        loss_cl = self.kl(similarity_normed, self.labels_cl)

        # rebuild loss
        positive_enc_out = enc_out[bs * n_vars:].reshape(bs * n_vars, self.configs.positive_nums, -1)
        hardNegative_enc_out = positive_enc_out[:, 0, :][negative_index]
        rebuild_weight_matrix = self.softmax(similarity_matrix)
        pos_att = rebuild_weight_matrix[:, :self.configs.positive_nums].unsqueeze(1)
        neg_att = rebuild_weight_matrix[:, self.configs.positive_nums:].unsqueeze(1)
        rebuild_embed = torch.matmul(pos_att, positive_enc_out) + torch.matmul(neg_att, hardNegative_enc_out)
        rebuild_embed = rebuild_embed.reshape(bs, n_vars, seq_len, -1)
        pred_x = self.head_pretrain(rebuild_embed)
        pred_x = pred_x.reshape(bs, n_vars, -1).permute(0, 2, 1)
        pred_x = self.revin_layer_encoder(pred_x, 'denorm')
        loss_rb = self.mse(batch_x, pred_x)
        loss = self.awl(loss_cl, loss_rb)

        return loss, loss_cl, loss_rb, None, None, None

    def forecast(self, x):
        bs, seq_len, n_vars = x.shape
        z = self.revin_layer_encoder(x, 'norm')
        x = z
        x = x.permute(0, 2, 1)
        if self.configs.forcastMode == "freq":
            x, _ = self.KnowledgeGuide_encoder(x)
            x = x.reshape(-1, seq_len, 1)
        else:
            x = x.reshape(-1, seq_len, 1)

        x = self.enc_embedding(x)

        # ============== 应用 RoPE ==============
        if self.use_rope:
            x = self._apply_rope(x)

        enc_out, _ = self.encoder(x)
        x = self.head_forecast(enc_out)
        x = x.reshape(bs, n_vars, -1).permute(0, 2, 1)
        x = self.revin_layer_encoder(x, 'denorm')
        return x

    def forward(self, batch_x, batch_x_mark=None):
        if self.task_name == 'pretrain':
            if self.configs.pretrain_mode == "1":
                return self.pretrainWithContrast(batch_x, batch_x_mark)
            elif self.configs.pretrain_mode == "0":
                return self.pretrain_residual(batch_x, batch_x_mark)
            elif self.configs.pretrain_mode == "residual":
                return self.pretrain_residual(batch_x, batch_x_mark)
            else:
                raise ValueError(f"Unsupported pretrain_mode: {self.configs.pretrain_mode}")

        if self.task_name == 'finetune':
            if self.task_type == 'c':
                dec_out = self.clf(batch_x)
                return dec_out
            elif self.task_type == 'reg':
                dec_out = self.forecast(batch_x,batch_x_mark)
                return dec_out
            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")


# 多尺度 moving average 辅助函数
def multi_scale_moving_average(x, kernel_list, hierarchical=True):
    """
    多尺度趋势提取
    x: [B, S, N]
    kernel_list: list of kernel sizes
    """
    trends = []
    for kernel_size in kernel_list:
        trend = moving_average(x, kernel_size)
        trends.append(trend)

    if hierarchical:
        # 层级融合：从大尺度开始，逐渐融合小尺度
        trend_total = trends[0]
        for i in range(1, len(trends)):
            trend_total = trend_total + trends[i]
        trend_total = trend_total / len(trends)
    else:
        trend_total = sum(trends) / len(trends)

    return trend_total, trends
