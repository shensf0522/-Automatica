import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from utils.losses import AutomaticWeightedLoss
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding

from utils.masking import get_mask



def augment_freqdrop(x, drop_rate=0.5, band_bins=1, eta=0.0):
    """
    窄带频域抑制增强（notch）
    x:  任意形状，最后一维是时间 S，比如 [..., S]
    drop_rate: 每条样本做抑制的概率
    band_bins: 半带宽（1 表示中心频点左右各 1 个 bin）
    eta: 衰减系数（0.0=完全静音；0.1/0.2=温和衰减）
    返回：形状与 x 一致
    """
    # 统一摊平到二维
    orig_shape = x.shape        # [..., S]
    S = orig_shape[-1]
    x_flat = x.reshape(-1, S)   # [M, S]

    # rFFT
    X = torch.fft.rfft(x_flat, dim=1)   # [M, F], complex
    M, F = X.shape

    if drop_rate > 0.0 and F > 2:       # 至少要有可选的中心频点
        # 逐样本随机决定是否抑制及中心频点
        # 避开 DC(0) 与 Nyquist(F-1)
        device = X.device
        for i in range(M):
            if torch.rand(1, device=device).item() < drop_rate:
                f0 = torch.randint(low=1, high=F-1, size=(1,), device=device).item()
                l = max(1, f0 - band_bins)
                r = min(F-1, f0 + band_bins + 1)
                X[i, l:r] *= eta

    # iFFT 并还原形状
    x_tilde = torch.fft.irfft(X, n=S, dim=1)   # [M, S], float
    return x_tilde.reshape(orig_shape)
# ---------------------------
# 1) Multi-Scale Temporal Extractor (MSTE)
#    用深度可分离 + 空洞卷积金字塔捕捉趋势/慢变化（不使用 avg pooling）
# ---------------------------
class DepthwiseConv1d(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.dw = nn.Conv1d(
            in_channels=channels, out_channels=channels,
            kernel_size=kernel_size, padding=padding,
            dilation=dilation, groups=channels, bias=False
        )
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()

    def forward(self, x_bcn):  # [B, C, S]
        y = self.dw(x_bcn)
        y = self.pw(y)
        # LayerNorm over channel dim -> 先转 [B, S, C]
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        return self.act(y)

class StructAdapter(nn.Module):
    def __init__(self, C, d_model):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)  # 时域全局汇聚
        in_dim = 3*C                         # s_ms/s_freq/s_cae 三路的通道均值
        hid = 128
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, 2*C*d_model)      # 输出 gamma,beta
        )

    def forward(self, s_ms, s_freq, s_cae):
        # s_*: [B,C,S] -> [B, C]
        def p(x): return self.pool(x).squeeze(-1)
        feat = torch.cat([p(s_ms), p(s_freq), p(s_cae)], dim=-1)  # [B, 3C]
        out = self.mlp(feat).view(feat.size(0), 2, -1)            # [B, 2, C*d]
        gamma, beta = out[:,0], out[:,1]                          # [B, C*d]
        # 形状对齐成 [B,C,1,d]，再扩展到 [B,C,S,d]
        B, Cd = gamma.size()
        d = Cd // (s_ms.size(1))
        C = s_ms.size(1)
        gamma = gamma.view(B, C, 1, d)
        beta  = beta.view(B, C, 1, d)
        # 限幅，避免发散
        gamma = 0.1 * torch.tanh(gamma)
        beta  = 0.1 * torch.tanh(beta)
        # 在 _apply_film 里会 broadcast 到 [B,C,S,d]
        return gamma, beta



class MSTE(nn.Module):
    """
    Multi-Scale Temporal Extractor
    - 一组不同空洞率与卷积核的深度可分离卷积并联，再经 1x1 混合与残差。
    - 输出与输入 shape 相同，代表被抽取的“缓慢/多尺度”结构。
    """
    def __init__(self, channels, kernels=(3,5,7), dilations=(1,2,4,8), mix=True):
        super().__init__()
        self.blocks = nn.ModuleList(
            [DepthwiseConv1d(channels, k, d) for k in kernels for d in dilations]
        )
        self.mix = nn.Conv1d(channels, channels, kernel_size=1, bias=False) if mix else nn.Identity()

    def forward(self, x_bcn):  # [B, C, S]
        outs = [blk(x_bcn) for blk in self.blocks]  # 每个 [B,C,S]
        y = torch.stack(outs, dim=0).mean(dim=0)    # 平均聚合
        y = self.mix(y)
        # 残差约束：只取低频/慢变（通过轻量卷积已实现），直接返回作为结构项
        return y


# ---------------------------
# 2) Spectral Kernel Bank (SKB)
#    可学习复数频域模板库；还提供简洁的周期外推（period tiling）
# ---------------------------
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.r = nn.Linear(in_features, out_features, bias=bias)
        self.i = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, z):  # z: complex tensor (..., F)
        real = self.r(z.real) - self.i(z.imag)
        imag = self.r(z.imag) + self.i(z.real)
        return torch.complex(real, imag)


class SKB(nn.Module):
    """
    Spectral Kernel Bank
    - 维护 K 个可学习复数模板 W[k] ∈ C^{F} （F = S//2+1）
    - 给定 x_fft ∈ C^{B×C×F}，计算与模板的相关性，softmax 得到权重 α，得到滤波器 w=∑α_k W[k]
    - y_fft = x_fft * w （Hadamard），iFFT 得到 s_freq
    - 额外提供 extrapolate()：根据 s_freq 的主频估计周期 P，做 tiling 外推 pred_len
    """
    def __init__(self, seq_len, K=16):
        super().__init__()
        self.S = seq_len
        self.F = seq_len // 2 + 1
        self.K = K
        # 复模板（小尺度初始化）
        W_real = torch.randn(K, self.F) * 0.01
        W_imag = torch.randn(K, self.F) * 0.01
        self.W = nn.Parameter(torch.complex(W_real, W_imag))
        self.proj_q = ComplexLinear(self.F, self.F, bias=True)  # 对 x_fft 做一个轻量投影
        self.proj_k = ComplexLinear(self.F, self.F, bias=True)  # 对模板库做投影
        self.tau = nn.Parameter(torch.tensor(1.0))              # 温度

    def forward(self, x_bcn):  # [B,C,S]
        x_fft = torch.fft.rfft(x_bcn, dim=-1, norm='ortho')         # [B,C,F]
        q = self.proj_q(x_fft)                                      # [B,C,F]
        k = self.proj_k(self.W)                                     # [K,F]
        # 相似度：对 F 维做点积（取实部）
        sim = torch.einsum('bcf,kf->bck', q.conj(), k).real / math.sqrt(self.F)  # [B,C,K]
        alpha = F.softmax(sim / (self.tau.abs() + 1e-6), dim=-1)                # [B,C,K]
        # 合成滤波器
        alpha_c = alpha.to(dtype=self.W.dtype)
        w = torch.einsum('bck,kf->bcf', alpha_c, self.W)              # [B,C,F] (complex)
        y_fft = x_fft * w
        s_freq = torch.fft.irfft(y_fft, n=x_bcn.size(-1), dim=-1, norm='ortho')  # [B,C,S]
        return s_freq, alpha

    @staticmethod
    def _estimate_period(x_bcn):  # 简洁主频估计：取幅度最大频率 -> 近似周期
        # x_bcn: [B,C,S]
        B, C, S = x_bcn.shape
        fft = torch.fft.rfft(x_bcn, dim=-1, norm='ortho')  # [B,C,F]
        mag = torch.abs(fft)                                # [B,C,F]
        mag[..., 0] = 0.0                                  # 忽略 DC
        idx = mag.argmax(dim=-1)                           # [B,C]
        # 避免 0；周期 ~ S / (idx * 2) 近似（rfft 频率索引→周期）
        idx = idx.clamp(min=1)
        P = (S // idx).clamp(min=4)                        # 下限 4
        return P  # [B,C]

    def extrapolate(self, s_freq, pred_len):
        # 周期外推：按估计主周期 P 做尾段 tiling
        B, C, S = s_freq.shape
        P = self._estimate_period(s_freq)
        out = []
        for b in range(B):
            row = []
            for c in range(C):
                p = int(P[b, c].item())
                base = s_freq[b, c, -p:].detach()
                reps = (pred_len + p - 1) // p
                ext = base.repeat(reps)[:pred_len]
                row.append(ext)
            row = torch.stack(row, dim=0)  # [C, pred_len]
            out.append(row)
        out = torch.stack(out, dim=0)      # [B, C, pred_len]
        return out


# ---------------------------
# 3) Cross-Channel Affinity Extractor (CAE)
#    CAFI 的 LayerNorm 版本，返回 A_bar（跨通道相关）
# ---------------------------
class CAEBlock(nn.Module):
    def __init__(self, feature_dim, seq_len, rank=4, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.query = nn.Linear(seq_len, rank, bias=False)
        self.key   = nn.Linear(seq_len, rank, bias=False)
        self.value = nn.Linear(seq_len, seq_len, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.time_gate = nn.Parameter(torch.tensor(0.1))
        # 改 LayerNorm（在 [L,C] 上做 LN 更稳定）
        self.ln = nn.LayerNorm([seq_len, feature_dim])

    def forward(self, x_blc, return_attn=False):
        # x_blc: [B, L, C] （注意：和 B,C,S 仅是转置关系）
        x = self.ln(x_blc)
        x_t = x.transpose(1, 2)        # [B, C, L]
        Q = self.query(x_t)            # [B, C, r]
        K = self.key(x_t)              # [B, C, r]
        A = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.rank)  # [B,C,C]
        A = torch.softmax(A, dim=-1)
        V = self.value(x_t)            # [B, C, L]
        out = x_t + self.alpha * self.dropout(torch.matmul(A, V))
        out = out + self.time_gate * self.dropout(V)
        out = out.transpose(1, 2)      # [B, L, C]
        if return_attn:
            return out, A
        return out


class CAE(nn.Module):
    def __init__(self, input_shape, patch=12, rank=4, dropout=0.1):
        super().__init__()
        self.S, self.C = input_shape   # (seq_len, n_vars)
        self.patch = patch
        self.rank = rank
        self.dropout = dropout
        self.block = CAEBlock(feature_dim=self.C, seq_len=self.patch, rank=self.rank, dropout=dropout)

    def forward(self, x_bcn, return_attn=False):  # [B,C,S]
        B, C, S = x_bcn.shape
        # 滑窗块：沿时间分片，逐块做跨通道互作，再拼回
        outs, As = [], []
        for i in range(0, S, self.patch):
            seg = x_bcn[:, :, i:i+self.patch]                  # [B,C,patch']
            seg_l = seg.transpose(1, 2)                        # [B,patch',C]
            if return_attn:
                y, A = self.block(seg_l, return_attn=True)
                As.append(A)
            else:
                y = self.block(seg_l, return_attn=False)
            outs.append(y.transpose(1, 2))
        y_all = torch.cat(outs, dim=-1)[:, :, :S]              # [B,C,S]
        if return_attn:
            # 平均得到 A_bar
            A_bar = torch.stack(As, dim=0).mean(dim=0)         # [B,C,C]
            return y_all, A_bar
        return y_all


# ---------------------------
# 4) StructureExtractor：串行逐层剥离 + 门控融合
# ---------------------------
class StructureExtractor(nn.Module):
    def __init__(self, input_shape, mste_kernels=(3,5,7), mste_dils=(1,2,4,8),
                 skb_K=16, cae_patch=12, dropout=0.1, gating_hidden=128):
        super().__init__()
        S, C = input_shape
        self.mste = MSTE(C, kernels=mste_kernels, dilations=mste_dils)
        self.skb  = SKB(seq_len=S, K=skb_K)
        self.cae  = CAE(input_shape=input_shape, patch=cae_patch, rank=4, dropout=dropout)

        # 全局三路门控（按样本）
        self.global_gate = nn.Sequential(
            nn.Linear(3*C, gating_hidden), nn.ReLU(),
            nn.Linear(gating_hidden, 3), nn.Softmax(dim=-1)
        )
        # 通道级门控（按通道）
        self.channel_gate = nn.Sequential(
            nn.Linear(3, gating_hidden//2), nn.ReLU(),
            nn.Linear(gating_hidden//2, 3), nn.Sigmoid()
        )

    @staticmethod
    def _gap(x_bcn):  # [B,C,S] -> [B,C]
        return x_bcn.mean(dim=-1)

    @staticmethod
    def _per_channel_stats(x_bcn):  # [B,C,S] -> [B,C,3]  (mean, var, spectral energy)
        mean = x_bcn.mean(dim=-1)
        var  = x_bcn.var(dim=-1, unbiased=False)
        spec = torch.mean(torch.abs(torch.fft.rfft(x_bcn, dim=-1, norm='ortho')), dim=-1)
        return torch.stack([mean, var, spec], dim=-1)  # [B,C,3]

    def forward(self, x_bcn, need_components=False):
        """
        x_bcn: [B,C,S]
        returns:
           s_hat: [B,C,S]
           r    : [B,C,S]
           aux  : dict(g_global, g_channel, A_bar, components=optional)
        """
        rem = x_bcn
        s_list = []
        # 1) MSTE
        s_ms = self.mste(rem)
        rem = rem - s_ms
        s_list.append(s_ms)
        # 2) SKB
        s_freq, _ = self.skb(rem)
        rem = rem - s_freq
        s_list.append(s_freq)
        # 3) CAE
        s_cae, A_bar = self.cae(rem, return_attn=True)
        rem = rem - s_cae
        s_list.append(s_cae)

        # 全局门控
        stats = torch.cat([self._gap(s) for s in s_list], dim=-1)  # [B, 3C]
        g_global = self.global_gate(stats)                         # [B,3]

        # 通道门控
        ch_feat = self._per_channel_stats(x_bcn)                   # [B,C,3]
        g_channel = self.channel_gate(ch_feat)                     # [B,C,3]

        # 融合
        g = g_global.unsqueeze(1).unsqueeze(-1) * g_channel.unsqueeze(-1)  # [B,C,3,1]
        s_stack = torch.stack(s_list, dim=2)                                 # [B,C,3,S]
        s_hat = (g * s_stack).sum(dim=2)                                     # [B,C,S]
        r = x_bcn - s_hat

        aux = {"g_global": g_global, "g_channel": g_channel, "A_bar": A_bar}
        if need_components:
            aux["components"] = {
                "s_bcn": s_hat,  # ✅ 最终融合后的历史结构项（供外推使用）
                "s_ms": s_ms,  # ✅ 多尺度路（MSTE）
                "s_freq": s_freq,  # ✅ 频域路（SKB）
                "s_cae": s_cae,  # ✅ 通道交互路（CAE）
                # 同时保留你原来的命名，避免其它代码受影响（可选）
                "mste": s_ms,
                "skb": s_freq,
                "cae": s_cae,
        }
        return s_hat, r, aux

    def extrapolate_structure(self, components, pred_len):
        """
        components: 一个 dict，至少包含 's_bcn'（历史结构项）或包含
                    你需要的频域模板/权重（例如 'alpha', 'W'）
        返回: s_future_bcn [B, C, pred_len]，可微
        """
        s_hist = components["s_bcn"]            # [B,C,S] 历史结构项（时域）
        B, C, S = s_hist.shape
        F = S // 2 + 1

        # FFT (历史结构)
        S_fft = torch.fft.rfft(s_hist, dim=-1)  # [B,C,F], complex
        A = torch.abs(S_fft)                    # 幅度 [B,C,F]
        phi = torch.angle(S_fft)                # 相位 [B,C,F]

        # 角频率（对应 rFFT 样本点）
        # w_k = 2π k / S, k=0..F-1
        k = torch.arange(F, device=s_hist.device).reshape(1,1,F)
        omega = 2 * math.pi * k / S             # [1,1,F]

        # 未来 pred_len 个点，构造相位推进 φ + ω t
        t = torch.arange(1, pred_len+1, device=s_hist.device).reshape(1,1,pred_len,1)  # [1,1,Plen,1]
        phi_t = phi.unsqueeze(-2) + omega.unsqueeze(-2) * t                             # [B,C,Plen,F]

        # 合成未来（实信号：对称谱的等价时域表示：A*cos(·)）
        # 注意：rFFT 只保留半谱，这里用 cos 近似重建未来足够稳定且可微
        s_future = (A.unsqueeze(-2) * torch.cos(phi_t)).sum(dim=-1) / F  # [B,C,Plen]
        return s_future


# ---------------------------
# 5) RevIN
# ---------------------------
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str, mask=None):  # (B,S,C)
        if mode == 'norm':
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
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - (self.last if self.subtract_last else self.mean)
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + (self.last if self.subtract_last else self.mean)
        return x


# ---------------------------
# 6) Heads
# ---------------------------
class CrossAttnHead(nn.Module):
    def __init__(self, d_model, pred_len, n_heads=4, dropout=0.0):
        super().__init__()
        self.pred_len = pred_len
        self.q = nn.Parameter(torch.randn(pred_len, d_model) * 0.01)  # [Plen, d]
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(d_model, 1)

    def forward(self, enc_out):  # enc_out: [B*C, S, d]
        Bn, S, d = enc_out.shape
        q = self.q.unsqueeze(0).expand(Bn, -1, -1)               # [Bn, Plen, d]
        attn_out, _ = self.mha(q, enc_out, enc_out, need_weights=False)  # [Bn, Plen, d]
        y = self.proj(attn_out).squeeze(-1)                      # [Bn, Plen]
        return y


class Flatten_Head(nn.Module):
    def __init__(self, seq_len, d_model, pred_len, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(seq_len * d_model, pred_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # [B*N, S, d] 或 [B, N, S, d]
        if x.dim() == 4:
            x = x.view(x.size(0)*x.size(1), x.size(2), x.size(3))
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Pooler_Head(nn.Module):
    def __init__(self, seq_len, d_model, head_dropout=0):
        super().__init__()
        pn = seq_len * d_model
        dim = 128
        self.pooler = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(pn, pn // 2),
            nn.BatchNorm1d(pn // 2),
            nn.ReLU(),
            nn.Linear(pn // 2, dim),
            nn.Dropout(head_dropout),
        )

    def forward(self, x):
        return self.pooler(x)


# ---------------------------
# 7) 轻量频带抑制增强（可选）
# ---------------------------

class HorizonMixGate(nn.Module):
    def __init__(self, C, pred_len, emb_dim=16):
        super().__init__()
        self.pred_len = pred_len
        self.h_emb = nn.Embedding(pred_len, emb_dim)
        # 输入特征：每通道的（s_hat 与 r）的 mean/var/谱能量差
        in_dim = 5 + emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    @staticmethod
    def _stats(x_bcn):  # [B,C,S] -> [B,C,3]
        mean = x_bcn.mean(dim=-1)
        var  = x_bcn.var(dim=-1, unbiased=False)
        spec = torch.mean(torch.abs(torch.fft.rfft(x_bcn, dim=-1, norm='ortho')), dim=-1)
        return torch.stack([mean, var, spec], dim=-1)

    def forward(self, s_hat, r_bcn):  # [B,C,S], [B,C,S]
        B, C, S = s_hat.shape
        fs = self._stats(s_hat)           # [B,C,3]
        fr = self._stats(r_bcn)           # [B,C,3]
        # 取 mean/var + 谱能量差，形成5维通道描述
        feat = torch.cat([fs[..., :2], fr[..., :2], (fs[..., 2:3] - fr[..., 2:3])], dim=-1)  # [B,C,5]

        # 地平线嵌入
        h = self.h_emb(torch.arange(self.pred_len, device=s_hat.device))  # [Plen, emb_dim]
        h = h.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)              # [B,C,Plen,emb_dim]

        feat = feat.unsqueeze(2).expand(-1, -1, self.pred_len, -1)        # [B,C,Plen,5]
        g = torch.sigmoid(self.mlp(torch.cat([feat, h], dim=-1))).squeeze(-1)  # [B,C,Plen]
        return torch.clamp(g, 0.05, 0.95)

class UnifiedMoEHead(nn.Module):
    """
    统一式 MoE 预测头：
    - 把“结构外推”当作 Expert 0
    - 再加上若干数据驱动专家（重复最后块、轻量AR、一组残差编码专家）
    - 门控按 (样本, 通道, 地平线) 产生，Top-K 后 softmax
    约定：
    - enc_out: [B*C, S, d]  （来自 residual encoder）
    - s_future_bcn: [B, C, Plen]
    - r_bcn: [B, C, S]     （残差时域，用于统计/频域特征）
    返回：
    - y_pred_bcn: [B, C, Plen]
    """

    def __init__(self, seq_len, pred_len, d_model, num_res_experts=4, top_k=2, ar_kernel=7, tail_k=24, dropout=0.1):
        super().__init__()
        self.S = seq_len
        self.P = pred_len
        self.d = d_model
        self.num_res_experts = num_res_experts
        self.top_k = top_k
        self.tail_k = tail_k

        # ------- 数据驱动专家 -------
        # E1: Repeat-Last-Patch（无参）
        # E2: 线性 AR 专家（轻量 1D conv 把 [S] -> [P]）
        self.ar = nn.Conv1d(1, 1, kernel_size=ar_kernel, padding=ar_kernel//2)

        # E3..E{2+num_res_experts}: 残差编码专家（线性映射 enc_out -> P）
        # 采用“时域均值池化 + 线性到 P”的形式，轻量稳定
        self.res_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.S, self.S // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.S // 2, self.P),
            ) for _ in range(num_res_experts)
        ])
        self.res_proj = nn.Linear(self.d, 1)  # enc_out [S,d] -> [S,1] 再 squeeze 成 [S]

        # ------- 地平线门控 -------
        # 门控特征：通道统计(均值/方差) + 频域能量 + 尾部均值 + 地平线嵌入
        self.h_emb = nn.Embedding(self.P, 16)
        self.g_mlp = nn.Sequential(
            nn.Linear(1 + 1 + 1 + 1 + 16, 64),  # mean/var/spec/tail + horizon_emb
            nn.ReLU(),
            nn.Linear(64, 1)  # 对每个 expert 都会复用一遍
        )

        # 每个 expert 各自一个“偏置”向量（可让门控更容易打破对称）
        self.expert_bias = nn.Parameter(torch.zeros(2 + num_res_experts + 1))  # +1 给结构外推 E0

    @staticmethod
    def _stats_feat(x_bcn):  # [B,C,S] -> (mean,var,spec) [B,C,1] 各一项
        mean = x_bcn.mean(dim=-1, keepdim=True)
        var  = x_bcn.var(dim=-1, unbiased=False, keepdim=True)
        spec = torch.mean(torch.abs(torch.fft.rfft(x_bcn, dim=-1, norm='ortho')), dim=-1, keepdim=True)
        return mean, var, spec

    @staticmethod
    def _repeat_last_patch(x_bcn, pred_len):
        # 取最后 p = min(pred_len, S) 段重复
        B, C, S = x_bcn.shape
        p = min(pred_len, S)
        base = x_bcn[:, :, -p:]
        reps = (pred_len + p - 1) // p
        return base.repeat(1, 1, reps)[:, :, :pred_len]

    def _topk_softmax(self, logits, k, dim=-1):
        # logits: [..., E]
        if k >= logits.size(dim):
            return torch.softmax(logits, dim=dim)
        topk = torch.topk(logits, k=k, dim=dim)
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(dim, topk.indices, topk.values)
        return torch.softmax(mask, dim=dim)

    def forward(self, enc_out, s_future_bcn, r_bcn):
        """
        enc_out:     [B*C, S, d]
        s_future_bcn:[B, C, P]
        r_bcn:       [B, C, S]
        """
        device = enc_out.device
        B, C, S = r_bcn.shape
        P = self.P

        # -------- 各专家输出 --------
        # E0: 结构外推（已给）
        e0 = s_future_bcn  # [B,C,P]

        # E1: 重复最后块（在规范化域里）
        e1 = self._repeat_last_patch(r_bcn, P)  # [B,C,P]

        # E2: 线性 AR：先把 r_bcn 映射到 [B*C,1,S]，1D conv，再把时间拉到 P 上（插值到 P）
        ar_in = r_bcn.reshape(B*C, 1, S)
        ar_feat = self.ar(ar_in).squeeze(1)      # [B*C, S]
        # 线性插值/自适应重采样到 P
        ar_resampled = F.interpolate(
            ar_feat.unsqueeze(1),  # [B*C, 1, S]
            size=P,
            mode='linear',  # 1D 线性插值
            align_corners=True  # 与线性缩放一致的坐标系，别落一点点常数偏移
        ).squeeze(1)  # [B*C, P]
        e2 = ar_resampled.view(B, C, P)

        # E3..: 残差编码专家：enc_out -> [B*C,S,d] -> [B*C,S] -> expert MLP -> [B*C,P]
        # 先把 enc_out 的 d 压到 1，再交给每个 expert
        enc_s = self.res_proj(enc_out).squeeze(-1)  # [B*C, S]
        e_list = [e0, e1, e2]
        for mlp in self.res_experts:
            e_list.append(mlp(enc_s).view(B, C, P))  # 每个 [B,C,P]
        # 拼成 [B,C,E,P]
        experts = torch.stack(e_list, dim=2)  # [B,C,E,P]
        E = experts.size(2)

        # -------- 门控（样本×通道×地平线）--------
        mean, var, spec = self._stats_feat(r_bcn)  # [B,C,1] * 3
        # 尾部均值（用 enc_out 尾部的 token 作 summary）
        tail = enc_out.view(B, C, S, self.d)[:, :, -min(self.tail_k, S):, :].mean(dim=(2, 3), keepdim=True)  # [B,C,1]

        # 地平线嵌入
        h = self.h_emb(torch.arange(P, device=device))  # [P, 16]
        h = h.view(1, 1, P, 16).expand(B, C, -1, -1)    # [B,C,P,16]
        tail = tail.reshape(B, C, -1)

        # 拼接门控特征： [B,C,P, (mean,var,spec,tail,emb16) ]
        base_feat = torch.cat([mean, var, spec, tail], dim=-1)    # [B,C,4]
        base_feat = base_feat.unsqueeze(2).expand(-1, -1, P, -1)  # [B,C,P,4]
        g_feat = torch.cat([base_feat, h], dim=-1)                # [B,C,P,20]

        # 为每个 expert 复用同一个 g_mlp，再加一个 expert 个性化 bias
        g_base = self.g_mlp(g_feat).squeeze(-1)  # [B,C,P]
        g_logits = []
        for e_idx in range(E):
            g_logits.append(g_base + self.expert_bias[e_idx])
        g_logits = torch.stack(g_logits, dim=2)  # [B,C,E,P]

        # Top-K 后 softmax
        g = self._topk_softmax(g_logits, k=self.top_k, dim=2)     # [B,C,E,P]

        # -------- 混合 --------
        y = (g * experts).sum(dim=2)  # [B,C,P]
        return y


# ---------------------------
# 8) 主 Model
# ---------------------------
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_type = configs.task_type
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # RevIN
        self.revin_layer_encoder = RevIN(configs.enc_in, affine=True, subtract_last=False)

        # 结构抽取器（串行逐层剥离）
        self.struct = StructureExtractor(
            input_shape=(configs.seq_len, configs.enc_in),
            mste_kernels=getattr(configs, "mste_kernels", (3,5,7)),
            mste_dils=getattr(configs, "mste_dils", (1,2,4,8)),
            skb_K=getattr(configs, "skb_K", 16),
            cae_patch=getattr(configs, "patch", 12),
            dropout=configs.dropout,
            gating_hidden=128
        )

        # 编码器
        self.enc_embedding = DataEmbedding(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(
                    DSAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                    configs.d_model, configs.n_heads
                ),
                configs.d_model, configs.d_ff,
                dropout=configs.dropout, activation=configs.activation
            ) for _ in range(configs.e_layers)
        ], norm_layer=torch.nn.LayerNorm(configs.d_model))

        # 头
        self.cl_projection = Pooler_Head(configs.seq_len, configs.d_model, head_dropout=configs.head_dropout)
        self.head_pretrain = Flatten_Head(configs.seq_len, configs.d_model, configs.seq_len, head_dropout=configs.head_dropout)
        if configs.task_type == "c":
            configs.cls_num = get_cls_num(configs.data)
            self.head_clf = Flatten_Head(configs.seq_len, configs.d_model, configs.cls_num, head_dropout=configs.head_dropout)
        else:
            # self.head_forecast = Flatten_Head(configs.seq_len, configs.d_model, configs.pred_len, head_dropout=configs.head_dropout)
            self.head_forecast = CrossAttnHead(configs.d_model, configs.pred_len,
                                               n_heads=configs.n_heads, dropout=configs.dropout)

        # 损失
        self.labels_cl = None
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax    = nn.Softmax(dim=-1)
        self.kl  = nn.KLDivLoss(reduction='batchmean')
        self.awl = AutomaticWeightedLoss(2)
        self.mse = nn.MSELoss()

        # 预训练增强设置
        self.freqdrop_rate = configs.freqdrop_rate # 0 表示不启用
        self.struct_adapter = StructAdapter(C=configs.enc_in, d_model=configs.d_model)

    def _apply_film(self, enc_in, gamma, beta, B, C):
        S, d = enc_in.size(1), enc_in.size(2)
        x = enc_in.view(B, C, S, d)
        x = x * (1.0 + gamma) + beta
        return x.view(B * C, S, d)

    # ---------- 预训练：对比 + 重建（残差域） ----------
    def pretrainWithContrast(self, batch_x):
        B, S, C = batch_x.shape
        BN = B * C
        K = self.configs.positive_nums
        Neg = self.configs.negative_nums
        device = batch_x.device

        # 1) norm
        x_n = self.revin_layer_encoder(batch_x, 'norm')  # [B,S,C]
        x_bcn = x_n.permute(0, 2, 1).contiguous()  # [B,C,S]

        # 对原始序列进行mask
        blank_mask = get_mask(x_bcn, "geometric", self.configs.masking_ratio, self.configs.lm, S).to(dtype=x_bcn.dtype)
        x_vis_bcn  = x_bcn * blank_mask

        # 2) 显式结构分解
        s_hat_vis, r_bcn_vis, aux_vis = self.struct(x_vis_bcn, need_components=True)  # [B,C,S], [B,C,S]
        s_ms, s_freq,s_cae = aux_vis["components"]["s_ms"], aux_vis["components"]["s_freq"], aux_vis["components"]["s_cae"]

        r_flat = r_bcn_vis.view(B * C, S)

        # 4) encoder + 结构调制（前级 FiLM）
        enc = self.enc_embedding(r_flat.unsqueeze(-1))  # [B*C,S,d]
        gamma, beta = self.struct_adapter(s_ms, s_freq, s_cae)  # [B,C,S,d]
        enc = self._apply_film(enc, gamma, beta, B, C)  # [B*C,S,d]
        enc, _ = self.encoder(enc)

        # 5) 先预测残差，再重建原序列
        r_hat = self.head_pretrain(enc)  # [B*C,S]
        r_hat = r_hat.view(B, C, S).permute(0, 2, 1)  # [B,S,C]
        x_hat = s_hat_vis.permute(0, 2, 1) + r_hat  # [B,S,C]  (norm 域)


        se = (x_hat - x_n) ** 2
        loss_rb = (se * w).sum() / (w.sum() + 1e-8)

        # 7) 可选：加你原有的对比项（在残差 r_bcn 上），小权重 λ_cl\
        # --- 7) 两视角对比：SimCLR on residual r_bcn（小权重） ---
        tau = getattr(self.configs, "temperature", 0.2)
        lambda_cl = getattr(self.configs, "lambda_cl", 0.1)

        # 构造两视角（在残差上做不同的随机历史掩码 + 可选频带抑制）
        def make_view(r_bcn_in):
            B, C, S = r_bcn_in.shape
            r_view = r_bcn_in.permute(0, 2, 1).contiguous()  # [B,S,C]
            # 随机历史 span 掩码（与主掩码独立）
            mh_v, _ = self._make_masks(B, S, C, pred_len=0, device=r_bcn_in.device,
                                       hist_mask_ratio=getattr(self.configs, "cl_hist_mask_ratio", 0.25),
                                       span_len=getattr(self.configs, "cl_span_len", 8))
            r_view[mh_v] = 0.0
            # 可选 freqdrop（轻量）
            if self.freqdrop_rate > 0:
                r_view = augment_freqdrop(r_view, drop_rate=self.freqdrop_rate)
            return r_view.permute(0, 2, 1).contiguous()  # 回到 [B,C,S]

        r_v1 = make_view(r_bcn)
        r_v2 = make_view(r_bcn)

        # 编码两视角（保持同一路径：embedding + FiLM + encoder）
        def encode_residual(r_bcn_in):
            B, C, S = r_bcn_in.shape
            r_flat = r_bcn_in.view(B * C, S)
            e = self.enc_embedding(r_flat.unsqueeze(-1))
            g, b = self.struct_adapter(s_ms, s_freq, s_cae)
            e = self._apply_film(e, g, b, B, C)
            e, _ = self.encoder(e)
            z = self.cl_projection(e)  # [B*C, d]
            z = F.normalize(z, dim=1)
            return z

        z1 = encode_residual(r_v1)
        z2 = encode_residual(r_v2)

        # NT-Xent（跨 B*C 的配对）
        N = z1.size(0)
        sim = torch.mm(z1, z2.t()) / tau  # [N,N]
        labels = torch.arange(N, device=sim.device)
        loss_pos = F.cross_entropy(sim, labels)  # z1->z2
        loss_pos_rev = F.cross_entropy(sim.t(), labels)  # z2->z1
        loss_cl = 0.5 * (loss_pos + loss_pos_rev)

        # 总损失（带自动权重/或手动加权）
        loss = self.awl(lambda_cl * loss_cl, loss_rb)

        return loss, loss_cl, loss_rb, None, None, None

    # ---------- 预测：显式结构回加 ----------
    def forecast(self, x):
        """
        y = s_future + r_forecast
        - s_future: 由 StructureExtractor.extrapolate_structure 产生
        - r_forecast: 在残差编码后由 forecast 头得到
        """

        B, S, C = x.shape
        x_n = self.revin_layer_encoder(x, 'norm')
        x_bcn = x_n.permute(0, 2, 1).contiguous()

        s_hat, r_bcn, aux = self.struct(x_bcn, need_components=True)
        s_ms, s_freq, s_cae = aux["components"]["s_ms"], aux["components"]["s_freq"], aux["components"]["s_cae"]

        r_flat = r_bcn.view(B * C, S)
        enc = self.enc_embedding(r_flat.unsqueeze(-1))
        gamma, beta = self.struct_adapter(s_ms, s_freq, s_cae)
        enc = self._apply_film(enc, gamma, beta, B, C)
        enc, _ = self.encoder(enc)

        # 选一个头：Flatten_Head（更快）或 CrossAttnHead（你已验证有效）
        y_n_flat = self.head_forecast(enc)  # [B*C, pred_len]
        y_n = y_n_flat.view(B, C, self.pred_len).permute(0, 2, 1)
        y = self.revin_layer_encoder(y_n, 'denorm')
        return y

    # ---------- 分类（若需要） ----------
    def clf(self, x):
        B, S, C = x.shape
        x_n = self.revin_layer_encoder(x, 'norm')
        x_bcn = x_n.permute(0,2,1).contiguous()
        # 简单：直接在残差上建表达
        _, r_bcn, _ = self.struct(x_bcn, need_components=False)
        r_flat = r_bcn.reshape(-1, S)
        enc = self.enc_embedding(r_flat.unsqueeze(-1))
        enc, _ = self.encoder(enc)
        y = self.head_clf(enc)
        return y

    # ---------- forward ----------
    def forward(self, batch_x):
        if self.task_name == 'pretrain':
            if self.configs.pretrain_mode == "1":
                return self.pretrainWithContrast(batch_x)
            elif self.configs.pretrain_mode == "0":
                # 可保留你的旧 pretrain 以方便 ablation；默认还是走新路径
                return self.pretrainWithContrast(batch_x)
            else:
                raise ValueError("Unknown pretrain_mode")
        if self.task_name == 'finetune':
            if self.task_type == 'c':
                return self.clf(batch_x)
            elif self.task_type == 'r':
                return self.forecast(batch_x)
            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")
        raise ValueError("Unsupported task_name")
