import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding
from utils.masking import get_mask
from utils.tools import RevIN


# ---------------------------
# 基础分支（保留）：MSTE / SKB / CAE + SIMGET
# ---------------------------
class DepthwiseConv1d(nn.Module):
    def __init__(self, channels, kernel, dilation=1):
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.dw = nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad,
                            dilation=dilation, groups=channels, bias=False)
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()
    def forward(self, x_bcn):  # [B,C,S]
        y = self.dw(x_bcn)
        y = self.pw(y)
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        return self.act(y)

class MSTE(nn.Module):
    def __init__(self, channels, kernels=(3, 5, 7), dilations=(1, 2, 4, 8)):
        super().__init__()
        self.blocks = nn.ModuleList([DepthwiseConv1d(channels, k, d) for k in kernels for d in dilations])
    def forward(self, x_bcn):
        outs = [blk(x_bcn) for blk in self.blocks]
        return torch.stack(outs, dim=0).mean(dim=0)  # [B,C,S]

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.r = nn.Linear(in_features, out_features, bias=bias)
        self.i = nn.Linear(in_features, out_features, bias=bias)
    def forward(self, z):  # complex
        real = self.r(z.real) - self.i(z.imag)
        imag = self.r(z.imag) + self.i(z.real)
        return torch.complex(real, imag)

class SKB(nn.Module):
    def __init__(self, seq_len, K=16):
        super().__init__()
        Fh = seq_len // 2 + 1
        W_real = torch.randn(K, Fh) * 0.01
        W_imag = torch.randn(K, Fh) * 0.01
        self.W = nn.Parameter(torch.complex(W_real, W_imag))
        self.proj_q = ComplexLinear(Fh, Fh)
        self.proj_k = ComplexLinear(Fh, Fh)
        self.tau = nn.Parameter(torch.tensor(1.0))
        self.seq_len = seq_len
    def forward(self, x_bcn):  # [B,C,S]
        x_fft = torch.fft.rfft(x_bcn, dim=-1, norm='ortho')     # [B,C,F]
        q = self.proj_q(x_fft)
        k = self.proj_k(self.W)                                 # [K,F]
        sim = torch.einsum('bcf,kf->bck', q.conj(), k).real / math.sqrt(q.size(-1))
        alpha = F.softmax(sim / (self.tau.abs() + 1e-6), dim=-1).to(self.W.dtype)  # [B,C,K]
        w = torch.einsum('bck,kf->bcf', alpha, self.W)          # [B,C,F], complex
        y_fft = x_fft * w
        s = torch.fft.irfft(y_fft, n=self.seq_len, dim=-1, norm='ortho')
        return s  # [B,C,S]

class CAEBlock(nn.Module):
    def __init__(self, feature_dim, seq_len, rank=4, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(seq_len, rank, bias=False)
        self.key   = nn.Linear(seq_len, rank, bias=False)
        self.value = nn.Linear(seq_len, seq_len, bias=False)
        self.ln = nn.LayerNorm([seq_len, feature_dim])
        self.dp = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta  = nn.Parameter(torch.tensor(0.1))
    def forward(self, x_blc):  # [B,L,C]
        x = self.ln(x_blc)
        x_t = x.transpose(1, 2)           # [B,C,L]
        Q, K = self.query(x_t), self.key(x_t)  # [B,C,r]
        A = torch.softmax(torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(Q.size(-1)), dim=-1)  # [B,C,C]
        V = self.value(x_t)
        out = x_t + self.alpha * self.dp(torch.matmul(A, V)) + self.beta * self.dp(V)
        return out.transpose(1, 2)        # [B,L,C]

class CAE(nn.Module):
    def __init__(self, input_shape, patch=12, rank=4, dropout=0.1):
        super().__init__()
        S, C = input_shape
        self.patch = patch
        self.block = CAEBlock(C, patch, rank, dropout)
    def forward(self, x_bcn):  # [B,C,S]
        B, C, S = x_bcn.shape
        outs = []
        for i in range(0, S, self.patch):
            seg = x_bcn[:, :, i:i+self.patch]        # [B,C,p']
            y = self.block(seg.transpose(1, 2)).transpose(1, 2)
            outs.append(y)
        return torch.cat(outs, dim=-1)[:, :, :S]     # [B,C,S]

class SIMGET(nn.Module):
    def __init__(self, C, k_dw=5):
        super().__init__()
        pad = (k_dw - 1) // 2
        self.dw = nn.Conv1d(3 * C, 3 * C, kernel_size=k_dw, padding=pad, groups=3 * C, bias=False)
        W0 = torch.eye(3).unsqueeze(0).repeat(C, 1, 1)  # [C,3,3]
        self.W = nn.Parameter(W0)
        self.b = nn.Parameter(torch.zeros(C, 3))
    def forward(self, s_ms, s_freq, s_cae):
        B, C, S = s_ms.shape
        feat = torch.cat([s_ms, s_freq, s_cae], dim=1)   # [B,3C,S]
        ctx  = self.dw(feat)
        ctx  = ctx.view(B, 3, C, S).permute(0, 2, 3, 1)  # [B,C,S,3]
        logits = torch.einsum('bcsp,cpq->bcsq', ctx, self.W) + self.b.view(1, C, 1, 3)
        g = torch.softmax(logits, dim=-1)                # [B,C,S,3]
        g_bc3s = g.permute(0, 1, 3, 2)                   # [B,C,3,S]
        s_stack = torch.stack([s_ms, s_freq, s_cae], dim=2)  # [B,C,3,S]
        s_mix = (g_bc3s * s_stack).sum(dim=2)            # [B,C,S]
        return s_mix, g_bc3s


# ---------------------------
# FiLM：由显式结构生成 (γ,β) 调制残差编码
# ---------------------------
class StructAdapter(nn.Module):
    def __init__(self, C, d_model, hidden=32, init_scale=0.1):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden)
        self.act = nn.GELU()
        self.gamma_fc = nn.Linear(hidden, d_model)
        self.beta_fc  = nn.Linear(hidden, d_model)
        self.scale = nn.Parameter(torch.tensor(init_scale))
        nn.init.zeros_(self.gamma_fc.weight); nn.init.zeros_(self.gamma_fc.bias)
        nn.init.zeros_(self.beta_fc.weight);  nn.init.zeros_(self.beta_fc.bias)
        with torch.no_grad(): self.scale.fill_(0.01)
    def forward(self, s_ms, s_freq, s_cae):  # [B,C,S]*3
        B, C, S = s_ms.shape
        x = torch.stack([s_ms, s_freq, s_cae], dim=-1)  # [B,C,S,3]
        h = self.act(self.fc1(x))
        gamma = self.gamma_fc(h) * self.scale          # [B,C,S,d]
        beta  = self.beta_fc(h)  * self.scale          # [B,C,S,d]
        return gamma, beta


# ---------------------------
# Cross-Attn 融合：Q=残差编码, K/V=结构编码（内容自适应融合）
# ---------------------------
class CrossFusion(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1, ff_mult=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dp = nn.Dropout(dropout)
    def forward(self, R_enc, S_enc):  # both: [B*, S, d]
        attn, _ = self.mha(R_enc, S_enc, S_enc, need_weights=False)
        x = self.ln1(R_enc + self.dp(attn))
        x = self.ln2(x + self.dp(self.ffn(x)))
        return x  # [B*, S, d]


# ---------------------------
# 逐 token 的重建头 & 未来预测头（你原有的）
# ---------------------------
class TokenwiseReconHead(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.dp = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, 1)
    def forward(self, x):  # [B*, S, d]
        return self.proj(self.dp(x)).squeeze(-1)  # [B*, S]

class CrossAttnHead(nn.Module):
    def __init__(self, d_model, pred_len, n_heads=4, dropout=0.0):
        super().__init__()
        self.q = nn.Parameter(torch.randn(pred_len, d_model) * 0.01)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(d_model, 1)
    def forward(self, enc_out):  # [B*, S, d]
        Bn, S, d = enc_out.shape
        q = self.q.unsqueeze(0).expand(Bn, -1, -1)  # [Bn,P,d]
        attn_out, _ = self.mha(q, enc_out, enc_out, need_weights=False)
        y = self.proj(attn_out).squeeze(-1)         # [Bn,P]
        return y


# ---------------------------
# 主 Model（精简+改造版）
# ---------------------------
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.cfg = configs
        self.task_type = configs.task_type
        self.task_name = configs.task_name
        self.S = configs.seq_len
        self.P = configs.pred_len
        self.C = configs.enc_in
        self.d = configs.d_model

        # RevIN
        self.revin = RevIN(self.C, affine=True, subtract_last=False)

        # 显式结构三路 + 门控
        self.mste = MSTE(self.C,
                         kernels=getattr(configs, "mste_kernels", (3, 5, 7)),
                         dilations=getattr(configs, "mste_dils", (1, 2, 4, 8)))
        self.skb  = SKB(seq_len=self.S, K=getattr(configs, "skb_K", 16))
        self.cae  = CAE(input_shape=(self.S, self.C),
                        patch=getattr(configs, "patch", 12),
                        rank=4, dropout=configs.dropout)
        self.gate = SIMGET(self.C, k_dw=5)

        # “弱 teacher” 仅用于结构正则（不再做 x = s + r）
        self.lowpass = nn.Conv1d(self.C, self.C, kernel_size=9, padding=4, groups=self.C, bias=False)
        with torch.no_grad():
            self.lowpass.weight.fill_(1 / 9)

        # 编码器
        self.embed = DataEmbedding(1, self.d, configs.embed, configs.freq, configs.dropout)
        self.embed_struct = DataEmbedding(1, self.d, configs.embed, configs.freq, configs.dropout)

        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(
                    DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                output_attention=configs.output_attention),
                    self.d, configs.n_heads
                ),
                self.d, configs.d_ff, dropout=configs.dropout, activation=configs.activation
            ) for _ in range(configs.e_layers)
        ], norm_layer=nn.LayerNorm(self.d))

        # FiLM：由结构提示调制残差编码
        self.adapter = StructAdapter(C=self.C, d_model=self.d, hidden=32, init_scale=0.1)

        # Cross-Attn 融合
        self.fusion = CrossFusion(self.d, n_heads=configs.n_heads, dropout=configs.dropout)

        # 头：预训练重建（mask位置重建原始值），预测（未来原始值）
        self.head_recon = TokenwiseReconHead(self.d, dropout=configs.head_dropout)
        if configs.task_type == "c":
            raise NotImplementedError("此精简版只保留回归/预测路径")
        else:
            self.head_forecast = CrossAttnHead(self.d, self.P, n_heads=configs.n_heads, dropout=configs.dropout)

        # 损失与超参
        self.mse = nn.MSELoss(reduction="none")
        self.mask_ratio   = getattr(configs, "mask_rate", 0.3)
        self.lm           = getattr(configs, "lm", 16)
        self.lambda_spec  = getattr(configs, "lambda_spec", 0.02)
        self.lambda_struct= getattr(configs, "lambda_struct", 0.5)
        self.lambda_struct_spec = getattr(configs, "lambda_struct_spec", 0.1)
        self.lambda_decor = getattr(configs, "lambda_decor", 1e-3)  # 去相关正则

    # FiLM 应用：enc_in [B*C,S,d]，γ/β [B,C,S,d]
    def _apply_film(self, enc_in, gamma, beta, B, C):
        S, d = enc_in.size(1), enc_in.size(2)
        x = enc_in.view(B, C, S, d)
        x = x * (1.0 + gamma) + beta
        return x.view(B * C, S, d)

    @staticmethod
    def _decorrelation_loss(S_tokens, R_tokens):
        """
        S_tokens, R_tokens: [B*, S, d]  (detach: 你可视需要在调用处处理)
        目标：min || Corr(S, R) ||_F^2
        """
        Bn, S, d = S_tokens.shape
        N = Bn * S
        S_flat = S_tokens.reshape(-1, d)     # [N, d]
        R_flat = R_tokens.reshape(-1, d)     # [N, d]
        S_flat = (S_flat - S_flat.mean(0)) / (S_flat.std(0) + 1e-6)
        R_flat = (R_flat - R_flat.mean(0)) / (R_flat.std(0) + 1e-6)
        C = (S_flat.T @ R_flat) / N          # [d, d]
        return (C ** 2).sum()

    # ---------- 预训练：mask重建（融合后直接回到“原始域”） ----------
    def pretrainWithCrossFusion(self, batch_x):
        """
        输入: batch_x [B,S,C]
        输出: loss_total, L_decor, L_rec_masked, ...
        """
        B, S, C = batch_x.shape
        device = batch_x.device

        # 1) RevIN
        x_n = self.revin(batch_x, 'norm')                  # [B,S,C]
        x_bcn = x_n.permute(0, 2, 1).contiguous()          # [B,C,S]

        # 2) 显式结构（student）+ 门控融合
        s_ms   = self.mste(x_bcn)                          # [B,C,S]
        s_freq = self.skb(x_bcn)
        s_cae  = self.cae(x_bcn)
        s_mix, _ = self.gate(s_ms, s_freq, s_cae)          # [B,C,S]

        # 3) 几何mask在“残差基底”上（用 s_mix 定义残差，更贴近结构/残差分工）
        r_base = x_bcn - s_mix.detach()                    # stop-grad 以稳住分工
        masked_fill = get_mask(r_base, "geometric", self.mask_ratio, self.lm, S).to(device=device)
        mask_keep = torch.as_tensor(masked_fill, device=device, dtype=r_base.dtype)  # [B,C,S]
        r_masked = r_base * mask_keep

        # 4) 残差编码（FiLM by structure）
        r_flat = r_masked.view(B * C, S)
        enc_r = self.embed(r_flat.unsqueeze(-1))           # [B*C,S,d]
        gamma, beta = self.adapter(s_ms.detach(), s_freq.detach(), s_cae.detach())
        enc_r = self._apply_film(enc_r, gamma, beta, B, C)
        enc_r, _ = self.encoder(enc_r)                     # [B*C,S,d]

        # 5) 结构编码（不参与反传或弱参与，避免互抢）
        s_flat = s_mix.view(B * C, S)
        enc_s = self.embed_struct(s_flat.unsqueeze(-1))    # [B*C,S,d]
        enc_s = enc_s.detach()

        # 6) Cross-Attn 融合（Q=残差编码, K/V=结构编码）
        enc_fused = self.fusion(enc_r, enc_s)              # [B*C,S,d]

        # 7) 逐 token 重建“原始值”
        x_hat_flat = self.head_recon(enc_fused)            # [B*C,S]
        x_hat_bcn  = x_hat_flat.view(B, C, S)              # [B,C,S]
        x_hat      = x_hat_bcn.permute(0, 2, 1).contiguous()  # [B,S,C]

        # ==================== 损失 ====================
        # (1) 只在 mask 处的重建
        mse_tok = self.mse(x_hat_bcn, x_bcn)               # [B,C,S]
        L_rec_masked = (mse_tok * mask_keep).sum() / mask_keep.sum().clamp_min(1.0)

        # (2) 频谱一致性（全局小权重）
        def mag(t): return torch.abs(torch.fft.rfft(t, dim=1, norm='ortho'))
        L_spec = F.l1_loss(mag(x_hat), mag(x_n))

        # (3) 结构正则（与低通弱对齐）
        with torch.no_grad():
            x_lowfreq = self.lowpass(x_bcn)
        L_struct_main = F.mse_loss(s_mix, x_lowfreq)
        L_struct_spec = F.l1_loss(mag(s_mix.permute(0, 2, 1)), mag(x_lowfreq.permute(0, 2, 1)))

        # (4) 去相关（互补）正则：鼓励结构/残差“分工”
        L_decor = self._decorrelation_loss(enc_s, enc_r)

        loss = (L_rec_masked
                + self.lambda_spec * L_spec
                + self.lambda_struct * L_struct_main
                + self.lambda_struct_spec * L_struct_spec
                + self.lambda_decor * L_decor)

        return loss, L_decor.detach(), L_rec_masked.detach(), None, None, None

    # ---------- 预测（微调）：同一路编码 + 融合 + Query头 ----------
    def forecast(self, x):
        """
        输入: x [B,S,C] ; 输出: y [B,P,C] (denorm)
        """
        B, S, C = x.shape
        x_n = self.revin(x, 'norm')
        x_bcn = x_n.permute(0, 2, 1).contiguous()

        # 显式结构
        s_ms   = self.mste(x_bcn)
        s_freq = self.skb(x_bcn)
        s_cae  = self.cae(x_bcn)
        s_mix, _ = self.gate(s_ms, s_freq, s_cae)

        # 残差基底（可选 detach，前几轮更稳）
        r_base = x_bcn - s_mix.detach()
        r_flat = r_base.view(B * C, S)

        enc_r = self.embed(r_flat.unsqueeze(-1))
        gamma, beta = self.adapter(s_ms.detach(), s_freq.detach(), s_cae.detach())
        enc_r = self._apply_film(enc_r, gamma, beta, B, C)
        enc_r, _ = self.encoder(enc_r)                     # [B*C,S,d]

        enc_s = self.embed_struct(s_mix.view(B * C, S).unsqueeze(-1))  # [B*C,S,d]

        enc_fused = self.fusion(enc_r, enc_s)              # [B*C,S,d]

        y_n_flat = self.head_forecast(enc_fused)           # [B*C,P]
        y_n_bcn  = y_n_flat.view(B, C, self.P)
        y_n = y_n_bcn.permute(0, 2, 1).contiguous()        # [B,P,C]

        y = self.revin(y_n, 'denorm')
        return y

    def forward(self, batch_x):
        if self.task_name == 'pretrain':
            return self.pretrainWithCrossFusion(batch_x)
        elif self.task_name == 'finetune':
            assert self.task_type == 'r'
            return self.forecast(batch_x)
        else:
            raise ValueError("Unsupported task_name")
