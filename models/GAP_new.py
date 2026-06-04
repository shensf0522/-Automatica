import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from utils.augmentations import augment_positive_test
from utils.tools import FFT_sim, generate_CLLabels
from utils.losses import AutomaticWeightedLoss
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding

from utils.masking import get_mask
from utils.tools import RevIN


class DepthwiseConv1d(nn.Module):
    def __init__(self, channels, kernel, dilation=1):
        super().__init__()
        pad = (kernel-1)//2 * dilation
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
    def __init__(self, channels, kernels=(3,5,7), dilations=(1,2,4,8)):
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
        Fh = seq_len//2 + 1
        W_real = torch.randn(K, Fh) * 0.01
        W_imag = torch.randn(K, Fh) * 0.01
        self.W = nn.Parameter(torch.complex(W_real, W_imag))
        self.proj_q = ComplexLinear(Fh, Fh)
        self.proj_k = ComplexLinear(Fh, Fh)
        self.tau = nn.Parameter(torch.tensor(1.0))
        self.seq_len = seq_len

    def forward(self, x_bcn):  # [B,C,S]
        x_fft = torch.fft.rfft(x_bcn, dim=-1, norm='ortho')  # [B,C,F]
        q = self.proj_q(x_fft)
        k = self.proj_k(self.W)                              # [K,F]
        sim = torch.einsum('bcf,kf->bck', q.conj(), k).real / math.sqrt(q.size(-1))
        alpha = F.softmax(sim / (self.tau.abs()+1e-6), dim=-1).to(self.W.dtype)  # [B,C,K]
        w = torch.einsum('bck,kf->bcf', alpha, self.W)       # [B,C,F], complex
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
        x_t = x.transpose(1,2)          # [B,C,L]
        Q, K = self.query(x_t), self.key(x_t)              # [B,C,r]
        A = torch.softmax(torch.matmul(Q, K.transpose(-1,-2)) / math.sqrt(Q.size(-1)), dim=-1)  # [B,C,C]
        V = self.value(x_t)
        out = x_t + self.alpha * self.dp(torch.matmul(A, V)) + self.beta * self.dp(V)
        return out.transpose(1,2)       # [B,L,C]

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
            seg = x_bcn[:, :, i:i+self.patch]     # [B,C,p']
            y = self.block(seg.transpose(1,2)).transpose(1,2)
            outs.append(y)
        return torch.cat(outs, dim=-1)[:, :, :S]  # [B,C,S]

# ---------------------------
# SIMGET：简单门控（并联三路 -> Softmax 融合）
#  - DWConv：提供时间上下文（每路都做，轻量）
#  - 1×1 線性：仅在“分支维度(3)”上混合，得到 logits
# ---------------------------
class SIMGET(nn.Module):
    def __init__(self, C, k_dw=5):
        super().__init__()
        pad = (k_dw - 1) // 2
        # 先做时间上下文（深度可分离卷积）
        self.dw = nn.Conv1d(3 * C, 3 * C, kernel_size=k_dw, padding=pad, groups=3 * C, bias=False)
        # 每个通道一套 3x3 的分支混合权重
        W0 = torch.eye(3).unsqueeze(0).repeat(C, 1, 1)  # [C,3,3]
        self.W = nn.Parameter(W0)  # 不能用 nn.init.eye_，因为是 3D
        self.b = nn.Parameter(torch.zeros(C, 3))

    def forward(self, s_ms, s_freq, s_cae):
        # s_*: [B,C,S]
        B, C, S = s_ms.shape
        feat = torch.cat([s_ms, s_freq, s_cae], dim=1)     # [B,3C,S]
        ctx  = self.dw(feat)                                # [B,3C,S]
        ctx  = ctx.view(B, 3, C, S).permute(0,2,3,1)       # [B,C,S,3]
        # 按通道独立： [S,3] @ [3,3] -> [S,3]
        logits = torch.einsum('bcsp,cpq->bcsq', ctx, self.W) + self.b.view(1, C, 1, 3)  # [B,C,S,3]
        g = torch.softmax(logits, dim=-1)                  # [B,C,S,3]
        g_bc3s = g.permute(0,1,3,2)                        # [B,C,3,S]
        s_stack = torch.stack([s_ms, s_freq, s_cae], dim=2)  # [B,C,3,S]
        s_mix = (g_bc3s * s_stack).sum(dim=2)              # [B,C,S]
        return s_mix, g_bc3s

# ---------------------------
# StructAdapter（FiLM调制）：(s_ms,s_freq,s_cae) -> (γ, β) ∈ R^{B,C,S,d}
# ---------------------------
class StructAdapter(nn.Module):
    def __init__(self, C, d_model, hidden=32, init_scale=0.1):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden)
        self.act = nn.GELU()
        self.gamma_fc = nn.Linear(hidden, d_model)
        self.beta_fc  = nn.Linear(hidden, d_model)
        self.scale = nn.Parameter(torch.tensor(init_scale))
        nn.init.zeros_(self.gamma_fc.weight)
        nn.init.zeros_(self.gamma_fc.bias)
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)
        with torch.no_grad():
            self.scale.fill_(0.01)  # 或者 0.0，再配合后面 warmup

    def forward(self, s_ms, s_freq, s_cae):  # [B,C,S]*3
        B, C, S = s_ms.shape
        x = torch.stack([s_ms, s_freq, s_cae], dim=-1)    # [B,C,S,3]
        h = self.act(self.fc1(x))                         # [B,C,S,h]
        gamma = self.gamma_fc(h) * self.scale             # [B,C,S,d]
        beta  = self.beta_fc(h) * self.scale              # [B,C,S,d]
        return gamma, beta

# ---------------------------
# Heads：CrossAttn（预测），Flatten（重建）
# ---------------------------
class CrossAttnHead(nn.Module):
    def __init__(self, d_model, pred_len, n_heads=4, dropout=0.0):
        super().__init__()
        self.q = nn.Parameter(torch.randn(pred_len, d_model) * 0.01)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(d_model, 1)

    def forward(self, enc_out):  # [B*C,S,d]
        Bn, S, d = enc_out.shape
        q = self.q.unsqueeze(0).expand(Bn, -1, -1)            # [Bn,P,d]
        attn_out, _ = self.mha(q, enc_out, enc_out, need_weights=False)
        y = self.proj(attn_out).squeeze(-1)                   # [Bn,P]
        return y

class Flatten_Head(nn.Module):
    def __init__(self, seq_len, d_model, out_len, head_dropout=0.0):
        super().__init__()
        self.flat = nn.Flatten(start_dim=-2)
        self.fc = nn.Linear(seq_len*d_model, out_len)
        self.dp = nn.Dropout(head_dropout)

    def forward(self, x):  # [B*,S,d]
        return self.dp(self.fc(self.flat(x)))


# ---------------------------
# 频带抑制（FreqDrop，轻量可选）
# ---------------------------
def augment_freqdrop(x, drop_rate=0.3, band_bins=1, eta=0.0):
    # x: [B,C,S] or [B,S,C] (我们在内部分两类都支持)
    if x.dim() == 3 and x.size(1) != x.size(2):  # [B,C,S]
        B,C,S = x.shape
        flat = x.view(B*C, S)
        X = torch.fft.rfft(flat, dim=1)
        M,F = X.shape
        device = X.device
        for i in range(M):
            if torch.rand(1, device=device).item() < drop_rate and F > 2:
                f0 = torch.randint(1, F-1, (1,), device=device).item()
                l = max(1, f0 - band_bins); r = min(F-1, f0+band_bins+1)
                X[i, l:r] *= eta
        out = torch.fft.irfft(X, n=S, dim=1)
        return out.view(B,C,S)
    else:  # [B,S,C]
        x_bcn = x.permute(0,2,1).contiguous()
        y = augment_freqdrop(x_bcn, drop_rate, band_bins, eta)
        return y.permute(0,2,1).contiguous()

# ---------------------------
# 主 Model
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

        # 显式结构三路
        self.mste = MSTE(self.C,
                         kernels=getattr(configs, "mste_kernels", (3,5,7)),
                         dilations=getattr(configs, "mste_dils", (1,2,4,8)))
        self.skb  = SKB(seq_len=self.S, K=getattr(configs, "skb_K", 16))
        self.cae  = CAE(input_shape=(self.S, self.C),
                        patch=getattr(configs, "patch", 12),
                        rank=4, dropout=configs.dropout)

        # SIMGET 门控融合
        self.gate = SIMGET(self.C, k_dw=5)

        # 低通（teacher 退化）
        self.lowpass = nn.Conv1d(self.C, self.C, kernel_size=9, padding=4, groups=self.C, bias=False)
        with torch.no_grad():
            self.lowpass.weight.fill_(1/9)

        # 编码器
        self.embed = DataEmbedding(1, self.d, configs.embed, configs.freq, configs.dropout)
        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(
                    DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                output_attention=configs.output_attention),
                    self.d, configs.n_heads
                ),
                self.d, configs.d_ff,
                dropout=configs.dropout, activation=configs.activation
            ) for _ in range(configs.e_layers)
        ], norm_layer=nn.LayerNorm(self.d))

        # FiLM 结构适配
        self.adapter = StructAdapter(C=self.C, d_model=self.d, hidden=32, init_scale=0.1)

        # 头：预训练重建（输出 S），预测（输出 P）
        self.head_pretrain = Flatten_Head(self.S, self.d, self.S, head_dropout=configs.head_dropout)
        if configs.task_type == "c":
            configs.cls_num = get_cls_num(configs.data)
            self.head_clf = Flatten_Head(self.S, self.d, configs.cls_num, head_dropout=configs.head_dropout)
        else:
            self.head_forecast = CrossAttnHead(self.d, self.P, n_heads=configs.n_heads, dropout=configs.dropout)

        self.struct2horizon = nn.Linear(self.S, self.P)
        # 损失
        self.mse = nn.MSELoss()
        self.awl = AutomaticWeightedLoss(2)  # 如果要自动平衡，可用；下方也支持手动权重
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # 预训练相关超参（默认值可在 configs 里改）
        self.mask_ratio   = getattr(configs, "mask_rate", 0.3)
        self.lm           = getattr(configs, "lm", 16)
        self.freqdrop_rate= getattr(configs, "freqdrop_rate", 0.0)
        self.lambda_spec  = getattr(configs, "lambda_spec", 0.02)
        self.lambda_cl    = getattr(configs, "lambda_cl", 0.1)
        self.loss_mask_only = getattr(configs, "loss_mask_only", False)  # 你要“全局 MSE”，就保持 False

    # FiLM 应用：enc_in [B*C,S,d]，γ/β [B,C,S,d]
    def _apply_film(self, enc_in, gamma, beta, B, C):
        S, d = enc_in.size(1), enc_in.size(2)
        x = enc_in.view(B, C, S, d)
        x = x * (1.0 + gamma) + beta
        return x.view(B*C, S, d)

    # ---------- 预训练：ProtectedStruct + FiLM 重建 ----------
    def pretrainWithContrast(self, batch_x):
        """
        输入: batch_x [B,S,C]
        输出: loss_total, loss_cl, loss_rec, None, None, None
        """
        B, S, C = batch_x.shape
        device = batch_x.device

        # 1) RevIN 到 norm 域
        x_n = self.revin(batch_x, 'norm')  # [B,S,C]
        x_bcn = x_n.permute(0, 2, 1).contiguous()  # [B,C,S]

        # 2) 并联显式结构（student），SIMGET 融合
        s_ms = self.mste(x_bcn)  # [B,C,S]
        s_freq = self.skb(x_bcn)  # [B,C,S]
        s_cae = self.cae(x_bcn)  # [B,C,S]
        s_mix, g = self.gate(s_ms, s_freq, s_cae)  # [B,C,S], [B,C,3,S]

        # 3) teacher 分支：stop-grad + 低通退化
        with torch.no_grad():
            s_teacher = self.lowpass(x_bcn)  # [B,C,S]
            if self.freqdrop_rate > 0:
                s_teacher = augment_freqdrop(s_teacher, drop_rate=0.2, band_bins=1, eta=0.0)

        # 4) 残差上做"几何连续掩码"
        r_bcn = x_bcn - s_teacher  # [B,C,S]
        masked_fill = get_mask(r_bcn, "geometric", self.mask_ratio, self.lm, S).to(device=device)
        mask_keep = torch.as_tensor(masked_fill, device=device, dtype=r_bcn.dtype)
        r_masked = r_bcn * mask_keep

        # 5) FiLM 注入 Encoder
        r_flat = r_masked.view(B * C, S)
        enc = self.embed(r_flat.unsqueeze(-1))  # [B*C,S,d]
        gamma, beta = self.adapter(s_ms.detach(), s_freq.detach(), s_cae.detach())
        enc = self._apply_film(enc, gamma, beta, B, C)
        enc, _ = self.encoder(enc)

        # 6) 预测残差并重建
        r_hat_flat = self.head_pretrain(enc)  # [B*C,S]
        r_hat = r_hat_flat.view(B, C, S)
        x_hat_bcn = s_teacher + r_hat  # [B,C,S]
        x_hat = x_hat_bcn.permute(0, 2, 1).contiguous()  # [B,S,C]

        # ==================== 损失计算 ====================

        # 损失1：重建损失（主损失，监督残差编码器）
        L_rec = self.mse(x_hat, x_n)

        # 损失2：重建的频谱一致性（监督残差编码器）
        def mag(t):
            return torch.abs(torch.fft.rfft(t, dim=1, norm='ortho'))  # [B,S,C] -> [B,F,C]

        L_spec = F.l1_loss(mag(x_hat), mag(x_n))  # ✅ 恢复这一行！

        # 损失3：结构分支主损失（监督MSTE/SKB/CAE学习低频）
        with torch.no_grad():
            x_lowfreq = self.lowpass(x_bcn)  # 目标：原序列的低频
        L_struct_main = self.mse(s_mix, x_lowfreq)

        # 损失4：结构分支频谱损失（额外的频域监督）
        L_struct_spec = F.l1_loss(mag(s_mix.permute(0, 2, 1)), mag(x_lowfreq.permute(0, 2, 1)))
        # 注意：mag期望 [B,S,C]，所以需要permute

        # 损失5：对比学习（可选，在残差空间）
        if self.lambda_cl > 0:
            def encode_view(r_src):
                vmask_fill = get_mask(r_src, "geometric", self.mask_ratio, self.lm, S).to(device=device)
                rv = r_src * vmask_fill
                if self.freqdrop_rate > 0:
                    rv = augment_freqdrop(rv, drop_rate=self.freqdrop_rate)
                e = self.embed(rv.view(B * C, S).unsqueeze(-1))
                g2, b2 = self.adapter(s_ms.detach(), s_freq.detach(), s_cae.detach())
                e = self._apply_film(e, g2, b2, B, C)
                e, _ = self.encoder(e)
                z = F.normalize(torch.mean(e, dim=1), dim=1)
                return z

            z1, z2 = encode_view(r_bcn), encode_view(r_bcn)
            tau = getattr(self.cfg, "temperature", 0.2)
            sim = torch.mm(z1, z2.t()) / tau
            labels = torch.arange(sim.size(0), device=device)
            L_cl = 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))
        else:
            L_cl = torch.tensor(0.0, device=device)

        # 损失6：结构平滑性正则（推动 s_mix 平滑）
        # 注意：这个和L_struct_main有点重复，可以二选一或降低权重
        # lp_x  = self.lowpass(x_bcn).detach()
        # lp_sm = self.lowpass(s_mix)
        # L_struct_smooth = F.mse_loss(lp_sm, lp_x)
        # lambda_struct_smooth = 0.05

        # ==================== 组装总损失 ====================
        lambda_struct_main = 0.5  # 结构主损失权重
        lambda_struct_spec = 0.1  # 结构频谱损失权重

        loss = (L_rec +  # 重建损失
                self.lambda_spec * L_spec +  # 重建频谱损失
                self.lambda_cl * L_cl +  # 对比损失
                lambda_struct_main * L_struct_main +  # 结构主损失 ✅
                lambda_struct_spec * L_struct_spec)  # 结构频谱损失 ✅
        # + lambda_struct_smooth * L_struct_smooth  # 可选：平滑正则

        return loss, L_cl.detach(), L_rec.detach(), None, None, None

    # ---------- 预测（微调）：同一路 FiLM 编码 + CrossAttnHead ----------
    def forecast(self, x):
        """
        输入: x [B,S,C]
        输出: y [B,P,C]  (已 denorm)
        """
        B, S, C = x.shape
        x_n = self.revin(x, 'norm')
        x_bcn = x_n.permute(0,2,1).contiguous()

        # 并联结构 + SIMGET（微调阶段不需要低通退化/teacher）
        s_ms   = self.mste(x_bcn)
        s_freq = self.skb(x_bcn)
        s_cae  = self.cae(x_bcn)
        s_mix, _ = self.gate(s_ms, s_freq, s_cae)

        # 3) 结构→未来（极轻投影头）。完全可学习，不做显式周期外推
        s_for_input = s_mix.detach() if getattr(self, "detach_struct_in_finetune", True) else s_mix
        s_future_bcn = self.struct2horizon(s_for_input)  # [B,C,P]

        # 4) 残差编码（同预训练路径的 FiLM 调制），预测未来“残差”
        #    -- 这里也建议先把结构从残差里 detach 掉几轮，避免两条路互相抢活
        r_base = x_bcn - (s_mix.detach() if getattr(self, "detach_struct_in_finetune", True) else s_mix)  # [B,C,S]
        r_flat = r_base.view(B * C, S)

        enc = self.embed(r_flat.unsqueeze(-1))  # [B*C,S,d]
        gamma, beta = self.adapter(
            s_ms.detach() if getattr(self, "detach_struct_in_finetune", True) else s_ms,
            s_freq.detach() if getattr(self, "detach_struct_in_finetune", True) else s_freq,
            s_cae.detach() if getattr(self, "detach_struct_in_finetune", True) else s_cae
        )
        enc = self._apply_film(enc, gamma, beta, B, C)
        enc, _ = self.encoder(enc)

        # 残差头：输出未来残差 r_future
        r_future_flat = self.head_forecast(enc)  # [B*C,P]
        r_future_bcn = r_future_flat.view(B, C, self.P)  # [B,C,P]

        # 5) 合成最终预测（norm 域）并反归一化
        y_n_bcn = s_future_bcn + r_future_bcn  # [B,C,P]
        y_n = y_n_bcn.permute(0, 2, 1).contiguous()  # [B,P,C]
        y = self.revin(y_n, 'denorm')
        return y

        # r_bcn = x_bcn - s_mix
        # r_flat = r_bcn.view(B*C, S)
        #
        # enc = self.embed(r_flat.unsqueeze(-1))             # [B*C,S,d]
        # gamma, beta = self.adapter(s_ms, s_freq, s_cae)    # 预测时可不用 detach
        # enc = self._apply_film(enc, gamma, beta, B, C)
        # enc, _ = self.encoder(enc)
        #
        # y_n_flat = self.head_forecast(enc)                 # [B*C,P]
        # y_n_bcn = y_n_flat.view(B, C, self.P)
        # y_n = y_n_bcn.permute(0,2,1).contiguous()          # [B,P,C]
        # y = self.revin(y_n, 'denorm')
        # return y

    # ---------- forward ----------
    def forward(self, batch_x):
        if self.task_name == 'pretrain':
            return self.pretrainWithContrast(batch_x)
        elif self.task_name == 'finetune':
            if self.task_type == 'c':
                return self.clf(batch_x)
            elif self.task_type == 'r':
                return self.forecast(batch_x)
            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")
        else:
            raise ValueError("Unsupported task_name")

