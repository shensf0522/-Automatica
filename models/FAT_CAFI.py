import torch
import torch.nn as nn
from utils.augmentations import augment_positive_test
from utils.tools import FFT_sim
from utils.losses import AutomaticWeightedLoss
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding
import torch.nn.functional as F
import math

# CAFIBlock
class CAFIBlock(nn.Module):
    def __init__(self, feature_dim, seq_len, rank=4, dropout=0.1):
        super(CAFIBlock, self).__init__()
        self.rank = rank
        self.seq_len = seq_len

        self.query = nn.Linear(seq_len, rank)
        self.key = nn.Linear(seq_len, rank)
        self.value = nn.Linear(seq_len, seq_len)

        self.dropout = nn.Dropout(dropout)
        #self.layernorm = nn.LayerNorm([seq_len, feature_dim])
        self.layernorm = nn.LayerNorm(feature_dim)

        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.time_gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, return_attn=False):
        # x: [B, L, C]
        x_norm = self.layernorm(x)
        x_t = x_norm.transpose(1, 2)  # [B, C, L]

        Q = self.query(x_t)
        K = self.key(x_t)
        A = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.rank)
        A = torch.softmax(A, dim=-1)      # [B, C, C]
        V = self.value(x_t)

        out = torch.matmul(A, V)

        out = x_t + self.alpha * self.dropout(out)
        out = out + self.time_gate * self.dropout(V)

        # out_time = V + self.beta * self.dropout(V)
        # out = out + out_time              # [B, C, L]
        out = out.transpose(1, 2)         # [B, L, C]

        return (out, A) if return_attn else out

# Channel attention
class CAFI(nn.Module):
    def __init__(self, input_shape, dropout=0.2, patch=12, rank=4, layernorm=True):
        super(CAFI, self).__init__()
        self.seq_len, self.feature_dim = input_shape
        self.patch = patch
        self.rank = rank
        self.layernorm = layernorm

        # if self.layernorm:
        #     self.norm = nn.BatchNorm1d(self.seq_len * self.feature_dim)
        # self.norm1 = nn.BatchNorm1d(patch * self.feature_dim)
        # self.norm2 = nn.BatchNorm1d(patch * self.feature_dim)

        self.pre_ln = nn.LayerNorm(self.feature_dim)

        # 对 [B, C, patch] 的最后一维做 LN（替换原来的 BN1d(flatten)）
        self.norm1 = nn.LayerNorm(self.patch)
        self.norm2 = nn.LayerNorm(self.patch)

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

    def forward(self, x, return_attn=False):
        B, C, L = x.shape

        x = x.transpose(1, 2)  # [B, L, C]
        x = self.pre_ln(x)
        x = x.transpose(1, 2)  # [B, C, L]

        # 使用列表来收集每个 patch 的输出，彻底避免原地修改
        output_chunks = []
        A_list = []

        # 将第一个 patch 直接作为初始 chunk
        current_chunk_input = x[:, :, :self.patch]
        output_chunks.append(current_chunk_input)

        # 你的原始逻辑是非自回归的，我们遵循这个逻辑
        # 在循环中，每个新 patch 的计算都基于原始输入 x
        for i in range(self.patch, self.seq_len, self.patch):

            # --- 核心计算逻辑，与你代码一致 ---
            # 注意：这里的 chunk 来自于原始输入 x，而不是上一步的输出
            chunk = x[:, :, i - self.patch: i]
            chunk = self.norm1(chunk)
            chunk = self.agg(chunk)
            tmp = chunk + x[:, :, i: i + self.patch]
            res = tmp

            tmp = self.norm2(tmp).transpose(1, 2)  # [B, patch, C]
            if return_attn:
                tmp, A = self.block(tmp, return_attn=True)  # A: [B, C, C]
                A_list.append(A)
            else:
                tmp = self.block(tmp)
            tmp = tmp.transpose(1, 2)  # [B, C, patch]
            # --- 核心计算逻辑结束 ---

            # 得到当前 patch 的新输出，并添加到列表中
            new_chunk = res + tmp
            output_chunks.append(new_chunk)

        # 在所有循环结束后，将列表中的所有 chunks 沿长度维度（dim=2）拼接成一个张量
        output = torch.cat(output_chunks, dim=2)

        if return_attn:
            if not A_list:
                identity = torch.eye(C, device=x.device).unsqueeze(0).repeat(B, 1, 1)
                return output, identity
            A_bar = torch.stack(A_list, dim=0).mean(dim=0)  # [B, C, C]
            return output, A_bar
        return output

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

def circular_convolution(self, x, w):
    x = torch.fft.rfft(x, dim=2, norm='ortho')
    w = torch.fft.rfft(w, dim=1, norm='ortho')
    y = x * w
    out = torch.fft.irfft(y, n=self.embed_size, dim=2, norm="ortho")
    return out

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

class FreNormLaryer_KB(nn.Module):
    def __init__(self, n_knlg, input_len, bias=True):
        super(FreNormLaryer_KB, self).__init__()
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
        q = self.in_proj_q(x) # (112, 49)  -》 #一个映射，用于做注意力机制的运算（相关性）
        k = self.in_proj_k(self.kb) # (n_knlg, 49) # 把随机化的一个知识库向量做了一个映射，用于注意力机制的运算。
        v = self.in_proj_v(self.kb)
        
        attn_weights = torch.matmul(q, torch.conj_physical(k).T) * self.scaling #(112, 32) #表示的是32个知识与序列的相似性权重，attn_weights 中的每个值代表一个特定 知识向量（频域特征） 对当前样本的 重要性权重
        real = torch.real(attn_weights)
        attn_weights = F.softmax(real, dim=-1).type(torch.complex64) #(112, 32)
        # attn_weights = self.cdropout(attn_weights)

        w = torch.matmul(attn_weights, v)  # 这里的v是知识本身
        # w = self.out_proj(w)

        return w

    def forward(self, x):
        x = torch.fft.rfft(x, dim=-1, norm='ortho') #快速傅里叶变换（rfft),执行实数信号的快速傅里叶变换（RFFT），返回一个包含实数频域表示的张量，其输出形状与输入不同：由于省略了负频率，输出只包含前半部分的频域信息
        w = self.retrive_w(x) # 输入的维度是（112,49）其中112是b*n_vars，49是做的快速傅里叶变换得到的维度
        y = x * w
        out = torch.fft.irfft(y, n=self.out_dim, dim=-1, norm="ortho") # 沿着时间维度
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
        self.task_type = configs.task_type
        self.task_name = configs.task_name
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # -------------- RevIN --------------
        self.revin_layer_encoder = RevIN(configs.enc_in, affine=True, subtract_last=False)

        # -------------- 频域 KB --------------
        self.fre_norm_encoder = FreNormLaryer_KB(configs.n_knlg, configs.seq_len)

        # -------------- Transformer 编码器 --------------
        self.enc_embedding = DataEmbedding(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(
                    DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                output_attention=configs.output_attention),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation
            ) for _ in range(configs.e_layers)
        ], norm_layer=torch.nn.LayerNorm(configs.d_model))

        # -------------- Heads --------------
        self.cl_projection   = Pooler_Head(configs.seq_len, configs.d_model, head_dropout=configs.head_dropout)
        self.infer_projection = Pooler_Head(configs.seq_len, configs.d_model, head_dropout=configs.head_dropout)
        self.head_pretrain   = Flatten_Head(configs.seq_len, configs.d_model, configs.seq_len,
                                            head_dropout=configs.head_dropout)
        if configs.task_type == "c":
            configs.cls_num = get_cls_num(configs.data)
            self.head_clf = Flatten_Head(configs.seq_len, configs.d_model, configs.cls_num,
                                         head_dropout=configs.head_dropout)
        else:
            self.head_forecast = Flatten_Head(configs.seq_len, configs.d_model, configs.pred_len,
                                              head_dropout=configs.head_dropout)

        # -------------- CAFI 作为第三视角 --------------
        self.cafi = CAFI((configs.seq_len, configs.enc_in),
                         patch=getattr(configs, "patch", 12),
                         rank=getattr(configs, "rank", 4),
                         dropout=configs.dropout)
        self.caf_neg_lambda = getattr(configs, "caf_neg_lambda", 0.0)

        # -------------- 对比 & 重建依赖 --------------
        self.labels_cl = None
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.softmax    = torch.nn.Softmax(dim=-1)
        self.kl  = torch.nn.KLDivLoss(reduction='batchmean')
        self.awl = AutomaticWeightedLoss(2)
        self.mse = torch.nn.MSELoss()

        # -------------- Ā 引导的超参（可不配，有默认） --------------
        self.caf_topk  = getattr(configs, "caf_topk", 3)    # 计算每个通道权重时，Ā 行向量的 top-k 平均
        self.caf_gamma = getattr(configs, "caf_gamma", 1.0) # 给 CAFI 正样本的 logit 偏置系数
        self.caf_temp  = getattr(configs, "caf_temp", 2.0)  # 计算权重的 sigmoid 温度

    # --------------------------- 预训练：对比 + 重建 ---------------------------
    def pretrainWithContrast(self, batch_x):
        B, S, C = batch_x.shape          # (bs, seq_len, n_vars)
        bs, seq_len, n_vars = B, S, C
        BN = bs * n_vars
        K  = self.configs.positive_nums
        Neg = self.configs.negative_nums

        # 1) RevIN 归一化
        z = self.revin_layer_encoder(batch_x, 'norm')       # [B, S, C]
        x_raw_BCN = z.permute(0, 2, 1)                      # [B, C, S]

        # 2) 负样本索引（频域相似度，用你原来的 FFT_sim）
        sim_matrix = FFT_sim(x_raw_BCN)                     # [BN, BN]  #保留库跨样本间的相似性计算，也就是不不包含本样本内的特征相似度。
        negative_index = torch.topk(sim_matrix, k=Neg, dim=1).indices   # 从一行内选取最相关的k个特征（跨样本）

        # 3) 频域 KB（对原始视角）
        x_raw = x_raw_BCN.reshape(-1, seq_len)              # [BN, S]
        x_reformed, _ = self.fre_norm_encoder(x_raw)        # [BN, S]

        # 4) K 个标准增强正样本
        x_pos = augment_positive_test(x_reformed,
                                      self.configs.mask_rate,
                                      self.configs.lm,
                                      k=K).reshape(-1, seq_len)  # [K*BN, S]

        # 5) CAFI：第三视角 + 通道注意力 Ā
        x_channel_BCN, A_bar = self.cafi(x_raw_BCN, return_attn=True)  # [B,C,S], [B,C,C]
        x_ch = x_channel_BCN.reshape(-1, seq_len)                      # [BN, S]

        # 6) 拼接三视角 → 编码
        #    顺序：raw | K*pos | cafi
        x_all = torch.cat([x_raw, x_pos, x_ch], dim=0)                 # [(1+K+1)BN, S]
        enc_out = self.enc_embedding(x_all.unsqueeze(-1))              # [(1+K+1)BN, S, d]
        enc_out, _ = self.encoder(enc_out)                             # 同上

        # 7) 对比表示（池化 + 归一化）
        z_all = self.cl_projection(enc_out)                            # [(1+K+1)BN, D]
        z_all = F.normalize(z_all, dim=1)

        q     = z_all[:BN]                                             # [BN, D]
        k_pos = z_all[BN:BN*(K+1)].reshape(BN, K, -1)                  # [BN, K, D]
        k_caf = z_all[BN*(K+1):BN*(K+2)]                               # [BN, D]

        # 8) 相似度：K 个增强 + 1 个 CAFI + Neg
        pos_sim = torch.matmul(q.unsqueeze(1), k_pos.permute(0, 2, 1)).squeeze()  # [BN, K]
        caf_sim = (q * k_caf).sum(dim=1, keepdim=True)                            # [BN, 1]

        # 负样本（复用你原来的做法：用第 1 个增强视角取 hard negative）
        neg_bank = k_pos[:, 0, :][negative_index]                                  # [BN, Neg, D]
        neg_sim  = (q.unsqueeze(1) * neg_bank).sum(dim=-1)                          # [BN, Neg]

        # 9) Ā 引导：给 CAFI 正样本打“相关性加分”
        #    A_rows[b,i,:] = 第 b 个样本，第 i 个通道与其它通道的相关性

        diag = torch.diagonal(A_bar, dim1=-2, dim2=-1).contiguous() # [B, C]
        w_caf = (1.0 - diag).reshape(BN, 1)  # [BN, 1]
        caf_sim = caf_sim + self.caf_gamma * torch.sigmoid(self.caf_temp * w_caf)

        # （可选）额外 CAF-neg，小权重，提高难度但避免喧宾夺主
        if getattr(self, "caf_neg_lambda", 0.0) > 0.0:
            caf_neg_bank = k_caf[negative_index]  # [BN, Neg, D]
            caf_neg_sim = (q.unsqueeze(1) * caf_neg_bank).sum(dim=-1)  # [BN, Neg]
            neg_sim = neg_sim + self.caf_neg_lambda * caf_neg_sim



        # topk = min(self.caf_topk, n_vars)
        # w_per_channel = A_bar.topk(topk, dim=-1).values.mean(dim=-1)  # [B, C]
        # w_caf = torch.sigmoid(self.caf_temp * w_per_channel.reshape(BN, 1))  # [BN, 1]
        # caf_sim = caf_sim + self.caf_gamma * w_caf                                   # 偏置加强 CAFI 的正对齐



        # 10) 组合 logits & 温度 & KL
        logits = torch.cat([pos_sim, caf_sim, neg_sim], dim=-1)                     # [BN, K+1+Neg]
        logits = logits / self.configs.temperature
        log_probs = self.log_softmax(logits)

        # soft labels（K+1 个正样本，平均分配）
        total_pos = K + 1
        total_neg = Neg
        if (self.labels_cl is None) or (self.labels_cl.size(1) != total_pos + total_neg):
            labels = torch.zeros(BN, total_pos + total_neg, device=logits.device)
            labels[:, :total_pos] = 1.0 / total_pos
            self.labels_cl = labels
        loss_cl = self.kl(log_probs, self.labels_cl)

        # 11) Ā 引导的重建
        #     先构好 latent “银行”：K 个增强 + 1 个 CAFI + Neg
        d_model = enc_out.size(-1)
        enc_bank = enc_out                                            # [(1+K+1)BN, S, d]
        pos_enc = enc_bank[BN:BN*(K+1)].reshape(BN, K,  seq_len, d_model)   # [BN,K,S,d]
        caf_enc = enc_bank[BN*(K+1):BN*(K+2)].reshape(BN, 1,  seq_len, d_model)   # [BN,1,S,d]
        # hard negative（与上面 neg_bank 对齐，但用 encoder latent）
        hard_neg = pos_enc[:, 0, :, :][negative_index]                       # [BN,Neg,S,d]

        # Ā 加权进入重建（把 caf 分支的权重先叠加在其 logit 上，已经体现在 logits 中）
        w_mat = self.softmax(logits)                                         # [BN, K+1+Neg]
        bank  = torch.cat([pos_enc, caf_enc, hard_neg], dim=1)               # [BN, K+1+Neg, S, d]
        caf_col = K
        g_caf = (0.1 + 0.9 * torch.clamp(w_caf, 0., 1.)).squeeze(-1)  # [BN], 让门控在[0.1,1.0]
        w_mat_gated = w_mat.clone()
        w_mat_gated[:, caf_col] = w_mat_gated[:, caf_col] * g_caf  # ✅ 强化 CAFI 权重
        w_mat_final  = w_mat_gated / (w_mat_gated.sum(dim=-1, keepdim=True) + 1e-8)  # ✅ 重新归一化

        rebuild_latent = (w_mat_final.unsqueeze(-1).unsqueeze(-1) * bank).sum(dim=1)  # [BN, S, d]

        # 复原到 [B, N, S, d] → head_pretrain → denorm → MSE
        rebuild = rebuild_latent.reshape(bs, n_vars, seq_len, d_model)
        pred = self.head_pretrain(rebuild).reshape(bs, n_vars, -1).permute(0, 2, 1)  # [B,S,C]
        pred = self.revin_layer_encoder(pred, 'denorm')
        loss_rb = self.mse(batch_x, pred)

        loss = self.awl(loss_cl, loss_rb)
        return loss, loss_cl, loss_rb, None, None, None


    def forecast(self, x):
        bs, seq_len, n_vars = x.shape
        z = x
        z = self.revin_layer_encoder(z, 'norm')
        x = z
        x = x.permute(0, 2, 1)
        if self.configs.forcastMode == "freq":
            x, _ = self.fre_norm_encoder(x)  # B, N, D
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

    def clf(self, x):
        # x = (bs, seq_len, nvars)
        bs, seq_len, n_vars = x.shape
        z = x
        z = self.revin_layer_encoder(z, 'norm')
        x = z
        x = x.permute(0, 2, 1) # x (512, 7, 178)
        x ,_ = self.fre_norm_encoder(x)  # B, N, D
        x = x.reshape(-1, seq_len, 1)
        x = self.enc_embedding(x)
        x, _ = self.encoder(x)
        y = self.head_clf(x)
        # y = y.reshape(bs, n_vars, -1)
        # y = y.mean(dim=1)
        return y


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

