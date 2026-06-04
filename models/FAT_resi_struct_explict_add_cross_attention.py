import torch
import torch.nn as nn
from utils.augmentations import augment_positive_test
from utils.tools import ContrastiveWeight, AggregationRebuild, generate_CLLabels, FFT_sim
from utils.losses import AutomaticWeightedLoss
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
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
        q = self.in_proj_q(x) # (448, 169)
        k = self.in_proj_k(self.kb) # (n_knlg, 169)
        v = self.in_proj_v(self.kb)

        attn_weights = torch.matmul(q, torch.conj_physical(k).T) * self.scaling #(448, 8)
        real = torch.real(attn_weights)
        attn_weights = F.softmax(real, dim=-1).type(torch.complex64)
        # attn_weights = self.cdropout(attn_weights)

        w = torch.matmul(attn_weights, v)
        # w = self.out_proj(w)
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

class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, top_k=3):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x, dim=1, norm='ortho')
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_trend = torch.fft.irfft(xf, dim=1, norm='ortho')
        x_resi = x - x_trend
        return x_trend, x_resi

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

        self.KnowledgeGuide_encoder = FreNormLaryer_KB(configs.n_knlg, configs.seq_len)
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
        # 是否使用“残差式剥离”（True = AMD 风格，从大尺度开始一层层剥）
        self.head_forecast = Flatten_Head(configs.seq_len, configs.d_model, configs.pred_len, head_dropout=configs.head_dropout)
        self.labels_cl = None
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')
        self.awl = AutomaticWeightedLoss(2)
        self.mse = torch.nn.MSELoss()
        self.DFT_series_decomp = DFT_series_decomp(top_k=3)
        self.season_linear = Flatten_Head(configs.seq_len, 1, configs.pred_len, head_dropout=configs.head_dropout)
        # ==================== [Version 1: 严格复刻 TimeVLM 组件] ====================
        # 1. Multimodal Enhancement (对应参考代码 multimodal_enhancement)
        # 结构: Linear -> GELU -> Dropout
        self.multimodal_enhancement = nn.Sequential(
            nn.Linear(1, configs.d_model),  # 将 1维 显式特征映射到 d_model
            nn.GELU(),
            nn.Dropout(configs.dropout)
        )

        # 2. LayerNorm (对应参考代码 self.layer_norm)
        # 参考代码在 step 5 和 step 7 复用了同一个 layer_norm，或者用了相同的定义
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # 3. Cross Attention (对应参考代码 self.cross_attention)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=configs.d_model,
            num_heads=4,
            dropout=configs.dropout,
            batch_first=True
        )

        # 4. Multimodal Head (对应参考代码 self.multimodal_head)
        # 结构: Linear -> LayerNorm -> GELU -> Dropout
        # 注意：参考代码映射到 pred_len，但因为我们是预训练重建任务，目标维度保持 d_model
        self.multimodal_head = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.LayerNorm(configs.d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout)
        )

        # 5. Gating (对应参考代码 self.gate)
        # 结构: Linear(2x) -> GELU -> Linear -> Softmax
        self.gate = nn.Sequential(
            nn.Linear(configs.d_model * 2, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, 2),
            nn.Softmax(dim=-1)
        )

        # 6. Final Fusion Layer (对应参考代码 self.fusion_layer)
        # 结构: Linear(2x) -> GELU -> Dropout
        self.fusion_layer = nn.Sequential(
            nn.Linear(configs.d_model * 2, configs.d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout)
        )
        # ===========================================================================

    def pretrain_residual(self, batch_x):
        """
        残差自监督预训练：
        1）使用 RevIN 归一化 -> z
        2）用 moving_average 得到显式结构 trend
        3）残差 r = z - trend
        4）在 r 上做随机掩码，用 FAT encoder 编码，重建被掩码的 r
        5）loss = 只在掩码位置上的 MSE
        """

        bs, seq_len, n_vars = batch_x.shape
        z = batch_x  # (b, s, n)  n(=n'*f)*n(=n'*f)
        z = self.revin_layer_encoder(z, 'norm')
        # augmentation
        x_normed = z
        # 1) 显式结构（趋势） + 残差
        # 分离出的x_reason的部分，在后续的重建过程中添加这部分知识
        x_trend, x_resi = self.DFT_series_decomp(z) # 从z中分离出显示项
        x_normed = x_normed.permute(0, 2, 1)
        sim_matrix = FFT_sim(x_normed)  # (b*n, b*n)   不同样本，不同特征之间的相似性
        x_normed = x_normed.reshape(-1, seq_len)  # (b*n, s)
        negative_index = torch.topk(sim_matrix, k=self.configs.negative_nums, dim=1).indices
        # Knowledge_guide
        x_reformed, _ = self.KnowledgeGuide_encoder(x_normed)  # B, N, D
        # torch.save(x_reformed, "./ecl_336_x_reformed.pt")
        x_positives = augment_positive_test(x_reformed, self.configs.mask_rate, self.configs.lm,
                                            k=self.configs.positive_nums)
        x_positives = x_positives.reshape(-1, seq_len)
        x_all = torch.cat([x_normed, x_positives], dim=0)

        # Encoderr
        enc_out = self.enc_embedding(x_all.unsqueeze(-1))
        enc_out, _ = self.encoder(enc_out)
        # Contrastive Learning
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

        # rebuild origin
        positive_enc_out = enc_out[bs * n_vars:].reshape(bs * n_vars, self.configs.positive_nums, -1)
        negative_enc_out = positive_enc_out[:, 0, :][negative_index]
        rebuild_weight_matrix = self.softmax(similarity_matrix)
        pos_att = rebuild_weight_matrix[:, :self.configs.positive_nums].unsqueeze(1)
        neg_att = rebuild_weight_matrix[:, self.configs.positive_nums:].unsqueeze(1)
        rebuild_embed = torch.matmul(pos_att, positive_enc_out) + torch.matmul(neg_att, negative_enc_out)
        rebuild_embed = rebuild_embed.reshape(bs, n_vars, seq_len, -1)

        # ==================== [Version 1: 严格复刻 TimeVLM 逻辑] ====================

        # 准备数据：Flatten 为 [Batch * n_vars, Seq_Len, d_model] 以对齐参考代码的 shape
        # implicit_feat 对应参考代码的 temporal_features / memory_features
        implicit_feat = rebuild_embed.reshape(bs * n_vars, seq_len, -1)

        # 准备 explicit_feat (显式特征)，对应参考代码的 multimodal_features
        # x_season: [B, S, N] -> [B, N, S, 1] -> [B * N, S, 1]
        explicit_feat = x_trend.permute(0, 2, 1).unsqueeze(-1).reshape(bs * n_vars, seq_len, 1)

        # --- Step 5. Process multimodal features ---
        # 1. Enhancement (Linear -> GELU -> Dropout)
        explicit_feat = self.multimodal_enhancement(explicit_feat)
        # 2. Expand (这里维度已经是 [B*N, S, D] 了，不需要像参考代码那样 expand n_vars)
        # 3. LayerNorm
        explicit_feat = self.layer_norm(explicit_feat)

        # --- Step 6. Cross-modal attention enhancement ---
        # 参考代码：temporal_features / norm, multimodal_features / norm
        # 注意：这里使用的是 torch.norm 做除法，而不是 LayerNorm 层
        temporal_norm = implicit_feat / (torch.norm(implicit_feat, dim=-1, keepdim=True) + 1e-6)
        explicit_norm = explicit_feat / (torch.norm(explicit_feat, dim=-1, keepdim=True) + 1e-6)

        # Q=Temporal, K=Multimodal, V=Multimodal
        # 输出赋值给 explicit_feat (复用变量名，对应 multimodal_features)
        explicit_feat, _ = self.cross_attention(
            query=temporal_norm,
            key=explicit_norm,
            value=explicit_norm
        )

        # --- Step 7. Normalize cross attention output ---
        # 1. LayerNorm (再次使用 self.layer_norm)
        explicit_feat = self.layer_norm(explicit_feat)
        # 2. Multimodal Head (Linear -> LN -> GELU -> Dropout)
        explicit_feat = self.multimodal_head(explicit_feat)

        # --- Step 8. Compute gating weights ---
        # 拼接 memory (implicit) 和 multimodal (explicit)
        # implicit_feat 对应 memory_features
        combined_features = torch.cat([implicit_feat, explicit_feat], dim=-1)
        gate_weights = self.gate(combined_features)  # [B*N, S, 2]

        # --- Step 9. Weighted fusion ---
        # 权重 0 给 implicit, 权重 1 给 explicit
        fused_features = (
                gate_weights[:, :, 0:1] * implicit_feat +
                gate_weights[:, :, 1:2] * explicit_feat
        )

        # --- Step 10. Final fusion ---
        # Concat(memory, fused) -> FusionLayer -> Residual(memory)
        predictions = self.fusion_layer(
            torch.cat([implicit_feat, fused_features], dim=-1)
        ) + implicit_feat

        # ===========================================================================
        # 最终预测
        pred_x = self.head_pretrain(predictions)
        pred_x = pred_x.reshape(bs, n_vars, -1).permute(0, 2, 1)
        pred_x = self.revin_layer_encoder(pred_x, 'denorm')
        loss_rb = self.mse(batch_x, pred_x)
        loss = self.awl(loss_cl, loss_rb)
        return loss, loss_cl, loss_rb, None, None, None


    def forecast(self, x):
        bs, seq_len, n_vars = x.shape
        # 处理残差
        z = self.revin_layer_encoder(x, 'norm')
        x = z
        x = x.permute(0, 2, 1)
        if self.configs.forcastMode == "freq":
            x, _ = self.KnowledgeGuide_encoder(x)  # B, N, D
            x = x.reshape(-1, seq_len, 1)
        else:
            x = x.reshape(-1, seq_len, 1)
        x = self.enc_embedding(x)
        enc_out, _ = self.encoder(x)
        x = self.head_forecast(enc_out)
        x = x.reshape(bs, n_vars, -1).permute(0, 2, 1)
        x = self.revin_layer_encoder(x, 'denorm')
        return x

    def forward(self, batch_x):

        if self.task_name == 'pretrain':
            if self.configs.pretrain_mode == "1":
                # 原版：对比学习 + 重建
                return self.pretrainWithContrast(batch_x)
            elif self.configs.pretrain_mode == "0":
                # 如果你本来还有 pretrain() 逻辑，可以继续保留
                return self.pretrain(batch_x)
            elif self.configs.pretrain_mode == "residual":
                # 新增：基于残差的掩码重建预训练
                return self.pretrain_residual(batch_x)
            else:
                raise ValueError(f"Unsupported pretrain_mode: {self.configs.pretrain_mode}")

        if self.task_name == 'finetune':
            if self.task_type == 'c':
                dec_out = self.clf(batch_x)
                return dec_out
            elif self.task_type == 'reg':
                dec_out = self.forecast(batch_x)
                return dec_out
            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")