import torch
import torch.nn as nn
from utils.augmentations import augment_positive_test
from utils.tools import ContrastiveWeight, AggregationRebuild, generate_CLLabels, FFT_sim
from utils.losses import AutomaticWeightedLoss
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding

import torch.nn.functional as F

def moving_average(x, kernel_size):
    """
    x: [B, S, N]  (已经是 RevIN 归一化后的序列)
    在时间维度 S 上做滑动平均（非因果 / 近似因果都可以，这里用简单的前置 padding）
    """
    if kernel_size <= 1:
        return x

    B, S, N = x.shape
    # 转成 [B*N, 1, S] 在时间维度做 avg_pool1d
    x_bn = x.permute(0, 2, 1).reshape(-1, 1, S)  # [B*N, 1, S]
    pad = kernel_size - 1
    # 在左侧补齐，做一个简单的“向前看 kernel_size 个点”的平均
    x_pad = F.pad(x_bn, (pad, 0), mode='replicate')  # [B*N, 1, S+pad]
    trend = F.avg_pool1d(x_pad, kernel_size, stride=1)  # [B*N, 1, S]
    trend = trend.reshape(B, N, S).permute(0, 2, 1)  # [B, S, N]
    return trend

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
        # finetube weight


        self.decomp_kernel = getattr(configs, 'decomp_kernel', 25)

        self.head_forecast = Flatten_Head(configs.seq_len, configs.d_model, configs.pred_len, head_dropout=configs.head_dropout)
        self.labels_cl = None
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')
        self.awl = AutomaticWeightedLoss(2)
        self.mse = torch.nn.MSELoss()
        self.trend_projector_forecast = nn.Linear(configs.seq_len, configs.pred_len)
        # #趋势项的预测
        self.trend_projection_pretrain = nn.Linear(1, configs.d_model)
        # 在enc_out处融合趋势的编码


    def pretrain_residual(self, batch_x):
        """
        残差自监督预训练：
        1）使用 RevIN 归一化 -> z
        2）用 moving_average 得到显式结构 trend
        3）残差 r = z - trend
        4）在 r 上做随机掩码，用 FAT encoder 编码，重建被掩码的 r
        5）loss = 只在掩码位置上的 MSE
        """
        bs, seq_len, n_vars = batch_x.shape  # [B, S, N]

        # 1) 显式结构（趋势） + 残差
        trend = moving_average(batch_x, self.decomp_kernel)  # [B, S, N]
        res_ts = batch_x - trend  # [B, S, N]

        # # 修改为多尺度的抽离
        # trend, _ = self.compute_trend(batch_x)  # [B, S, N] + list
        # res_ts = batch_x - trend  # [B, S, N]

        # 2) RevIN 归一化（完全沿用 FAT 的逻辑）
        z = self.revin_layer_encoder(res_ts, 'norm')  # [B, S, N]

        r_normed = z
        r_normed = r_normed.permute(0, 2, 1)
        sim_matrix = FFT_sim(r_normed)  # (b*n, b*n)   不同样本，不同特征之间的相似性
        r_normed = r_normed.reshape(-1, seq_len)  # (b*n, s)
        negative_index = torch.topk(sim_matrix, k=self.configs.negative_nums, dim=1).indices
        # Knowledge_guide
        r_reformed, _ = self.KnowledgeGuide_encoder(r_normed)  # B, N, D
        # torch.save(x_reformed, "./ecl_336_x_reformed.pt")
        r_positives = augment_positive_test(r_reformed, self.configs.mask_rate, self.configs.lm,
                                            k=self.configs.positive_nums)
        r_positives = r_positives.reshape(-1, seq_len)
        r_all = torch.cat([r_normed, r_positives], dim=0)

        # Encoderr
        enc_out = self.enc_embedding(r_all.unsqueeze(-1))
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
        trend_feat_in = trend.permute(0, 2, 1).unsqueeze(-1)
        trend_feat = self.trend_projection_pretrain(trend_feat_in)

        fused_embed = rebuild_embed + trend_feat
        pred_x = self.head_pretrain(fused_embed)
        pred_x = pred_x.reshape(bs, n_vars, -1).permute(0, 2, 1)
        pred_x = self.revin_layer_encoder(pred_x, 'denorm')
        loss_rb = self.mse(batch_x, pred_x)

        loss = self.awl(loss_cl, loss_rb)

        return loss, loss_cl, loss_rb, None, None, None

    def pretrainWithContrast(self, batch_x):
        bs, seq_len, n_vars = batch_x.shape   # 这里对应的就是bs,178,1
        z = batch_x  # (b, s, n)  n(=n'*f)*n(=n'*f)
        z = self.revin_layer_encoder(z, 'norm')  # 保留batch 和 n_vars，仅在seq 上计算均值，每一个batch的每一个n_vars 计算了一个均值
        # augmentation
        x_raw = z  # (z:16,96,7)
        x_raw = x_raw.permute(0, 2, 1) # (b,n,s)
        sim_matrix = FFT_sim(x_raw)  # (b*n, b*n)  不同样本，不同特征之间的相似性
        x_raw = x_raw.reshape(-1, seq_len) # (b*n, s)  (x_raw:16,7,96  ->  112,96)
        # negative_case
        negative_index = torch.topk(sim_matrix, k = self.configs.negative_nums, dim=1).indices   # 找出每一行最相似的特征
        # fre norm
        x_reformed, _ = self.fre_norm_encoder(x_raw)  # B, N, D  (112,96)
        # positive_cases
        x_positives = augment_positive_test(x_reformed, self.configs.mask_rate, self.configs.lm, k=self.configs.positive_nums) # 根据遮挡，噪声添加，缩放三种方式生成了正样本
        x_positives = x_positives.reshape(-1, seq_len) # 对正样本仅保留了seq的维度  (112.3,96) -> (336,96)
        ## ( (K+1)*b*n, s)
        x_all = torch.cat([x_raw,  x_positives], dim=0)  # (b*n*4, seq) (448,96)
        #Cl_learning
        ## mask only positive samples
        # mask = get_mask(x_positives, "geometric", self.configs.mask_rate, self.configs.lm, seq_len).squeeze()
        # x_positive_masked = x_all[ bs * n_vars:] * mask
        ## encode all sampels
        enc_out =  self.enc_embedding(x_all.unsqueeze(-1)) #x_all.unsqueeze(-1) -> (b*n*4, seq, 1)  - > (b*n*4, seq, 512)
        enc_out, _ = self.encoder(enc_out)
        # bs * n_vars * （k+1） , d_model
        ############# cl loss ##############
        s_enc_out = self.cl_projection(enc_out)   # [(bs * n_vars) x dimension]
        s_enc_out_norm = F.normalize(s_enc_out, dim=1)
        s_q = s_enc_out_norm[: bs * n_vars]
        s_k = s_enc_out_norm[bs * n_vars:].reshape(bs * n_vars, self.configs.positive_nums, -1)  # 正样本
        if self.labels_cl is None:
            self.labels_cl = generate_CLLabels(x_raw, self.configs.positive_nums, self.configs.negative_nums)
        if self.configs.positive_nums == 1:
            positive_similarity_matrix = torch.matmul(s_q.unsqueeze(1), s_k.permute(0, 2, 1)).squeeze(-1)
        else:
            positive_similarity_matrix = torch.matmul(s_q.unsqueeze(1), s_k.permute(0, 2, 1)).squeeze()   # bs * pos_num
        if self.configs.negative_nums == 1:
            negative_similarity_matrix = torch.matmul(s_q.unsqueeze(1), s_k[:, 0, :][negative_index].permute(0, 2, 1)).squeeze(-1)
        else:
            negative_similarity_matrix = torch.matmul(s_q.unsqueeze(1), s_k[:, 0, :][negative_index].permute(0, 2, 1)).squeeze()
        similarity_matrix = torch.cat([positive_similarity_matrix, negative_similarity_matrix], dim=-1)
        similarity_matrix = similarity_matrix / self.configs.temperature
        similarity_normed = self.log_softmax(similarity_matrix)

        loss_cl = self.kl(similarity_normed, self.labels_cl)

        ######### rebuild loss ##############
        positive_enc_out = enc_out[bs * n_vars:].reshape(bs * n_vars, self.configs.positive_nums, -1)
        hardNegative_enc_out = positive_enc_out[:, 0, :][negative_index]
        rebuild_weight_matrix = self.softmax(similarity_matrix)
        pos_att = rebuild_weight_matrix[:, :self.configs.positive_nums].unsqueeze(1)
        neg_att = rebuild_weight_matrix[:, self.configs.positive_nums:].unsqueeze(1)
        rebuild_embed = torch.matmul(pos_att,  positive_enc_out) + torch.matmul(neg_att, hardNegative_enc_out)
        rebuild_embed = rebuild_embed.reshape(bs, n_vars, seq_len, -1)
        ## reconstruct
        pred_x = self.head_pretrain(rebuild_embed)
        pred_x = pred_x.reshape(bs, n_vars, -1).permute(0, 2, 1)
        pred_x = self.revin_layer_encoder(pred_x, 'denorm')
        loss_rb = self.mse(batch_x, pred_x)
        loss = self.awl(loss_cl, loss_rb)

        return loss, loss_cl, loss_rb, None, None, None

    def forecast(self, x):
        # bs, seq_len, n_vars = x.shape
        # # 添加moving_average
        # trend_input = moving_average(x, self.decomp_kernel)
        # res_input = x - trend_input
        # z = self.revin_layer_encoder(res_input, 'norm')
        # x = z
        # x = x.permute(0, 2, 1)
        # if self.configs.forcastMode == "freq":
        #     x, _ = self.KnowledgeGuide_encoder(x)  # B, N, D
        #     x = x.reshape(-1, seq_len, 1)
        # else:
        #     x = x.reshape(-1, seq_len, 1)
        # enc_emb = self.enc_embedding(x)
        # enc_out, _ = self.encoder(enc_emb)
        # res_pred = self.head_forecast(enc_out)
        # res_pred = res_pred.reshape(bs, n_vars, -1).permute(0, 2, 1)
        # res_pred = self.revin_layer_encoder(res_pred, 'denorm')
        # #合并
        # trend_input = trend_input.reshape(-1,seq_len,1)
        # trend_in_permuted = self.enc_embedding(trend_input)
        #
        #
        # trend_pred = self.trend_projection(trend_in_permuted)
        # # 换回 [B, P, N]
        # trend_pred = trend_pred.reshape(-1, n_vars, self.configs.pred_len).permute(0, 2, 1)
        # final_pred = res_pred + trend_pred
        # return final_pred
        '''
        ========================分隔线=======================================
        直接预测
        '''
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
        '''
        在预测的时候，分离出趋势项，依靠encoder预测残差项，然后再加回去
        '''
        # bs, seq_len, n_vars = x.shape
        #
        # # 1. 分解
        # trend = moving_average(x, self.decomp_kernel)
        # res_ts = x - trend
        #
        # # 2. 残差路 (进 Encoder)
        # z_res = self.revin_layer_encoder(res_ts, 'norm')
        # # [B, S, N] -> [B*N, S, 1] (根据你的 DataEmbedding 需求调整)
        # x = z_res
        # x = x.permute(0, 2, 1)
        # if self.configs.forcastMode == "freq":
        #     x, _ = self.KnowledgeGuide_encoder(x)  # B, N, D
        #     x = x.reshape(-1, seq_len, 1)
        # else:
        #     x = x.reshape(-1, seq_len, 1)
        # enc_in = self.enc_embedding(x)
        # enc_out, _ = self.encoder(enc_in)  # 得到 Residual Features
        #
        # trend_pred = self.trend_projector_forecast(trend.permute(0, 2, 1)).permute(0, 2, 1)  # 需在init定义
        #
        # pred_res = self.head_forecast(enc_out)
        # pred_res = pred_res.reshape(bs, n_vars, -1).permute(0, 2, 1)
        # pred_res = self.revin_layer_encoder(pred_res, 'denorm')
        #
        # return pred_res + trend_pred

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