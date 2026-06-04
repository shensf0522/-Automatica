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


    def retrive_w(self, x):

        q = self.in_proj_q(x) # (448, 169)

        k = self.in_proj_k(self.kb) # (n_knlg, 169)

        v = self.in_proj_v(self.kb)

        attn_weights = torch.matmul(q, torch.conj_physical(k).T) * self.scaling #(448, 8)

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



        self.use_residual_pretrain = getattr(configs, 'use_residual_pretrain', False)

        self.decomp_kernel = getattr(configs, 'decomp_kernel', 25)

        self.residual_mask_rate = getattr(configs, 'residual_mask_rate', 0.5)



        # ====== 预留：是否在结构预测中使用 AMD ======

        self.use_amd_struct_head = getattr(configs, 'use_amd_struct_head', False)

        raw_kernels = getattr(configs, 'decomp_kernels', None)

        if raw_kernels is None:

            self.decomp_kernels = [self.decomp_kernel]

        elif isinstance(raw_kernels, (list, tuple)):

            self.decomp_kernels = list(raw_kernels)

        else:

            # 兼容单个 int 或 "9,19,37" 格式

            self.decomp_kernels = [raw_kernels]



        # 是否使用“残差式剥离”（True = AMD 风格，从大尺度开始一层层剥）

        self.hierarchical_trend = getattr(configs, 'hierarchical_trend', True)



        self.head_trend_forecast = nn.Linear(configs.seq_len, configs.pred_len, bias=True)

        self.head_forecast = Flatten_Head(configs.seq_len, configs.d_model, configs.pred_len, head_dropout=configs.head_dropout)

        self.labels_cl = None

        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

        self.softmax = torch.nn.Softmax(dim=-1)

        self.kl = torch.nn.KLDivLoss(reduction='batchmean')

        self.awl = AutomaticWeightedLoss(2)

        self.mse = torch.nn.MSELoss()



        # self.trend_extractor = LearnableMultiScaleTrend(

        #     seq_len=configs.seq_len,

        #     n_vars=configs.enc_in,  # 根据你的输入通道数

        #     kernel_sizes=self.decomp_kernels,  # 使用你解析好的 self.decomp_kernels 列表

        #     hidden_dim=configs.d_model  # 可选，用你的 d_model

        # )



    def compute_trend(self, x):

        """

        统一的趋势计算接口：

        - 训练下游任务时给 trend_y / trend_hat 用

        - 残差预训练时给 res_ts = x - trend 用

        """

        trend_total, trends = multi_scale_moving_average(

            x,

            kernel_list=self.decomp_kernels,

            hierarchical=self.hierarchical_trend

        )

        return trend_total, trends



        # # 可学习多尺度的替代

        # trend_total = self.trend_extractor(x)  # [B, S, N]

        # 如果你想也返回每个尺度的趋势，可修改趋势模块返回 list

        # return trend_total



    def pretrain_residual(self, batch_x):

        """

        残差自监督预训练：

        1）使用 RevIN 归一化 -> z

        2）用 moving_average 得到显式结构 trend

        3）残差 r = z - trend

        4）在 r 上做随机掩码，用 FAT encoder 编码，重建被掩码的 r

        5）loss = 只在掩码位置上的 MSE

        """

        device = batch_x.device

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

        pred_r = self.head_pretrain(rebuild_embed)

        pred_r = pred_r.reshape(bs, n_vars, -1).permute(0, 2, 1)

        pred_r = self.revin_layer_encoder(pred_r, 'denorm')

        loss_rb = self.mse(res_ts, pred_r)



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

        bs, seq_len, n_vars = x.shape

        z = x

        z = self.revin_layer_encoder(z, 'norm')

        x = z

        x = x.permute(0, 2, 1)

        if self.configs.forcastMode == "freq":

            x, _ = self.KnowledgeGuide_encoder(x)  # B, N, D

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



'''
EXP代码：
'''

from data_provider.data_factory import data_provider

from exp.exp_basic import Exp_Basic

from utils.tools import EarlyStopping, adjust_learning_rate, transfer_weights, show_series, show_matrix, get_record

from utils.augmentations import masked_data

from utils.metrics import metric

import torch

import torch.nn as nn

from torch import optim

import os

import time

import warnings

import numpy as np

from collections import OrderedDict

from torch.optim import lr_scheduler



from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from models.FAT_resi import moving_average



warnings.filterwarnings('ignore')

class Exp_fresim(Exp_Basic):

    def __init__(self, args):

        super(Exp_fresim, self).__init__(args)

        self.writer = SummaryWriter(f"./outputs/logs/{args.data}/{args.model}/{args.pretrain_mode}")



    def _build_model(self):

        model = self.model_dict[self.args.model].Model(self.args).float()



        if self.args.load_checkpoints:

            if self.args.trs:

                print("train from scratch")

            else:

              print("Loading ckpt: {}".format(self.args.load_checkpoints))

              model = transfer_weights(self.args.load_checkpoints, model, device=self.device, freeze=self.args.freeze)



        # if torch.cuda.device_count() > 1:

        # #     print("Let's use", torch.cuda.device_count(), "GPUs!", self.args.device_ids)

        #       model = nn.DataParallel(model, device_ids=self.args.device_ids)



        # print out the model size

        print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))



        return model



    def _get_data(self, flag):

        data_set, data_loader = data_provider(self.args, flag)

        return data_set, data_loader

    def _select_optimizer(self):

        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate,weight_decay=self.args.weight_decay)

        return model_optim



    def _select_criterion(self):

        criterion = nn.MSELoss()

        return criterion



    def pretrain(self):



        if self.args.task_type == "clf":

            train_loader, vali_loader, test_loader = self._get_classification_loader()

        else:

            # data preparation

            train_data, train_loader = self._get_data(flag='train')

            vali_data, vali_loader = self._get_data(flag='val')

            test_data, test_loader = self._get_data(flag='test')



        path = os.path.join(self.args.pretrain_checkpoints, self.args.data, self.args.exp_name)

        if not os.path.exists(path):

            os.makedirs(path)

        # optimizer

        model_optim = self._select_optimizer()

        model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optim,

                                                                     T_max=self.args.pretrain_epochs)



        # pre-training

        min_vali_loss = None

        for epoch in range(self.args.pretrain_epochs):

            start_time = time.time()

            if self.args.task_type == "clf":

                train_loss, train_cl_loss, train_rb_loss = self.pretrain_one_epoch(train_loader,

                                                                                                 model_optim,

                                                                                                 model_scheduler)

                vali_loss, valid_cl_loss, valid_rb_loss = self.valid_one_epoch(vali_loader)

                test_loss, test_cl_loss, test_rb_loss = self.valid_one_epoch(test_loader)



            else:

                train_loss, train_cl_loss, train_rb_loss = self.pretrain_one_epoch(train_loader, model_optim, model_scheduler)

                vali_loss, valid_cl_loss, valid_rb_loss = self.valid_one_epoch(vali_loader)

                test_loss, test_cl_loss, test_rb_loss = self.valid_one_epoch(test_loader)



            # log and Loss

            end_time = time.time()



            print(

                "Epoch: {0}, Lr: {1:.7f}, Time: {2:.2f}s | Train Loss: {3:.4f}/{4:.4f}/{5:.4f} Val Loss: {6:.4f}/{7:.4f}/{8:.4f} Test Loss: {9:.4f}/{10:.4f}/{11:.4f}\n"

                .format(epoch, model_scheduler.get_lr()[0], end_time - start_time, train_loss, train_cl_loss,

                        train_rb_loss,

                        vali_loss, valid_cl_loss, valid_rb_loss,  test_loss, test_cl_loss, test_rb_loss))



            pretrain_txt = path + "/" + "pretrain_loss.txt"

            pretrain_content = "Epoch: {0}, Lr: {1:.7f}, Time: {2:.2f}s | Train Loss: {3:.4f}/{4:.4f}/{5:.4f} Val Loss: {6:.4f}/{7:.4f}/{8:.4f} Test Loss: {9:.4f}/{10:.4f}/{11:.4f}\n".format(

                epoch, model_scheduler.get_lr()[0], end_time - start_time, train_loss, train_cl_loss,

                train_rb_loss,

                vali_loss, valid_cl_loss, valid_rb_loss,  test_loss, test_cl_loss, test_rb_loss)

            get_record(pretrain_txt, pretrain_content)



            loss_scalar_dict = {

                'train_loss': train_loss,

                'train_cl_loss': train_cl_loss,

                'train_rb_loss': train_rb_loss,

                'vali_loss': vali_loss,

                'valid_cl_loss': valid_cl_loss,

                'valid_rb_loss': valid_rb_loss,

                'test_loss': test_loss,

                'test_cl_loss': test_cl_loss,

                'test_rb_loss': test_rb_loss,

            }



            self.writer.add_scalars(f"/pretrain_loss", loss_scalar_dict, epoch)



            # checkpoint saving

            if not min_vali_loss or vali_loss <= min_vali_loss:

                if epoch == 0:

                    min_vali_loss = vali_loss



                print(

                    "Validation loss decreased ({0:.4f} --> {1:.4f}).  Saving model epoch{2} ...\n".format(min_vali_loss, vali_loss, epoch))

                save_info = "Validation loss decreased ({0:.4f} --> {1:.4f}).  Saving model epoch{2} ...\n".format(min_vali_loss, vali_loss, epoch)

                get_record(pretrain_txt, save_info)



                min_vali_loss = vali_loss

                self.encoder_state_dict = OrderedDict()

                for k, v in self.model.state_dict().items():

                    if 'encoder' in k or 'enc_embedding' in k:

                        if 'module.' in k:

                            k = k.replace('module.', '')  # multi-gpu

                        self.encoder_state_dict[k] = v

                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.encoder_state_dict}

                torch.save(encoder_ckpt, os.path.join(path, f"ckpt_best_ptmode:{self.args.pretrain_mode}.pth"))



            if (epoch + 1) % 10 == 0:

                print("Saving model at epoch {}...".format(epoch + 1))

                get_record(pretrain_txt, "Saving model at epoch {}...\n".format(epoch + 1))



                self.encoder_state_dict = OrderedDict()

                for k, v in self.model.state_dict().items():

                    if 'encoder' in k or 'enc_embedding' in k:

                        if 'module.' in k:

                            k = k.replace('module.', '')

                        self.encoder_state_dict[k] = v

                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.encoder_state_dict}

                torch.save(encoder_ckpt, os.path.join(path, f"ckpt{epoch + 1}.pth"))





    def pretrain_one_epoch(self, train_loader, model_optim, model_scheduler):



        train_loss = []

        train_cl_loss = []

        train_rb_loss = []



        self.model.train()

        for i, (batch_x, batch_y ,*others) in enumerate(train_loader):

            model_optim.zero_grad()

            batch_x = batch_x.float().to(self.device)

            loss, loss_cl, loss_rb,_ ,_ ,_ = self.model(batch_x)



            # backward

            loss.backward()

            model_optim.step()



            # record

            train_loss.append(loss.item())

            train_cl_loss.append(loss_cl.item())

            train_rb_loss.append(loss_rb.item())



        model_scheduler.step()



        train_loss = np.average(train_loss)

        train_cl_loss = np.average(train_cl_loss)

        train_rb_loss = np.average(train_rb_loss)



        return train_loss, train_cl_loss, train_rb_loss



    def valid_one_epoch(self, vali_loader):

        valid_loss = []

        valid_cl_loss = []

        valid_rb_loss = []



        self.model.eval()

        for i, (batch_x, batch_y, *others) in enumerate(vali_loader):



            batch_x = batch_x.float().to(self.device)

            # encoder

            loss, loss_cl, loss_rb, _, _, _ = self.model(batch_x)



            # Record

            valid_loss.append(loss.item())
            valid_cl_loss.append(loss_cl.item())
            valid_rb_loss.append(loss_rb.item())

        vali_loss = np.average(valid_loss)
        valid_cl_loss = np.average(valid_cl_loss)
        valid_rb_loss = np.average(valid_rb_loss)



        self.model.train()

        return vali_loss, valid_cl_loss, valid_rb_loss



    def train(self, setting):



        # data preparation

        train_data, train_loader = self._get_data(flag='train')

        vali_data, vali_loader = self._get_data(flag='val')

        test_data, test_loader = self._get_data(flag='test')



        path = os.path.join(self.args.checkpoints, self.args.data, self.args.exp_name)

        if not os.path.exists(path):

            os.makedirs(path)



        train_steps = len(train_loader)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)



        # Optimizer

        model_optim = self._select_optimizer()

        criterion = self._select_criterion()

        if self.args.freeze == 0:

            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,

                                                steps_per_epoch=train_steps,

                                                pct_start=self.args.pct_start,

                                                epochs=self.args.train_epochs,

                                                max_lr=self.args.learning_rate)



        for epoch in range(self.args.train_epochs):

            iter_count = 0

            train_loss = []



            self.model.train()

            start_time = time.time()

            for i, (batch_x, batch_y, *others) in enumerate(train_loader):

                iter_count += 1

                model_optim.zero_grad()



                # to device

                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)



                # encoder

                outputs = self.model(batch_x)



                f_dim = -1 if self.args.features == 'MS' else 0



                outputs = outputs[:, -self.args.pred_len:, f_dim:]

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)



                trend_y = moving_average(batch_y, kernel_size=self.args.decomp_kernel)

                trend_hat = moving_average(outputs, kernel_size=self.args.decomp_kernel)  # [B, pred_len, N]

                #

                loss_trend = criterion(trend_hat, trend_y)

                #

                # # 施加分布一致性

                # eps = 1e-6

                # mu_y = trend_y.mean(dim=1)  # [B, N]

                # std_y = trend_y.std(dim=1, unbiased=False) + eps  # [B, N]

                # mu_hat = trend_hat.mean(dim=1)  # [B, N]

                # std_hat = trend_hat.std(dim=1, unbiased=False) + eps  # [B, N]

                # loss_mu = criterion(mu_hat, mu_y)

                # loss_std = criterion(std_hat, std_y)

                # loss_var = loss_mu + loss_std



                # loss

                # loss = criterion(outputs, batch_y) + self.args.lambda_trend * loss_trend + self.args.lambda_var * loss_var

                # loss = criterion(outputs, batch_y)

                loss = criterion(outputs, batch_y) + self.args.lambda_trend * loss_trend

                loss.backward()

                model_optim.step()

                # record

                train_loss.append(loss.item())



            train_loss = np.average(train_loss)

            vali_loss = self.vali(vali_loader, criterion)

            test_loss = self.vali(test_loader, criterion)



            end_time = time.time()

            print(

            "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(

                epoch + 1, train_steps, end_time - start_time, train_loss, vali_loss, test_loss))



            finetune_txt = path + '/' + 'finetune_loss.txt'

            finetune_content = "lr_{0}_dp_{1}_{2}_Epoch: {3}, Steps: {4}, Time: {5:.2f}s | Train Loss: {6:.7f} Vali Loss: {7:.7f} Test Loss: {8:.7f} \n".format(

                self.args.learning_rate, self.args.dropout, self.args.pred_len, epoch + 1, train_steps, end_time - start_time, train_loss, vali_loss, test_loss)

            get_record(finetune_txt, finetune_content)



            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:

                print("Early stopping")

                break



            if self.args.freeze == 0:

               adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)



        best_model_path = path + '/' + 'checkpoint.pth'

        self.model.load_state_dict(torch.load(best_model_path))



        self.lr = model_optim.param_groups[0]['lr']



        return self.model



    def vali(self, vali_loader, criterion):

        total_loss = []



        self.model.eval()

        with torch.no_grad():

            for i, (batch_x, batch_y, *others) in enumerate(vali_loader):

                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)



                # encoder

                outputs = self.model(batch_x)



                # loss

                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()

                true = batch_y.detach().cpu()

                loss = criterion(pred, true)



                # record

                total_loss.append(loss)



        total_loss = np.average(total_loss)

        self.model.train()

        return total_loss



    def test(self):

        test_data, test_loader = self._get_data(flag='test')

        preds = []

        trues = []

        folder_path = './outputs/test_results/{}'.format(self.args.data)

        if not os.path.exists(folder_path):

            os.makedirs(folder_path)



        self.model.eval()

        with torch.no_grad():

            for i, (batch_x, batch_y, *others) in enumerate(test_loader):

                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)



                # encoder

                outputs = self.model(batch_x)



                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu().numpy()

                true = batch_y.detach().cpu().numpy()



                preds.append(pred)

                trues.append(true)



        preds = np.array(preds)

        trues = np.array(trues)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])



        mae, mse, rmse, mape, mspe = metric(preds, trues)

        print('{0}->{1}, mse:{2:.3f}, mae:{3:.3f}'.format(self.args.seq_len, self.args.pred_len, mse, mae))



        path = os.path.join(self.args.checkpoints, self.args.data, self.args.exp_name)

        text_txt = path + '/' + 'test_loss.txt'

        text_content = 'lr_{0}_dp_{1}_{2}___{3}->{4}, {5:.5f}, {6:.5f} \n'.format(self.args.learning_rate, self.args.dropout, self.args.exp_name, self.args.seq_len, self.args.pred_len, mse, mae)

        get_record(text_txt, text_content)





        if self.args.trs:

            filename = "{}/score_{}_train_from_scratch.txt".format(folder_path,self.args.model,)

        else:

            filename = "{}/score_{}_{}_{}.txt".format(folder_path,self.args.model,self.args.lm,self.args.pretrain_mode)

        f = open(filename,'a')

        f.write('lr_{0}_dp_{1}_{2}->{3}, {4:.3f}, {5:.3f} \n'.format(self.args.learning_rate, self.args.dropout, self.args.seq_len, self.args.pred_len, mse, mae))

        f.close()



    def show(self, num, epoch, type='valid'):



        # show cases

        if type == 'valid':

            batch_x, batch_y, batch_x_mark, batch_y_mark = self.valid_show

        else:

            batch_x, batch_y, batch_x_mark, batch_y_mark = self.train_show



        # data augumentation

        batch_x_m, batch_x_mark_m, mask = masked_data(batch_x, batch_x_mark, self.args.mask_rate, self.args.lm,

                                                      self.args.positive_nums)

        batch_x_om = torch.cat([batch_x, batch_x_m], 0)



        # masking matrix

        mask = mask.to(self.device)

        mask_o = torch.ones(size=batch_x.shape).to(self.device)

        mask_om = torch.cat([mask_o, mask], 0).to(self.device)



        # to device

        batch_x = batch_x.float().to(self.device)

        batch_x_om = batch_x_om.float().to(self.device)

        batch_x_mark = batch_x_mark.float().to(self.device)



        # Encoder

        with torch.no_grad():

            loss, loss_cl, loss_rb, positives_mask, logits, rebuild_weight_matrix, pred_batch_x = self.model(batch_x_om, batch_x_mark, batch_x, mask=mask_om)



        for i in range(num):



            if i >= batch_x.shape[0]:

                continue



        fig_logits, fig_positive_matrix, fig_rebuild_weight_matrix = show_matrix(logits, positives_mask, rebuild_weight_matrix)

        self.writer.add_figure(f"/{type} show logits_matrix", fig_logits, global_step=epoch)

        self.writer.add_figure(f"/{type} show positive_matrix", fig_positive_matrix, global_step=epoch)

        self.writer.add_figure(f"/{type} show rebuild_weight_matrix", fig_rebuild_weight_matrix, global_step=epoch)