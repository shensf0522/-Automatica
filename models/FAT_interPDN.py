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
from scipy.stats import norm
import numpy as np

# --- 在 FAT.py 中添加概率生成头类 ---
class ProbabilisticHead(nn.Module):
    def __init__(self, d_model, seq_len, n_points=25, boundary=4.0):
        super().__init__()
        self.T = seq_len
        self.S = n_points
        self.B = boundary
        
        # 双分支线性映射：将编码器输出映射到概率空间
        self.fc1 = nn.Linear(d_model * seq_len, self.T * self.S)
        self.fc2 = nn.Linear(d_model * seq_len, self.T * self.S)
        
        # 初始化交错支持集 (Interleaved Support Sets)
        self.register_buffer('sp1', self._get_support_set(n_points, boundary, interleaved=False))
        self.register_buffer('sp2', self._get_support_set(n_points, boundary, interleaved=True))

    # 断点数量不匹配的问题（应该是25个断点），然而实际是26个
    # def _get_support_set(self, n_points, boundary, interleaved=False):
    #     # 模拟 interPDN 的非均匀等概率划分逻辑
    #     # 在 [-B, B] 之间根据标准正态分布划分
    #     points = norm.ppf(np.linspace(0.01, 0.99, n_points))
    #     # 缩放到 [-B, B]
    #     points = (points - points.min()) / (points.max() - points.min()) * 2 * boundary - boundary
    #     points = torch.tensor(points, dtype=torch.float32)
        
    #     if interleaved:
    #         # 交错处理：取中点并补齐边界
    #         mid_points = (points[:-1] + points[1:]) / 2
    #         points = torch.cat([torch.tensor([-boundary]), mid_points, torch.tensor([boundary])])
    #     return points

    # def forward(self, x):
    #     # x: (bs*n_vars, seq_len, d_model)
    #     x_flat = x.reshape(x.shape[0], -1)
        
    #     # 分支 1 预测
    #     out1 = self.fc1(x_flat).reshape(-1, self.T, self.S)
    #     p1 = F.softmax(out1, dim=-1)
    #     x_sp1 = torch.matmul(p1, self.sp1) # (bs*n_vars, T)
        
    #     # 分支 2 预测
    #     out2 = self.fc2(x_flat).reshape(-1, self.T, self.S)
    #     p2 = F.softmax(out2, dim=-1)
    #     x_sp2 = torch.matmul(p2, self.sp2) # (bs*n_vars, T)
        
    #     # 基于置信度 (Max Probability) 的加权融合
    #     e1 = torch.max(p1, dim=-1)[0]
    #     e2 = torch.max(p2, dim=-1)[0]
    #     w = (e1 / (e1 + e2)).unsqueeze(-1)
        
    #     x_fused = w * x_sp1 + (1 - w) * x_sp2
    #     return x_fused, x_sp1, x_sp2

    def _get_support_set(self, n_points, boundary, interleaved=False):
        # 计算基于标准正态分布的等概率断点
        p_start = norm.cdf(-boundary)
        p_end = norm.cdf(boundary)
        
        # 想要 n_points 个区间中点，需要 n_points + 1 个断点
        p_breakpoints = np.linspace(p_start, p_end, n_points + 1)
        breakpoints = norm.ppf(p_breakpoints)
        
        # 取相邻断点的均值，得到严格的 n_points 个支持点 (sp1)
        sp1 = (breakpoints[:-1] + breakpoints[1:]) / 2
        
        if not interleaved:
            return torch.tensor(sp1, dtype=torch.float32)
        else:
            # 交错处理：取 sp1 的中点，此时长度变为 n_points - 1
            mid_points = (sp1[:-1] + sp1[1:]) / 2
            # 严格遵照论文，只补充一个边界点，使长度恢复为 n_points
            boundary_pt = np.array([-boundary])
            sp2 = np.concatenate([boundary_pt, mid_points])
            return torch.tensor(sp2, dtype=torch.float32)

    def forward(self, x):
        # x: (bs*n_vars, seq_len, d_model)
        x_flat = x.reshape(x.shape[0], -1)
        
        # 分支 1 预测
        out1 = self.fc1(x_flat).reshape(-1, self.T, self.S)
        p1 = F.softmax(out1, dim=-1)
        x_sp1 = torch.matmul(p1, self.sp1) # (bs*n_vars, T)
        
        # 分支 2 预测
        out2 = self.fc2(x_flat).reshape(-1, self.T, self.S)
        p2 = F.softmax(out2, dim=-1)
        x_sp2 = torch.matmul(p2, self.sp2) # (bs*n_vars, T)
        
        # 基于置信度 (Max Probability) 的加权融合
        e1 = torch.max(p1, dim=-1)[0]
        e2 = torch.max(p2, dim=-1)[0]
        w = (e1 / (e1 + e2))
        
        x_fused = w * x_sp1 + (1 - w) * x_sp2
        return x_fused, x_sp1, x_sp2

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
        self.task_name = configs.task_name
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.scale = 0.02
        self.revin_layer_encoder = RevIN(configs.enc_in, affine=True, subtract_last=False)

        self.embed_size = self.seq_len
        self.hidden_size = configs.hidden_size

        self.KnowledgeGuide_encoder = FreNormLayer_KB(configs.n_knlg, configs.seq_len)
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

        # 添加概率预测头
        self.prob_head = ProbabilisticHead(configs.d_model, configs.seq_len)

    def pretrain(self, batch_x):
        bs, seq_len, n_vars = batch_x.shape
        z = batch_x  # (b, s, n)  n(=n'*f)*n(=n'*f)
        z = self.revin_layer_encoder(z, 'norm')
        # augmentation
        x_normed = z
        x_normed = x_normed.permute(0, 2, 1)  
        sim_matrix = FFT_sim(x_normed)  # (b*n, b*n)   不同样本，不同特征之间的相似性
        x_normed = x_normed.reshape(-1, seq_len)  # (b*n, s)
        negative_index = torch.topk(sim_matrix, k=self.configs.negative_nums, dim=1).indices  # 记录第i个样本的第j个特征和第非i个样本中最相似的特征。
        # Knowledge_guide
        # torch.save(x_normed, "./ecl_336_x_normed_permute.pt")
        x_reformed, _ = self.KnowledgeGuide_encoder(x_normed)  # B, N, D  # 过滤噪声
        # torch.save(x_reformed, "./ecl_336_x_reformed.pt")
        x_positives = augment_positive_test(x_reformed, self.configs.mask_rate, self.configs.lm, k=self.configs.positive_nums)
        x_positives = x_positives.reshape(-1, seq_len)
        x_all = torch.cat([x_normed, x_positives], dim=0)

        # Encoder
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

        # 调整这里的重建头来适配概率头
        rebuild_embed = rebuild_embed.reshape(bs * n_vars, seq_len, -1)
        # 2. 调用新的概率头
        pred_x_exp, x_sp1, x_sp2 = self.prob_head(rebuild_embed)
        
        # 3. 将概率期望预测值塑形回原有的 (bs, seq_len, n_vars) 结构以便反归一化
        pred_x_exp = pred_x_exp.reshape(bs, n_vars, -1).permute(0, 2, 1)
        
        # 4. 反归一化
        pred_x = self.revin_layer_encoder(pred_x_exp, 'denorm')
        
        # 5. 计算损失
        # 5.1 期望重构损失 (用原来的 MSE 就可以)
        loss_rb_exp = self.mse(batch_x, pred_x)
        
        # 5.2 双分支一致性损失 (Consistency Loss)
        loss_consistency = self.mse(x_sp1, x_sp2)
        
        # 5.3 组合重构损失 (加入一致性约束，alpha 暂时设为 0.1)
        alpha = 0.1
        loss_rb = loss_rb_exp + alpha * loss_consistency

        # 使用现有的 AutomaticWeightedLoss 融合 对比学习损失 和 组合后的重构损失
        loss = self.awl(loss_cl, loss_rb)
        
        # 保持原本的返回格式，以免你的 exp_main.py 里的解包报错
        return loss, loss_cl, loss_rb, None, None, None
        # rebuild_embed = rebuild_embed.reshape(bs, n_vars, seq_len, -1)
        # pred_x = self.head_pretrain(rebuild_embed)
        # pred_x = pred_x.reshape(bs, n_vars, -1).permute(0, 2, 1)
        # pred_x = self.revin_layer_encoder(pred_x, 'denorm')
        # loss_rb = self.mse(batch_x, pred_x)

        # loss = self.awl(loss_cl, loss_rb)
        # return loss, loss_cl, loss_rb, None, None, None

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
        x, _ = self.encoder(x) # (bs*n, seq_len, d_model)
        torch.save(x, 'repre_FAT.pt')

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