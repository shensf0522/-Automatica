import argparse
import torch
import random
import numpy as np
import os

from fsspec.registry import default

# from exp.exp_fresim import Exp_fresim
from exp.exp_fresim_new import Exp_fresim

fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
torch.cuda.manual_seed_all(fix_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='FAT')


# basic config
parser.add_argument('--task_name', type=str, required=True, default='pretrain', help='task name, options:[pretrain, finetune]')
parser.add_argument('--task_type', type=str, default='reg', help='task name, options:[pretrain, finetune]')
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, default='SAT', help='model id')
parser.add_argument('--model', type=str, help='model name')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
parser.add_argument("--pretrain_data",type=str,required=True,default="",help='pretrain datasets')
parser.add_argument('--root_path', type=str, default='./datasets', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./results/checkpoints/', help='location of model fine-tuning checkpoints')
parser.add_argument('--pretrain_checkpoints', type=str, default='./results/pretrain_checkpoints/', help='location of model pre-training checkpoints')
parser.add_argument('--transfer_checkpoints', type=str, default='ckpt_best_ptmode_{}.pth', help='checkpoints we will use to finetune, options:[ckpt_best.pth, ckpt10.pth, ckpt20.pth...]')
parser.add_argument('--transfer_expname',type=str,default='0000_0000',help='record pretrain exp name for load pretrain weight parameters')
parser.add_argument('--load_checkpoints', type=str, default=None, help='location of model checkpoints')
parser.add_argument('--select_channels', type=float, default=1, help='select the rate of channels to train')

# forecasting task
parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length for label')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--num_kernels', type=int, default=3, help='for Inception')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout for pretrained model')
parser.add_argument('--fc_dropout', type=float, default=0, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.1, help='head dropout')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--patch_len', type=int, default=12, help='path length')
parser.add_argument('--stride', type=int, default=12, help='stride')

# optimization
parser.add_argument('--num_workers', type=int, default=5, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--pretrain_epochs', type=int, default=50, help='train epochs')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

# Pre-train
parser.add_argument('--lm', type=int, default=3, help='average masking length')
parser.add_argument('--positive_nums', type=int, default=3, help='masking series numbers')
parser.add_argument('--negative_nums', type=int, default=1, help='masking series numbers')
parser.add_argument('--rbtp', type=int, default=1, help='0: rebuild the embedding of oral series; 1: rebuild oral series')
parser.add_argument('--temperature', type=float, default=0.2, help='temperature')
parser.add_argument('--masked_rule', type=str, default='geometric', help='geometric, random, masked tail, masked head')
# parser.add_argument('--augmode', type=str, default='DistortUnmaskedPosition', help='DistortUnmaskedPosition, DistortMaskArea, Mixed')
parser.add_argument('--pretrain_mode', type=str, default='1', help='0:native_masking, 1:contrast_with_noise')
parser.add_argument('--mask_rate', type=float, default=0.5, help='mask ratio')
parser.add_argument('--distort', type=bool, default=True, help='whether_to_distort')
parser.add_argument('--forcastMode', type=str, default='freq', help='freq or unfreq')
parser.add_argument('--use_fft_sim', type=int, default=0, help='whether to use kb in pre-training stage:0 False')
parser.add_argument('--fft_var', type=int, default=0, help='decide to fft methods, variate and sequences:0 False')
parser.add_argument('--use_kb',type=int,default=0, help='whether to use kb in pre-training stage:0 False')
parser.add_argument('--ft_use_kb',type=int,default=0, help='whether to use kb in pre-training stage:0 False')
parser.add_argument('--use_dfbp',type=int,default=0, help='whether to Decoupled Frequency Balanced Pre-training')
parser.add_argument('--distill_weight',type=float,default=0.5, help='distill weight')
parser.add_argument('--n_knlg', type=int, default=32, help='base frequency nums')
parser.add_argument('--freqdrop_rate',type=float,default=0.5,help='GAP:freq drop rate')
parser.add_argument('--lambda_struct',type=float,default=0.05,help='GAP:lambda_struct')
parser.add_argument('--weight_decay',type=float,default=0.01,help='GAP:lambda_struct')


parser.add_argument('--n_fft', type=int, default=128, help='Number of FFT points for STFT')  # STFT的傅里叶变换点数
parser.add_argument('--hop_length', type=int, default=32, help='Hop length for STFT, typically n_fft // 4')  # 每步移动长度，通常为 n_fft 的四分之一
parser.add_argument('--win_length', type=int, default=128, help='Window length for STFT, usually same as n_fft')  # 窗口长度，通常等于 n_fft
parser.add_argument('--window', type=str, default='hann', help='Type of window function, e.g., hann')  # 窗口函数类型，常用hann窗
parser.add_argument('--stft_top_k', type=int, default=5, help='Number of top elements to select')  # 选择的 top 元素个数

parser.add_argument('--exp_name', type=str, default='test', help='name of exp')
parser.add_argument('--hidden_size', type=int, default=256, help='size of Pai∂ç')
parser.add_argument('--trs', type=int, default = 0, help='1: train_from_scratch, 0 : not train_from_scratch')
parser.add_argument('--freeze', type=int, default = 0, help='1: freeze_weight, 0: not freeze')

# AMD
parser.add_argument('--n_block',type=int, default=1,help='number of block for deep architecture',)
parser.add_argument('--alpha', type=float,default=1.0,help='feature feature dimension',)
parser.add_argument('--patch',type=int,default=12, help='fully-connected history len',)

# pretrain-resduial
parser.add_argument('--use_residual_pretrain',type=bool,default=True,help='whether to use residual pretrain')
parser.add_argument('--decomp_kernel',type=int,default=25,help='the size of kernel to extract trend features')
parser.add_argument('--lambda_trend',type=float,default=0.01, help='the adjust the loss of trend information')
parser.add_argument('--lambda_var',type=float,default=0.0005, help='the adjust the loss of var information')
parser.add_argument(
    '--decomp_kernels',
    nargs='+',              # 一个或多个值
    type=int,               # 将每个值转换为 int
    default=[25],           # 默认值为 [25]
    help='the sizes of kernel(s) to extract trend features, e.g. "--decomp_kernels 37 19 9"'
)
parser.add_argument('--struct_dropout',type=float,default=0.1,help='fuse the struct knowledge to pretrain encoder')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')


# FAT_interPDN_v2 components
parser.add_argument('--lambda_prob', type=float, default=0.05, help='weight for dual-view prob regularizer (Comp A)')
parser.add_argument('--lambda_scale', type=float, default=0.05, help='weight for cross-scale consistency (Comp C)')
parser.add_argument('--alpha_con', type=float, default=0.1, help='weight for time-freq recon consistency (Comp B)')
parser.add_argument('--use_comp_a', type=int, default=1, help='enable Comp A: dual-view prob regularizer; 1=on 0=off')
parser.add_argument('--use_comp_b', type=int, default=1, help='enable Comp B: freq recon branch; 1=on 0=off')
parser.add_argument('--use_comp_c', type=int, default=1, help='enable Comp C: cross-scale consistency; 1=on 0=off')

parser.add_argument('--use_time_index', type=int, default=1, help='whether to use raw calendar time features in trend context gate')
parser.add_argument('--time_feature_dim', type=int, default=6, help='maximum number of raw calendar time features projected into the trend context gate')
parser.add_argument('--memory_size', type=int, default=64, help='trend context memory bank size')
parser.add_argument('--top_k', type=int, default=5, help='top-k trend contexts retrieved from memory')
parser.add_argument('--res_aug_version', type=str, default='denoise', choices=['origin', 'mask_indep', 'new', 'denoise', 'denoise_freq', 'noise_inject'], help='positive augmentation: origin=shared mask, mask_indep=independent masks, new=original positive test, denoise=multi-view denoising, denoise_freq=denoising with freq view, noise_inject=multi-noise injection (strategy F)')
parser.add_argument('--res_use_kb', type=int, default=1, help='whether to use KnowledgeGuide_encoder in FAT_res_trend_gate_new residual pretrain; 1=on 0=off')
parser.add_argument('--res_use_revin', type=int, default=1, help='whether to use RevIN on residual series in FAT_res_trend_gate_new residual pretrain; 1=on 0=off')
parser.add_argument('--res_pretrain_use_time', type=int, default=0, help='whether to pass time features batch_x_mark to FAT_res_trend_gate_new during pretraining; 1=on 0=off')
parser.add_argument('--res_recon_target', type=str, default='raw', choices=['raw', 'consensus', 'mix', 'double', 'noise_penalty'], help='reconstruction target / loss paradigm: raw=original batch_x, consensus=mean of denoised views, mix=Option C (mix target), double=Option D (double loss), noise_penalty=Option E (noise correlation penalty)')
parser.add_argument('--res_mix_alpha', type=float, default=0.5, help='mix alpha for Option C: target = alpha * consensus + (1 - alpha) * raw')
parser.add_argument('--res_double_beta', type=float, default=0.3, help='double beta for Option D: loss = loss_clean + beta * loss_faithful')
parser.add_argument('--res_penalty_gamma', type=float, default=0.1, help='penalty gamma for Option E: loss = loss_raw + gamma * noise_correlation_penalty')
parser.add_argument('--trend_kernels', type=str, default='13,25,49', help='kernel sizes for learnable multi-scale decomposition')
parser.add_argument('--trend_ema_decay', type=float, default=0.999, help='ema decay for memory bank online update')
args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]

print('Args in experiment:')
print(args)
Exp = Exp_fresim

if args.task_name == 'pretrain':
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_sl{}_ll{}_pl{}_dm{}_df{}_nh{}_el{}_dl{}_fc{}_dp{}_hdp{}_ep{}_bs{}_lr{}_lm{}_pn{}_mr{}_tp{}'.format(
            args.task_name,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.d_ff,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.factor,
            args.dropout,
            args.head_dropout,
            args.train_epochs,
            args.batch_size,
            args.learning_rate,
            args.lm,
            args.positive_nums,
            args.mask_rate,
            args.temperature,
            args.decomp_kernel,
            args.lambda_trend
        )

        exp = Exp(args)  # set experiments
        print('>>>>>>>start pre_training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.pretrain()
        torch.cuda.empty_cache()

elif args.task_name == 'finetune':
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_sl{}_ll{}_pl{}_dm{}_df{}_nh{}_el{}_dl{}_fc{}_dp{}_hdp{}_ep{}_bs{}_lr{}_ptmode{}'.format(
            args.task_name,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.d_ff,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.factor,
            args.dropout,
            args.head_dropout,
            args.train_epochs,
            args.batch_size,
            args.learning_rate,
            args.pretrain_mode,
            args.decomp_kernel,
            args.lambda_trend,
            args.lambda_var
        )

        if args.model == 'SimMTM':
            args.load_checkpoints = os.path.join(args.pretrain_checkpoints, args.data, 'ckpt_best.pth')
        else:
            args.load_checkpoints = os.path.join(args.pretrain_checkpoints, args.pretrain_data, args.transfer_expname, args.transfer_checkpoints.format(args.pretrain_mode))

        exp = Exp(args)  # set experiments

        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test()
        torch.cuda.empty_cache()
