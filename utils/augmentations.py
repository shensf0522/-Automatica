import numpy as np
import torch
import torch.nn.functional as F
import math
from utils.masking import get_mask


def add_noise(time_series, masks, degrees_of_freedom=2, replace = False):
    """
    Adds noise to masked positions in a batch of time-series data.

    Args:
    - time_series (torch.Tensor): The input time-series data of shape (Batch, Variables, Length).
    - masks (torch.Tensor): The binary masks of the same shape as time_series, indicating where to add noise.
    - degrees_of_freedom (int): Degrees of freedom for the Student's t-distribution.

    Returns:
    - torch.Tensor: The time-series data with added noise at masked positions.
    """
    # # Ensure the time series and masks are torch tensors
    # if not isinstance(time_series, torch.Tensor):
    #     time_series = torch.tensor(time_series, dtype=torch.float32)
    # if not isinstance(masks, torch.Tensor):
    #     masks = torch.tensor(masks, dtype=torch.float32)

    # Generate noise from Student's t-distribution
    noise = np.random.standard_t(df=degrees_of_freedom, size=time_series.shape)
    noise = torch.tensor(noise, dtype=time_series.dtype)

    # Apply the noise to the masked positions
    if replace:
        noisy_time_series = time_series * masks + noise * ~masks
    else:
        noisy_time_series = time_series + noise * ~masks

    return noisy_time_series

def masked_data(sample, masking_ratio, lm, positive_nums=1, distribution='geometric',sample_mark=None):
    """Masked time series in time dimension"""

    sample = sample.permute(0, 2, 1)  # [bs x nvars x seq_len]

    sample_repeat = sample.repeat(positive_nums, 1, 1)  # [(bs * positive_nums) x nvars x seq_len]

    mask = noise_mask(sample_repeat, masking_ratio, lm, distribution=distribution)

    x_masked = mask * sample_repeat

    sample_mark_repeat = sample_mark.repeat(positive_nums, 1, 1)

    return x_masked.permute(0, 2, 1), sample_mark_repeat, mask.permute(0, 2, 1)



def masked_and_distort_data(sample, sample_mark, masking_ratio, lm, positive_nums=1, distribution='geometric', scale = 0.1):
    """Masked time series in time dimension"""

    sample = sample.permute(0, 2, 1)  # [bs x nvars x seq_len]

    sample_repeat = sample.repeat(positive_nums, 1, 1)  # [(bs * positive_nums) x nvars x seq_len
    t_noise  = torch.tensor(np.random.standard_t(df = 2, size=sample_repeat.shape), dtype=sample_repeat.dtype) * scale
    mask = noise_mask(sample_repeat, masking_ratio, lm, distribution=distribution)

    # mask_noise = noise_mask(sample_repeat, 0.5, lm, distribution=distribution)
    # x_masked = mask * (sample_repeat + t_noise * mask_noise)

    x_masked = mask * (sample_repeat + t_noise)

    sample_mark_repeat = sample_mark.repeat(positive_nums, 1, 1)

    return x_masked.permute(0, 2, 1), sample_mark_repeat, mask.permute(0, 2, 1)


def mixed_aug_data(sample, sample_mark, masking_ratio, lm, positive_nums=1, distribution='geometric', scale = 0.1):
    """Masked time series in time dimension"""

    sample = sample.permute(0, 2, 1)  # [bs x nvars x seq_len]
    _,_,seq_len = sample.shape
    # blank mask
    x_blank_masked = sample.repeat(int(positive_nums/3), 1, 1)  # [(bs * positive_nums) x nvars x seq_len]
    blank_mask = noise_mask(x_blank_masked, masking_ratio, lm, distribution=distribution, max_length= seq_len)
    x_blank_masked = blank_mask * x_blank_masked

    # student_t distort
    x_t_ditorted = sample.repeat(int(positive_nums/3), 1, 1)
    t_noise = torch.tensor(np.random.standard_t(df=2, size=x_t_ditorted.shape), dtype=x_t_ditorted.dtype) * scale
    x_ditorted_mask = noise_mask(x_t_ditorted, masking_ratio, lm, distribution=distribution,max_length= seq_len)
    x_t_ditorted = x_t_ditorted + t_noise * x_ditorted_mask

    # zoomed_mask
    x_zoomed = sample.repeat(int(positive_nums/3), 1, 1)
    x_zoomed_mask = noise_mask(x_t_ditorted, masking_ratio, lm, distribution=distribution, max_length= seq_len)
    x_zoomed = x_zoomed * x_zoomed_mask + x_zoomed * ~x_zoomed_mask * (torch.rand(size=(x_zoomed.shape[0],x_zoomed.shape[1],1),dtype=x_zoomed.dtype) * 1.5 + 0.5)

    x_masked = torch.cat([x_blank_masked, x_t_ditorted, x_zoomed], 0).to(sample.device)
    ones_mask = torch.ones(size=(int(positive_nums/3) * 2 * sample.shape[0], sample.shape[1], sample.shape[2])).to(sample.device)
    mask = torch.cat([blank_mask, ones_mask], 0).to(sample.device)

    sample_mark_repeat = sample_mark.repeat(positive_nums, 1, 1)

    return x_masked.permute(0, 2, 1), sample_mark_repeat, mask.permute(0, 2, 1)

def distort_masked_area_data(sample, sample_mark, masking_ratio, lm, positive_nums=1, distribution='geometric', scale = 0.1):
    """Masked time series in time dimension"""

    sample = sample.permute(0, 2, 1)  # [bs x nvars x seq_len]

    sample_repeat = sample.repeat(positive_nums, 1, 1)  # [(bs * positive_nums) x nvars x seq_len]
    t_noise  = torch.tensor(np.random.standard_t(df = 2, size=sample_repeat.shape), dtype=sample_repeat.dtype) * scale
    mask = noise_mask(sample_repeat, masking_ratio, lm, distribution=distribution)

    # mask_noise = noise_mask(sample_repeat, 0.5, lm, distribution=distribution)
    # x_masked = mask * (sample_repeat + t_noise * mask_noise)

    x_masked = sample_repeat + mask * t_noise

    sample_mark_repeat = sample_mark.repeat(positive_nums, 1, 1)

    return x_masked.permute(0, 2, 1), sample_mark_repeat, mask.permute(0, 2, 1)

def geom_noise_mask_single(L, lm, masking_ratio, max_length):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked
    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (
            1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    keep_mask[0] = state
    continues_count = 1
    # Start in state 0 with masking_ratio probability
    for i in range(1, L):
        if np.random.rand() < p[state] or continues_count >= int(max_length / 5):
            state = 1 - state
        keep_mask[i] = state
        if keep_mask[i] == keep_mask[i - 1]:
            continues_count += 1
        else:
            continues_count = 1
    return keep_mask


def noise_mask(X, masking_ratio=0.25, lm=3, distribution='geometric', exclude_feats=None, max_length=336):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)
    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        mask = geom_noise_mask_single(X.shape[0] * X.shape[1] * X.shape[2], lm, masking_ratio, max_length = max_length)
        mask = mask.reshape(X.shape[0], X.shape[1], X.shape[2])
        if (mask.sum(axis=-1) == 0).any():
            print("error")
    elif distribution == 'masked_tail':
        mask = np.ones(X.shape, dtype=bool)
        for m in range(X.shape[0]):  # feature dimension
            keep_mask = np.zeros_like(mask[m, :], dtype=bool)
            n = math.ceil(keep_mask.shape[1] * (1 - masking_ratio))
            keep_mask[:, :n] = True
            mask[m, :] = keep_mask  # time dimension
    elif distribution == 'masked_head':
        mask = np.ones(X.shape, dtype=bool)
        for m in range(X.shape[0]):  # feature dimension
            keep_mask = np.zeros_like(mask[m, :], dtype=bool)
            n = math.ceil(keep_mask.shape[1] * masking_ratio)
            keep_mask[:, n:] = True
            mask[m, :] = keep_mask  # time dimension
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                p=(1 - masking_ratio, masking_ratio))
    return torch.tensor(mask)


def one_hot_encoding(X):
    X = [int(x) for x in X]
    n_values = np.max(X) + 1
    b = np.eye(n_values)[X]
    return b


def DataTransform(sample, config):
    """Weak and strong augmentations"""
    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    # weak_aug = permutation(sample, max_segments=config.augmentation.max_seg)
    strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)

    return weak_aug, strong_aug


def remove_frequency(x, pertub_ratio=0.0):
    mask = torch.cuda.FloatTensor(x.shape).uniform_() > pertub_ratio # maskout_ratio are False
    mask = mask.to(x.device)
    return x*mask


def add_frequency(x, pertub_ratio=0.0):

    mask = torch.cuda.FloatTensor(x.shape).uniform_() > (1-pertub_ratio) # only pertub_ratio of all values are True
    mask = mask.to(x.device)
    max_amplitude = x.max()
    random_am = torch.rand(mask.shape)*(max_amplitude*0.1)
    pertub_matrix = mask*random_am
    return x+pertub_matrix


def generate_binomial_mask(B, T, D, p=0.5): # p is the ratio of not zero
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T, D))).to(torch.bool)


def masking(x, keepratio=0.9, mask= 'binomial'):
    global mask_id
    nan_mask = ~x.isnan().any(axis=-1)
    x[~nan_mask] = 0
    # x = self.input_fc(x)  # B x T x Ch

    if mask == 'binomial':
        mask_id = generate_binomial_mask(x.size(0), x.size(1), x.size(2), p=keepratio).to(x.device)
    # elif mask == 'continuous':
    #     mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
    # elif mask == 'all_true':
    #     mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
    # elif mask == 'all_false':
    #     mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
    # elif mask == 'mask_last':
    #     mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
    #     mask[:, -1] = False

    # mask &= nan_mask
    x[~mask_id] = 0
    return x


def augment_positive(sample, masking_ratio, lm, distribution='geometric', scale=0.1, k = 3):
    # raw
    b_n, seq_len = sample.shape
    sample = sample.repeat(int(k), 1, 1)  # [(bs * positive_nums) x nvars x seq_len]

    # # blank mask
    # blank_mask = torch.tensor(
    #     get_mask(sample, distribution, masking_ratio, lm, seq_len),
    #     dtype=sample.dtype,
    #     device=sample.device
    # )
    # x_blank_masked = blank_mask * sample

    # noise
    noise_mask = torch.tensor(
        get_mask(sample, distribution, masking_ratio, lm, seq_len),
        dtype=sample.dtype,
        device=sample.device
    )
    t_noise = torch.tensor(
        np.random.standard_t(df=2, size=sample.shape), dtype=sample.dtype, device=sample.device
    ) * scale
    # x_t_distorted = sample.clone()
    x_t_distorted = sample + t_noise * (1 - noise_mask)
    x_augmented = x_t_distorted
    #
    # # zoom
    # zoom_mask = torch.tensor(
    #     get_mask(sample, distribution, masking_ratio, lm, seq_len),
    #     dtype=sample.dtype,
    #     device=sample.device
    # )
    # scale_factor = torch.rand(sample.shape, dtype=sample.dtype, device=sample.device) * 1.5 + 0.5
    # # x_zoomed = sample.clone()
    # x_zoomed = sample * blank_mask + sample * (1 - blank_mask) * scale_factor
    #
    # # 合并增强后的样本
    # x_augmented = torch.cat([x_blank_masked, x_t_distorted, x_zoomed], dim=0)
    return x_augmented.permute(1,0,2)


def augment_positive_test(sample, masking_ratio, lm, distribution='geometric', scale=0.1, k = 3):
    # 4/12
    b_n, seq_len = sample.shape
    sample_repeat = sample.repeat(int(k/3), 1, 1)  # [(bs * positive_nums) x nvars x seq_len]
    
    # 视图1: blank mask
    blank_mask1 = get_mask(sample_repeat, distribution, masking_ratio, lm, seq_len)
    x_blank_masked = blank_mask1.to(sample_repeat.dtype) * sample_repeat
    
    # 视图2: noise mask (使用独立生成的 mask)
    blank_mask2 = get_mask(sample_repeat, distribution, masking_ratio, lm, seq_len)
    t_noise = torch.tensor(
        np.random.standard_t(df=2, size=sample_repeat.shape), dtype=sample_repeat.dtype, device=sample_repeat.device
    ) * scale
    x_t_distorted = sample_repeat + t_noise * (~blank_mask2).to(sample_repeat.dtype)
    
    # 视图3: zoom mask (使用独立生成的 mask)
    blank_mask3 = get_mask(sample_repeat, distribution, masking_ratio, lm, seq_len)
    scale_factor = torch.rand(sample_repeat.shape, dtype=sample_repeat.dtype, device=sample_repeat.device) * 1.5 + 0.5
    # x_zoomed = sample_repeat * blank_mask3 + sample_repeat * (1 - blank_mask3) * scale_factor
    x_zoomed = (
        sample_repeat * blank_mask3.to(sample_repeat.dtype)
        + sample_repeat * (~blank_mask3).to(sample_repeat.dtype) * scale_factor
    )
    
    # 合并增强后的样本
    x_augmented = torch.cat([x_blank_masked, x_t_distorted, x_zoomed], dim=0)
    return x_augmented.permute(1,0,2)


def augment_positive_test_origin(sample, masking_ratio, lm, distribution='geometric', scale=0.1, k = 3):
    """Original positive augmentation: reuse the same mask for all three views."""
    b_n, seq_len = sample.shape
    sample = sample.repeat(int(k/3), 1, 1)  # [(bs * positive_nums / 3) x nvars x seq_len]
    blank_mask = get_mask(sample, distribution, masking_ratio, lm, seq_len)
    blank_mask = blank_mask.to(sample.dtype)

    x_blank_masked = blank_mask * sample
    t_noise = torch.tensor(
        np.random.standard_t(df=2, size=sample.shape), dtype=sample.dtype, device=sample.device
    ) * scale
    x_t_distorted = sample + t_noise * (1 - blank_mask)

    scale_factor = torch.rand(sample.shape, dtype=sample.dtype, device=sample.device) * 1.5 + 0.5
    x_zoomed = sample * blank_mask + sample * (1 - blank_mask) * scale_factor

    x_augmented = torch.cat([x_blank_masked, x_t_distorted, x_zoomed], dim=0)
    return x_augmented.permute(1,0,2)


def _temporal_smooth(x, kernel_size=3):
    """去高频抖动：小窗口移动平均，保留局部趋势，去除逐点震荡。"""
    b_n, seq_len = x.shape
    x_1d = x.unsqueeze(1)                                       # (b_n, 1, seq_len)
    pad_left = (kernel_size - 1) // 2
    pad_right = kernel_size - 1 - pad_left
    x_padded = F.pad(x_1d, (pad_left, pad_right), mode='replicate')
    smoothed = F.avg_pool1d(x_padded, kernel_size, stride=1)
    return smoothed.squeeze(1)                                   # (b_n, seq_len)


def _median_filter(x, kernel_size=3):
    """去脉冲突变：中值滤波，保留边沿，去除极端尖刺。"""
    b_n, seq_len = x.shape
    pad = kernel_size // 2
    x_padded = F.pad(x.unsqueeze(1), (pad, pad), mode='replicate')  # (b_n, 1, seq_len+2*pad)
    x_unfolded = x_padded.unfold(2, kernel_size, 1)                  # (b_n, 1, seq_len, kernel_size)
    median_vals = x_unfolded.median(dim=-1).values.squeeze(1)        # (b_n, seq_len)
    return median_vals


def _predictability_filter(x):
    """去不可预测成分：用左右邻居的均值估计每个点，保留可预测结构。"""
    padded = F.pad(x.unsqueeze(1), (1, 1), mode='replicate').squeeze(1)  # (b_n, seq_len+2)
    pred_view = (padded[:, :-2] + padded[:, 2:]) / 2.0                  # (b_n, seq_len)
    return pred_view


def _frequency_denoise(x, keep_ratio=0.5):
    """频域去噪：保留能量最强的频率分量，柔性衰减弱频率分量。"""
    fft = torch.fft.rfft(x, dim=-1)
    magnitudes = torch.abs(fft)
    n_freq = fft.shape[-1]
    k = max(1, int(n_freq * keep_ratio))
    sorted_mag, _ = torch.sort(magnitudes, dim=-1, descending=True)
    threshold = sorted_mag[:, k - 1: k]                                  # (b_n, 1)
    # 强频保留原样，弱频按比例衰减（不是直接归零，避免信息丢失过多）
    attenuation = torch.clamp(magnitudes / (threshold + 1e-8), max=1.0)
    fft_filtered = fft * attenuation
    return torch.fft.irfft(fft_filtered, n=x.shape[-1], dim=-1)


def denoise_multi_view(sample, masking_ratio=None, lm=None, distribution=None, scale=None, k=3):
    """
    多视角去噪增强：每个视图是一种不同的信号处理"翻译器"。

    不同于传统的破坏式增强（mask/噪声/拉伸），这里每个视图从不同角度
    去除一种特定类型的噪声，保留有用信息。Encoder通过CL学习三个视图的
    共识表示，从而实现去噪。

    View 1 – 时间平滑：去除高频逐点震荡
    View 2 – 中值滤波：去除脉冲突变/离群点
    View 3 – 可预测滤波：去除不可预测的随机成分

    Args:
        sample: (b_n, seq_len) 输入信号
        masking_ratio, lm, distribution, scale: 占位参数，保持接口兼容
        k: 正样本数量（目前固定为3）

    Returns:
        (b_n, k, seq_len) 去噪视图
    """
    view1 = _temporal_smooth(sample, kernel_size=3)
    view2 = _median_filter(sample, kernel_size=3)
    view3 = _predictability_filter(sample)

    views = torch.stack([view1, view2, view3], dim=1)            # (b_n, 3, seq_len)
    return views[:, :k, :]


def denoise_multi_view_freq(sample, masking_ratio=None, lm=None, distribution=None, scale=None, k=3):
    """
    多视角去噪增强（含频域视角）：

    View 1 – 时间平滑：去除高频逐点震荡
    View 2 – 中值滤波：去除脉冲突变/离群点
    View 3 – 频域滤波：柔性衰减弱能量频率分量

    Args:
        sample: (b_n, seq_len) 输入信号
        masking_ratio, lm, distribution, scale: 占位参数，保持接口兼容
        k: 正样本数量（目前固定为3）

    Returns:
        (b_n, k, seq_len) 去噪视图
    """
    view1 = _temporal_smooth(sample, kernel_size=3)
    view2 = _median_filter(sample, kernel_size=3)
    view3 = _frequency_denoise(sample, keep_ratio=0.5)

    views = torch.stack([view1, view2, view3], dim=1)            # (b_n, 3, seq_len)
    return views[:, :k, :]


def augment_noise_views(sample, masking_ratio, lm, distribution='geometric', scale=0.1, k=3):
    """
    策略F：多类型噪声注入增强，用于去噪预训练。

    保留 masking 机制，但将「置零」和「拉伸」替换为另外两种噪声注入。
    三个视图分别在各自独立的 mask 位置注入不同类型的噪声：
      View 1 – 高斯噪声 (白噪声, i.i.d.)：模拟标准测量噪声
      View 2 – 拉普拉斯噪声 (重尾, 稀疏脉冲型)：模拟偶发的大幅扰动
      View 3 – 有色噪声 (时域相关, 低频漂移型)：模拟缓慢漂移噪声

    模型的任务：从含噪视图中重建原始干净信号，从而学会去噪。

    Args:
        sample: (b_n, seq_len) 输入信号
        masking_ratio: 掩码比例
        lm: 几何分布的平均掩码长度
        distribution: 掩码分布类型
        scale: 噪声幅度缩放因子
        k: 正样本数量（固定为3）

    Returns:
        (b_n, k, seq_len) 含噪视图
    """
    b_n, seq_len = sample.shape
    sample_repeat = sample.repeat(int(k / 3), 1, 1)  # (k/3, b_n, seq_len)

    # --- View 1: 高斯白噪声 ---
    mask1 = get_mask(sample_repeat, distribution, masking_ratio, lm, seq_len)
    gaussian_noise = torch.randn_like(sample_repeat) * scale
    x_gaussian = sample_repeat + gaussian_noise * (~mask1).to(sample_repeat.dtype)

    # --- View 2: 拉普拉斯噪声（重尾脉冲型）---
    mask2 = get_mask(sample_repeat, distribution, masking_ratio, lm, seq_len)
    # 手动生成拉普拉斯分布：sign(U-0.5) * (-b * log(1 - 2|U-0.5|))
    u = torch.rand_like(sample_repeat) - 0.5
    laplace_noise = -scale * torch.sign(u) * torch.log1p(-2.0 * torch.abs(u) + 1e-7)
    x_laplace = sample_repeat + laplace_noise * (~mask2).to(sample_repeat.dtype)

    # --- View 3: 有色噪声（时域相关漂移型）---
    mask3 = get_mask(sample_repeat, distribution, masking_ratio, lm, seq_len)
    # 先生成白噪声，再用移动平均平滑为时域相关噪声
    white_noise = torch.randn_like(sample_repeat) * scale * 2.0
    smooth_kernel = 5
    pad_l = (smooth_kernel - 1) // 2
    pad_r = smooth_kernel - 1 - pad_l
    flat_noise = white_noise.reshape(-1, 1, seq_len)  # (k/3 * b_n, 1, seq_len)
    padded_noise = F.pad(flat_noise, (pad_l, pad_r), mode='replicate')
    colored_noise = F.avg_pool1d(padded_noise, smooth_kernel, stride=1)
    colored_noise = colored_noise.reshape(sample_repeat.shape)
    x_colored = sample_repeat + colored_noise * (~mask3).to(sample_repeat.dtype)

    # 合并增强后的样本
    x_augmented = torch.cat([x_gaussian, x_laplace, x_colored], dim=0)  # (k, b_n, seq_len)
    return x_augmented.permute(1, 0, 2)  # (b_n, k, seq_len)

