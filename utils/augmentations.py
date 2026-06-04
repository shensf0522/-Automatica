import numpy as np
import torch
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
    sample = sample.repeat(int(k/3), 1, 1)  # [(bs * positive_nums) x nvars x seq_len]
    # mask
    blank_mask = torch.tensor(
        get_mask(sample, distribution, masking_ratio, lm, seq_len),
        dtype=sample.dtype,
        device=sample.device
    )
    # blank_mask
    x_blank_masked = blank_mask * sample
    # noise_mask
    t_noise = torch.tensor(
        np.random.standard_t(df=2, size=sample.shape), dtype=sample.dtype, device=sample.device
    ) * scale
    x_t_distorted = sample + t_noise * (1 - blank_mask)
    # zoom_mask
    scale_factor = torch.rand(sample.shape, dtype=sample.dtype, device=sample.device) * 1.5 + 0.5
    x_zoomed = sample * blank_mask + sample * (1 - blank_mask) * scale_factor
    # # 合并增强后的样本
    x_augmented = torch.cat([x_blank_masked, x_t_distorted, x_zoomed], dim=0)
    return x_augmented.permute(1,0,2)
