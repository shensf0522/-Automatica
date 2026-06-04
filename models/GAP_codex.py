from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.common import DDI, MDM
from utils.losses import AutomaticWeightedLoss
from utils.tools import RevIN

class PatternExtractor(nn.Module):
    """Extract deterministic patterns using MDM and DDI blocks."""

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        enc_in: int,
        *,
        ddi_patch: int = 12,
        ddi_alpha: float = 0.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        input_shape = (seq_len, enc_in)
        self.mdm = MDM(input_shape, k=3, c=2, layernorm=True)
        self.ddi = DDI(
            input_shape,
            dropout=dropout,
            patch=ddi_patch,
            alpha=ddi_alpha,
            layernorm=True,
        )
        self.proj = nn.Linear(seq_len, d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return deterministic patterns and intermediate features.

        Args:
            x: Input sequence shaped as ``(batch, seq_len, enc_in)``.

        Returns:
            pattern: Aggregated deterministic pattern in the original domain.
            features: Latent representation used by the downstream modules.
        """

        # [batch, feature, seq_len]
        x_feature_first = x.transpose(1, 2)
        mdm_out = self.mdm(x_feature_first)
        ddi_out = self.ddi(mdm_out)
        pattern = ddi_out.transpose(1, 2)
        latent = self.proj(pattern.transpose(1, 2))  # -> [batch, enc_in, d_model]
        latent = latent.transpose(1, 2)  # [batch, d_model, enc_in]
        return pattern, latent


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ResidualEncoder(nn.Module):
    """Encode residual sequences with a masked reconstruction objective."""

    def __init__(self, seq_len: int, enc_in: int, d_model: int, n_heads: int, e_layers: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(enc_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=seq_len)
        self.output_proj = nn.Linear(d_model, enc_in)

    def forward(self, residual: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedded = self.input_proj(residual)
        embedded = self.pos_encoding(embedded)
        encoded = self.encoder(embedded, src_key_padding_mask=mask) # 确保模型在计算注意力权重时，不会关注（或说，不会从）那些因为批处理而添加的 填充（padding） token 中获取信息。
        return self.output_proj(encoded)


class ResidualForecaster(nn.Module):
    """Forecast future residuals given encoded context."""

    def __init__(self, enc_in: int, hidden_dim: int, pred_len: int) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.gru = nn.GRU(input_size=enc_in, hidden_size=hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, enc_in)

    def forward(self, residual_context: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(residual_context)
        last_hidden = output[:, -1]
        repeated = last_hidden.unsqueeze(1).repeat(1, self.pred_len, 1)
        return self.readout(repeated)


@dataclass
class PretrainConfig:
    seq_len: int
    pred_len: int
    enc_in: int
    d_model: int = 128
    n_heads: int = 4
    e_layers: int = 4
    mask_ratio: float = 0.2


class Model(nn.Module):
    """End-to-end module that orchestrates pattern removal and residual modelling."""

    def __init__(self, config: PretrainConfig) -> None:
        super().__init__()
        self.config = config
        self.revin_layer_encoder = RevIN(config.enc_in, affine=True, subtract_last=False)
        self.pattern_extractor = PatternExtractor(
            seq_len=config.seq_len,
            d_model=config.d_model,
            enc_in=config.enc_in,
        )
        self.residual_encoder = ResidualEncoder(
            seq_len=config.seq_len,
            enc_in=config.enc_in,
            d_model=config.d_model,
            n_heads=config.n_heads,
            e_layers=config.e_layers,
        )
        self.residual_forecaster = ResidualForecaster(
            enc_in=config.enc_in,
            hidden_dim=config.d_model,
            pred_len=config.pred_len,
        )
        self.reconstruction_loss = nn.MSELoss(reduction="none")
        self.awl = AutomaticWeightedLoss(2)

    def generate_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        mask = torch.rand(batch_size, self.config.seq_len, device=device) < self.config.mask_rate
        return mask

    def mask_residual(self, residual: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = mask.unsqueeze(-1)
        return residual.masked_fill(mask_expanded, 0.0)

    def pretrain(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 添加RevIN的变化
        x_n = self.revin_layer_encoder(x, 'norm')
        x_bcn = x_n.permute(0, 2, 1).contiguous()
        s_mdm = self.mdm(x_bcn)
        s_ddi = self.ddi(x_bcn)

        s_mix = self.gate(s_mdm, s_ddi)

        pattern, _ = self.pattern_extractor(x)
        residual = x - pattern
        masked_residual = self.mask_residual(residual, mask)
        reconstruction = self.residual_encoder(masked_residual, mask)
        residual_loss = self.reconstruction_loss(reconstruction, residual)
        residual_loss = (residual_loss * mask.unsqueeze(-1).float()).sum() / mask.sum().clamp_min(1.0)
        return residual_loss, _,

    def forecast(self, history: torch.Tensor) -> torch.Tensor:
        pattern, _ = self.pattern_extractor(history)
        residual = history - pattern
        residual_future = self.residual_forecaster(residual)
        pattern_tail = pattern[:, -1:].repeat(1, self.config.pred_len, 1)
        return residual_future + pattern_tai
    def forward(self, x: torch.Tensor, *, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.pretrain(x, mask=mask)
