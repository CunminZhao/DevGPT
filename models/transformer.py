import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ChannelAttentionHead(nn.Module):
    def __init__(self, num_channels=32, channel_dim=64, hidden_dim=128):
        super().__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.encoder = nn.Sequential(
            nn.Linear(channel_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        expected_dim = self.num_channels * self.channel_dim
        if input_dim != expected_dim:
            raise ValueError(
                f"ChannelAttentionHead expects input_dim={expected_dim}, got {input_dim}"
            )

        x_ch = x.view(batch_size, seq_len, self.num_channels, self.channel_dim)
        global_ch = x_ch.mean(dim=2, keepdim=True).expand(-1, -1, self.num_channels, -1)
        score_input = torch.cat([x_ch, global_ch], dim=-1)
        logits = self.score(self.encoder(score_input)).squeeze(-1)
        attn = torch.softmax(logits, dim=-1)
        x_weighted = x_ch * attn.unsqueeze(-1) * self.num_channels
        return x_weighted.reshape(batch_size, seq_len, input_dim), attn


class CausalBackbone(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        d_model=192,
        num_heads=6,
        num_layers=4,
        ffn_dim=768,
        dropout=0.25,
        max_len=512,
        channel_attention=True,
        num_channels=32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.channel_attention = None
        if channel_attention:
            if input_dim % num_channels != 0:
                raise ValueError(f"input_dim={input_dim} is not divisible by num_channels={num_channels}")
            self.channel_attention = ChannelAttentionHead(
                num_channels=num_channels,
                channel_dim=input_dim // num_channels,
                hidden_dim=max(128, d_model // 2),
            )

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_key_padding_mask=None, return_channel_attn=False):
        seq_len = x.size(1)
        channel_attn = None
        if self.channel_attention is not None:
            x, channel_attn = self.channel_attention(x)

        h = self.input_proj(x)
        h = self.pos_enc(h)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device),
            diagonal=1,
        )
        h = self.transformer(
            h,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        h = self.norm(h)
        if return_channel_attn:
            return h, channel_attn
        return h


class NTPHead(nn.Module):
    def __init__(self, d_model, input_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, input_dim),
        )

    def forward(self, h):
        return self.net(h)


class ClsHead(nn.Module):
    def __init__(self, d_model, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, last_h):
        return self.net(last_h)


class PretrainModel(nn.Module):
    def __init__(self, backbone: CausalBackbone, ntp_head: NTPHead):
        super().__init__()
        self.backbone = backbone
        self.ntp_head = ntp_head

    def forward(self, x, src_key_padding_mask=None, return_channel_attn=False):
        if return_channel_attn:
            h, channel_attn = self.backbone(
                x,
                src_key_padding_mask=src_key_padding_mask,
                return_channel_attn=True,
            )
            return self.ntp_head(h), channel_attn
        h = self.backbone(x, src_key_padding_mask=src_key_padding_mask)
        return self.ntp_head(h)


class FinetuneModel(nn.Module):
    def __init__(self, backbone: CausalBackbone, cls_head: ClsHead, ntp_head: NTPHead = None):
        super().__init__()
        self.backbone = backbone
        self.cls_head = cls_head
        self.ntp_head = ntp_head

    def forward(self, x, src_key_padding_mask=None, lengths=None, return_channel_attn=False):
        batch_size = x.size(0)
        if return_channel_attn:
            h, channel_attn = self.backbone(
                x,
                src_key_padding_mask=src_key_padding_mask,
                return_channel_attn=True,
            )
        else:
            h = self.backbone(x, src_key_padding_mask=src_key_padding_mask)
            channel_attn = None

        if lengths is not None:
            last_idx = (lengths - 1).clamp(min=0)
            idx_gather = last_idx.view(batch_size, 1, 1).expand(-1, 1, h.size(-1))
            last_h = h.gather(1, idx_gather).squeeze(1)
        else:
            last_h = h[:, -1]

        cls_logits = self.cls_head(last_h)
        next_pred = self.ntp_head(h) if self.ntp_head is not None else None
        if return_channel_attn:
            return cls_logits, next_pred, channel_attn
        return cls_logits, next_pred


class RegressionHead(nn.Module):
    def __init__(self, d_model, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, last_h):
        return self.net(last_h).squeeze(-1)


class FinetuneRegressionModel(nn.Module):
    def __init__(self, backbone: CausalBackbone, reg_head: RegressionHead, ntp_head: NTPHead = None):
        super().__init__()
        self.backbone = backbone
        self.reg_head = reg_head
        self.ntp_head = ntp_head

    def forward(self, x, src_key_padding_mask=None, lengths=None, return_channel_attn=False):
        batch_size = x.size(0)
        if return_channel_attn:
            h, channel_attn = self.backbone(
                x,
                src_key_padding_mask=src_key_padding_mask,
                return_channel_attn=True,
            )
        else:
            h = self.backbone(x, src_key_padding_mask=src_key_padding_mask)
            channel_attn = None

        if lengths is not None:
            last_idx = (lengths - 1).clamp(min=0)
            idx_gather = last_idx.view(batch_size, 1, 1).expand(-1, 1, h.size(-1))
            last_h = h.gather(1, idx_gather).squeeze(1)
        else:
            last_h = h[:, -1]

        pred = self.reg_head(last_h)
        next_pred = self.ntp_head(h) if self.ntp_head is not None else None
        if return_channel_attn:
            return pred, next_pred, channel_attn
        return pred, next_pred
