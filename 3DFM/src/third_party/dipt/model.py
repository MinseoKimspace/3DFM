"""
Diffusion Point Transformer (DiPT) v1m1 base class.

Author: Matteo Bastico (matteo.bastico@gmail.com)
Please cite our work if the code is helpful to you.
"""
from functools import partial
import math
import torch
import torch.nn as nn

from .misc import offset2bincount
from .modules import PointModule, PointSequential
from .pdnorm import PDNorm
from .point_transformer import Block, Embedding
from .structure import Point


# From https://github.com/DiT-3D/DiT-3D/blob/main/models/dit3d.py#L53
class TimestepEmbedding(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(
            self,
            hidden_size,
            frequency_embedding_size=256,
            act_layer=nn.SiLU
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            act_layer(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        # print('token drop drop_ids:', drop_ids, drop_ids.shape)
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class DiffusionBlock(Block):
    def __init__(
        self,
        channels,
        act_layer=nn.SiLU,
        **kwargs
    ):
        super().__init__(channels=channels, **kwargs)
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(channels, 6 * channels, bias=True)
        )

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale) + shift

    def forward(self, point: Point):
        modulation = self.adaLN_modulation(point.condition)
        modulation = torch.repeat_interleave(modulation, offset2bincount(point.offset), dim=0)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=1)
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point.feat = self.modulate(point.feat, shift_msa, scale_msa)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + gate_msa * point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point.feat = self.modulate(point.feat, shift_mlp, scale_mlp)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + gate_mlp * point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class DiffusionPointTransformer(PointModule):
    def __init__(
        self,
        in_channels=6,
        num_classes=3,
        cls_drop=0.1,
        order=("z", "z-trans"),
        depth=12,
        channels=768,
        num_head=12,
        patch_size=48,
        mlp_ratio=4,
        frequency_embedding_size=256,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        super().__init__()
        self.patch_size = [patch_size] * depth if isinstance(patch_size, int) else patch_size
        self.num_head = [num_head] * depth if isinstance(num_head, int) else num_head
        self.channels = [channels] * depth if isinstance(channels, int) else channels
        assert len(self.patch_size) == depth
        assert len(self.num_head) == depth
        assert len(self.channels) == depth
        self.order = [order] if isinstance(order, str) else order
        self.shuffle_orders = shuffle_orders
        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        self.act_layer = nn.GELU

        self.timestep_embedding = TimestepEmbedding(
            hidden_size=channels,
            frequency_embedding_size=frequency_embedding_size,
            act_layer=self.act_layer
        )

        self.cls_embedding = LabelEmbedder(
            num_classes=num_classes,
            hidden_size=channels,
            dropout_prob=cls_drop,
        )

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=channels,
            norm_layer=bn_layer,
            act_layer=self.act_layer,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, depth)
        ]
        self.enc = PointSequential()
        for d in range(depth):
            enc_drop_path_ = enc_drop_path[d]
            self.enc.add(
                DiffusionBlock(
                    channels=self.channels[d],
                    num_heads=self.num_head[d],
                    patch_size=self.patch_size[d],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path=enc_drop_path_,
                    norm_layer=ln_layer,
                    act_layer=self.act_layer,
                    pre_norm=pre_norm,
                    # order_index=random.randint(0, len(self.order) - 1),
                    order_index=d % len(self.order),
                    cpe_indice_key=f"block{d + 1}",
                    enable_rpe=enable_rpe,
                    enable_flash=enable_flash,
                    upcast_attention=upcast_attention,
                    upcast_softmax=upcast_softmax,
                ),
                name=f"block{d + 1}",
            )

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        timestep_embedding = self.timestep_embedding(point.timesteps)
        cls_embedding = self.cls_embedding(point.cls_token, train=self.training)
        point.condition = timestep_embedding + cls_embedding

        point = self.embedding(point)
        point = self.enc(point)
        return point
