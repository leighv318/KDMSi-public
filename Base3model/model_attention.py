"""KDMSi-Base3 final architecture and matched manuscript ablations.

Only the final no-SE model, the matched +SE comparison, and the six
single-component ablations reported in the manuscript are exposed here.
Historical architectures were removed to prevent accidental selection.
"""

import torch
import torch.nn as nn

from model_layers import PositionalEncodingWithTime, SELayer


class _Base3CrossAccessV2SrcAblation(nn.Module):
    """Shared scaffold for the final Base3 model and matched ablations."""

    SOURCE_DIM = 8

    def __init__(
        self,
        dp=0.1,
        device=None,
        sirna_ts=4,
        mrna_ts=2,
        use_cross=True,
        use_access=True,
        use_source=True,
        use_knowledge=True,
        use_se=False,
    ):
        super().__init__()
        self.flat = nn.Flatten()
        self.use_cross = use_cross
        self.use_access = use_access
        self.use_source = use_source
        self.use_knowledge = use_knowledge
        self.use_se = use_se

        self.x_map = nn.Linear(640, 32)
        self.encoder_layer_1 = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dropout=dp
        )
        self.pos_embed = PositionalEncodingWithTime(
            dim=32, dropout=dp, max_len=5000, DEVICE=device
        )
        self.y_map = nn.Linear(32 * sirna_ts, 4)
        self.timesteps = sirna_ts

        self.x_map_2 = nn.Linear(640, 16)
        self.encoder_layer_2 = nn.TransformerEncoderLayer(
            d_model=16, nhead=2, dropout=dp
        )
        self.pos_embed_2 = PositionalEncodingWithTime(
            dim=16, dropout=dp, max_len=5000, DEVICE=device
        )
        self.y_map_2 = nn.Linear(16 * mrna_ts, 1)
        self.timesteps_2 = mrna_ts

        if self.use_cross:
            self.mrna_proj = nn.Linear(16, 32)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=32, num_heads=4, dropout=dp
            )
            self.cross_norm = nn.LayerNorm(32)
            self.cross_compress = nn.Linear(32, 4)

        if self.use_source:
            self.source_embed = nn.Embedding(
                num_embeddings=4,
                embedding_dim=self.SOURCE_DIM,
                padding_idx=3,
            )

        dim = 21 * 4 + 59
        if self.use_cross:
            dim += 21 * 4
        if self.use_knowledge:
            dim += 110
        if self.use_access:
            dim += 3
        if self.use_source:
            dim += self.SOURCE_DIM
        self.DNN_in_c = dim

        self.se_block = (
            SELayer(channel=self.DNN_in_c, reduction=1)
            if self.use_se
            else nn.Identity()
        )
        self.bn = nn.BatchNorm1d(self.DNN_in_c)
        self.DNN = nn.Sequential(
            nn.Linear(self.DNN_in_c, 256),
            nn.LeakyReLU(),
            nn.Dropout(dp),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Dropout(dp),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Dropout(dp),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _knowledge_features(inputs):
        return [
            inputs["sirna_gibbs_energy"].to(torch.float32).squeeze(2),
            inputs["pssm_score"].to(torch.float32).squeeze(2),
            inputs["gc_sterch"].to(torch.float32).squeeze(2),
            inputs["sirna_second_percent"].to(torch.float32).squeeze(2),
            inputs["sirna_second_energy"].to(torch.float32).squeeze(2),
            inputs["tri_nt_percent"].to(torch.float32).squeeze(2),
            inputs["di_nt_percent"].to(torch.float32).squeeze(2),
            inputs["single_nt_percent"].to(torch.float32).squeeze(2),
            inputs["gc_content"].to(torch.float32).squeeze(2),
        ]

    def forward(self, inputs):
        x = self.x_map(inputs["rnafm_encode"])
        x = x.permute((1, 0, 2))
        x_steps = []
        for step in range(self.timesteps):
            x = self.pos_embed(x, step)
            x = self.encoder_layer_1(x)
            x_steps.append(x)
        as_encoded = x
        x = torch.cat(x_steps, dim=2).permute((1, 0, 2))
        as_att_out = self.flat(self.y_map(x))

        x = self.x_map_2(inputs["rnafm_encode_mrna"])
        x = x.permute((1, 0, 2))
        x_steps = []
        for step in range(self.timesteps_2):
            x = self.pos_embed_2(x, step)
            x = self.encoder_layer_2(x)
            x_steps.append(x)
        mrna_encoded = x
        x = torch.cat(x_steps, dim=2).permute((1, 0, 2))
        mrna_att_out = self.flat(self.y_map_2(x))

        parts = [as_att_out, mrna_att_out]

        if self.use_cross:
            mrna_proj = self.mrna_proj(mrna_encoded)
            cross_out, _ = self.cross_attn(
                query=as_encoded,
                key=mrna_proj,
                value=mrna_proj,
            )
            cross_out = self.cross_norm(as_encoded + cross_out)
            cross_out = cross_out.permute((1, 0, 2))
            parts.append(self.flat(self.cross_compress(cross_out)))

        if self.use_knowledge:
            parts.extend(self._knowledge_features(inputs))

        if self.use_access:
            parts.append(
                inputs["target_accessibility"].to(torch.float32).squeeze(2)
            )

        if self.use_source:
            source_id = inputs["source_id"].squeeze(-1).squeeze(-1).long()
            parts.append(self.source_embed(source_id))

        inputs_out = torch.cat(parts, dim=1)
        assert inputs_out.shape[1] == self.DNN_in_c, (
            f"Expected {self.DNN_in_c}, got {inputs_out.shape[1]}"
        )

        inputs_out = self.se_block(self.bn(inputs_out))
        return self.DNN(inputs_out).squeeze(1)


class Base3_cross_access_v2_src_no_se(_Base3CrossAccessV2SrcAblation):
    """Final KDMSi-Base3 model."""

    def __init__(self, dp=0.1, device=None, sirna_ts=4, mrna_ts=2):
        super().__init__(
            dp=dp,
            device=device,
            sirna_ts=sirna_ts,
            mrna_ts=mrna_ts,
            use_cross=True,
            use_access=True,
            use_source=True,
            use_knowledge=True,
            use_se=False,
        )


class Base3_cross_access_v2_src(_Base3CrossAccessV2SrcAblation):
    """Matched comparison that adds an SE feature-weighting block."""

    def __init__(self, dp=0.1, device=None, sirna_ts=4, mrna_ts=2):
        super().__init__(
            dp=dp,
            device=device,
            sirna_ts=sirna_ts,
            mrna_ts=mrna_ts,
            use_cross=True,
            use_access=True,
            use_source=True,
            use_knowledge=True,
            use_se=True,
        )


class Base3_cross_access_v2_no_src_no_se(
    _Base3CrossAccessV2SrcAblation
):
    """Ablation without the source embedding."""

    def __init__(self, dp=0.1, device=None, sirna_ts=4, mrna_ts=2):
        super().__init__(
            dp=dp,
            device=device,
            sirna_ts=sirna_ts,
            mrna_ts=mrna_ts,
            use_cross=True,
            use_access=True,
            use_source=False,
            use_knowledge=True,
            use_se=False,
        )


class Base3_cross_noterm_v2_src_no_se(_Base3CrossAccessV2SrcAblation):
    """Ablation without the three target-accessibility descriptors."""

    def __init__(self, dp=0.1, device=None, sirna_ts=4, mrna_ts=2):
        super().__init__(
            dp=dp,
            device=device,
            sirna_ts=sirna_ts,
            mrna_ts=mrna_ts,
            use_cross=True,
            use_access=False,
            use_source=True,
            use_knowledge=True,
            use_se=False,
        )


class Base3_no_cross_access_v2_src_no_se(
    _Base3CrossAccessV2SrcAblation
):
    """Ablation without guide-to-mRNA cross-attention."""

    def __init__(self, dp=0.1, device=None, sirna_ts=4, mrna_ts=2):
        super().__init__(
            dp=dp,
            device=device,
            sirna_ts=sirna_ts,
            mrna_ts=mrna_ts,
            use_cross=False,
            use_access=True,
            use_source=True,
            use_knowledge=True,
            use_se=False,
        )


class Base3_cross_access_v2_src_no_recurrent_no_se(
    _Base3CrossAccessV2SrcAblation
):
    """Single-pass Transformer ablation without recurrent sharing."""

    def __init__(self, dp=0.1, device=None, sirna_ts=1, mrna_ts=1):
        super().__init__(
            dp=dp,
            device=device,
            sirna_ts=1,
            mrna_ts=1,
            use_cross=True,
            use_access=True,
            use_source=True,
            use_knowledge=True,
            use_se=False,
        )


class Base3_cross_access_v2_src_noknow_no_se(
    _Base3CrossAccessV2SrcAblation
):
    """Ablation without the 110 handcrafted siRNA descriptors."""

    def __init__(self, dp=0.1, device=None, sirna_ts=4, mrna_ts=2):
        super().__init__(
            dp=dp,
            device=device,
            sirna_ts=sirna_ts,
            mrna_ts=mrna_ts,
            use_cross=True,
            use_access=True,
            use_source=True,
            use_knowledge=False,
            use_se=False,
        )


class Base3_no_rnafm_v2_src_no_se(nn.Module):
    """Ablation without guide or target-mRNA RNA-FM representations."""

    SOURCE_DIM = 8

    def __init__(self, dp=0.1, device=None, sirna_ts=4, mrna_ts=2):
        super().__init__()
        self.source_embed = nn.Embedding(
            num_embeddings=4,
            embedding_dim=self.SOURCE_DIM,
            padding_idx=3,
        )
        self.DNN_in_c = 110 + 3 + self.SOURCE_DIM
        self.bn = nn.BatchNorm1d(self.DNN_in_c)
        self.DNN = nn.Sequential(
            nn.Linear(self.DNN_in_c, 256),
            nn.LeakyReLU(),
            nn.Dropout(dp),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Dropout(dp),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Dropout(dp),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _knowledge_features(inputs):
        return [
            inputs["sirna_gibbs_energy"].to(torch.float32).squeeze(2),
            inputs["pssm_score"].to(torch.float32).squeeze(2),
            inputs["gc_sterch"].to(torch.float32).squeeze(2),
            inputs["sirna_second_percent"].to(torch.float32).squeeze(2),
            inputs["sirna_second_energy"].to(torch.float32).squeeze(2),
            inputs["tri_nt_percent"].to(torch.float32).squeeze(2),
            inputs["di_nt_percent"].to(torch.float32).squeeze(2),
            inputs["single_nt_percent"].to(torch.float32).squeeze(2),
            inputs["gc_content"].to(torch.float32).squeeze(2),
        ]

    def forward(self, inputs):
        source_id = inputs["source_id"].squeeze(-1).squeeze(-1).long()
        inputs_out = torch.cat(
            self._knowledge_features(inputs)
            + [
                inputs["target_accessibility"].to(torch.float32).squeeze(2),
                self.source_embed(source_id),
            ],
            dim=1,
        )

        assert inputs_out.shape[1] == self.DNN_in_c, (
            f"Expected {self.DNN_in_c}, got {inputs_out.shape[1]}"
        )
        return self.DNN(self.bn(inputs_out)).squeeze(1)


ACTIVE_MODEL_CLASSES = (
    "Base3_cross_access_v2_src_no_se",
    "Base3_cross_access_v2_src",
    "Base3_cross_access_v2_no_src_no_se",
    "Base3_cross_noterm_v2_src_no_se",
    "Base3_no_cross_access_v2_src_no_se",
    "Base3_cross_access_v2_src_no_recurrent_no_se",
    "Base3_cross_access_v2_src_noknow_no_se",
    "Base3_no_rnafm_v2_src_no_se",
)
