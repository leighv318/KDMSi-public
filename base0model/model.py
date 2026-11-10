import torch
import torch.nn as nn

try:
    from .model_layers import SequenceConv, PositionalEncodingWithTime, SELayer, _PositionAttentionModule
except ImportError:  # Preserve direct `python train.py` execution.
    from model_layers import SequenceConv, PositionalEncodingWithTime, SELayer, _PositionAttentionModule


PAD_LEN = 25
AUX_DIM = 15
CONC_DIM = 2

def build_dnn(DNN_in_c=0, dp=0.2):
    assert DNN_in_c != 0
    return nn.Sequential(
        nn.Linear(DNN_in_c, 256),
        nn.ReLU(),
        nn.Dropout(dp),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(dp),
        nn.Linear(64, 16),
        nn.ReLU(),
        nn.Dropout(dp),
        nn.Linear(16, 1),
        nn.Sigmoid(),
    )


def _conv_flat_dim(L=PAD_LEN, kernels=(1, 3, 6, 9)):
    return sum(max(L - k + 1, 0) for k in kernels)


class _Base0Pad25Flatten(nn.Module):
    """Shared fixed-pad-length Base0 implementation.

    All sequence branches are padded to length 25 in load_data.py and then
    flattened before the DNN. This keeps position-specific information while
    allowing 16--25 nt records to share the same model architecture.
    """
    def __init__(self, device, dp=0.2, as_ts=2, ss_ts=2,
                 input_mode='embedding', use_aux=False, use_knowledge=True,
                 use_attsipred=True, use_conv=True, use_initial_attention=True,
                 use_concentration=True, DNN_in_c=None, cell_embed_dim=8):
        super().__init__()
        self.pad_len = PAD_LEN
        self._as_ts = as_ts
        self._ss_ts = ss_ts
        # Cell-type (source) embedding — learns a bias vector per cell line
        self.cell_embed_dim = cell_embed_dim
        if cell_embed_dim > 0:
            # index 0 = unknown/padding (zero vector); indices 1-N = known cell types
            try:
                from .load_data import NUM_CELL_TYPES
            except ImportError:
                from load_data import NUM_CELL_TYPES
            self.cell_type_embed = nn.Embedding(NUM_CELL_TYPES, cell_embed_dim, padding_idx=0)
        self.input_mode = input_mode
        self.use_aux = use_aux
        self.use_knowledge = use_knowledge
        self.use_attsipred = use_attsipred
        self.use_conv = use_conv
        self.use_concentration = use_concentration
        self.kernels = [1, 3, 6, 9]

        if input_mode == 'onehot':
            input_dim = 28
        elif input_mode == 'embedding':
            input_dim = 5 + 101 + 56
        else:
            raise ValueError(f'unknown input_mode: {input_mode}')
        if use_aux:
            input_dim += AUX_DIM

        as_layers = [nn.Linear(input_dim, 16)]
        ss_layers = [nn.Linear(input_dim, 16)]
        if use_initial_attention:
            as_layers += [nn.LeakyReLU(), _PositionAttentionModule(dim=16)]
            ss_layers += [nn.LeakyReLU(), _PositionAttentionModule(dim=16)]
        self._as_initial_map = nn.Sequential(*as_layers)
        self._ss_initial_map = nn.Sequential(*ss_layers)

        if self.use_conv:
            self._as_conv_model = SequenceConv(kernel_size=self.kernels, kernel_dim=16, pool=False)
            self._ss_conv_model = SequenceConv(kernel_size=self.kernels, kernel_dim=16, pool=False)

        self._as_encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=2, dropout=dp)
        self._as_pos_embed = PositionalEncodingWithTime(dim=16, dropout=dp, max_len=5000, DEVICE=device)
        self._as_map_2 = nn.Linear(16 * self._as_ts, 1)

        self._ss_encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=2, dropout=dp)
        self._ss_pos_embed = PositionalEncodingWithTime(dim=16, dropout=dp, max_len=5000, DEVICE=device)
        self._ss_map_2 = nn.Linear(16 * self._ss_ts, 1)

        self._energy_embed = nn.Embedding(32, 1)

        if DNN_in_c is None:
            DNN_in_c = self._expected_input_dim()
        self.DNN_in_c = int(DNN_in_c)
        self.bn = nn.BatchNorm1d(self.DNN_in_c)
        self.se_block = SELayer(channel=self.DNN_in_c, reduction=8)
        self.DNN = build_dnn(self.DNN_in_c, dp)

    def _expected_input_dim(self):
        dim = 0
        if self.use_conv:
            dim += 2 * _conv_flat_dim(self.pad_len, self.kernels)
        dim += 2 * self.pad_len          # AS/SS attention scalar at each position
        dim += 2 * self.pad_len          # AS/SS energy scalar at each position
        if self.use_knowledge:
            dim += 2 * (125 + 25 + 28 + 1 + 1 + 1)
        if self.use_attsipred:
            dim += 16
        if self.use_concentration:
            dim += CONC_DIM
        if self.cell_embed_dim > 0:
            dim += self.cell_embed_dim
        return dim

    def _select_input(self, inputs, strand):
        if self.input_mode == 'onehot':
            x = inputs[f'cm_{strand}_onehot'].to(torch.float32)
        else:
            x = inputs[f'cm_{strand}_embedding'].to(torch.float32)
        if self.use_aux:
            x = torch.cat([x, inputs[f'{strand}_mod_aux'].to(torch.float32)], dim=-1)
        if x.size(1) != self.pad_len:
            raise RuntimeError(
                f'Expected padded length {self.pad_len}, got {x.size(1)}. '
                'Run with load_data.py pad25 version and train.py --max_len 25.'
            )
        return x

    def _strand_branch(self, x, strand):
        if strand == 'as':
            initial_map = self._as_initial_map
            encoder = self._as_encoder_layer
            pos_embed = self._as_pos_embed
            map_2 = self._as_map_2
            conv_model = getattr(self, '_as_conv_model', None)
            ts = self._as_ts
        else:
            initial_map = self._ss_initial_map
            encoder = self._ss_encoder_layer
            pos_embed = self._ss_pos_embed
            map_2 = self._ss_map_2
            conv_model = getattr(self, '_ss_conv_model', None)
            ts = self._ss_ts

        x = initial_map(x)               # b,25,16
        out_parts = []
        if self.use_conv:
            out_parts.append(conv_model(x))  # b,85 for pad25 and kernels 1/3/6/9

        z = x.permute((1, 0, 2))
        z_all = []
        for t in range(ts):
            z = pos_embed(z, t)
            z = encoder(z)
            z_all.append(z)
        z = torch.cat(z_all, dim=2).permute((1, 0, 2))
        att_out = map_2(z).squeeze(2)    # b,25
        out_parts.append(att_out)
        return out_parts

    def _knowledge_features(self, inputs, strand):
        return [
            inputs[f'{strand}_tri_nt_percent'].squeeze(2),
            inputs[f'{strand}_di_nt_percent'].squeeze(2),
            inputs[f'{strand}_single_nt_percent'].squeeze(2),
            inputs[f'{strand}_gc_content'].squeeze(2),
            inputs[f'{strand}_gc_sterch'].squeeze(2),
            inputs[f'{strand}_pssm_score'].squeeze(2),
        ]

    def forward(self, inputs):
        as_in = self._select_input(inputs, 'as')
        ss_in = self._select_input(inputs, 'ss')

        parts = []
        parts += self._strand_branch(as_in, 'as')
        parts += self._strand_branch(ss_in, 'ss')

        as_energy_out = self._energy_embed(inputs['as_enenrgy_index'].long()).float().squeeze(2)
        ss_energy_out = self._energy_embed(inputs['ss_enenrgy_index'].long()).float().squeeze(2)
        parts += [as_energy_out, ss_energy_out]

        if self.use_knowledge:
            parts += self._knowledge_features(inputs, 'as')
            parts += self._knowledge_features(inputs, 'ss')

        if self.use_attsipred:
            parts.append(inputs['attsipred_index'].squeeze(1))

        if self.use_concentration:
            parts.append(inputs['concentration_feature'].to(torch.float32))

        if self.cell_embed_dim > 0 and 'cell_type_idx' in inputs:
            cell_idx = inputs['cell_type_idx'].squeeze(-1).long()
            parts.append(self.cell_type_embed(cell_idx))  # [B, cell_embed_dim]

        input_out = torch.cat(parts, dim=1)
        if input_out.size(1) != self.DNN_in_c:
            raise RuntimeError(f'DNN input dim mismatch: got {input_out.size(1)}, expected {self.DNN_in_c}')
        out = self.se_block(self.bn(input_out))
        return self.DNN(out).squeeze(1)


class Base0_abla_attsipred(_Base0Pad25Flatten):
    def __init__(self, device, DNN_in_c=None, dp=0.2, as_ts=2, ss_ts=2, cell_embed_dim=8):
        # Same as Base0, but without AttSiPred prior. Includes mod_aux features.
        super().__init__(device=device, dp=dp, as_ts=as_ts, ss_ts=ss_ts,
                         input_mode='embedding', use_aux=True, use_knowledge=True,
                         use_attsipred=False, use_conv=True, use_initial_attention=True,
                         use_concentration=True, DNN_in_c=DNN_in_c, cell_embed_dim=cell_embed_dim)


DEPRECATED_ATTSIPRED_MODELS = r'''
class Base0(_Base0Pad25Flatten):
    def __init__(self, device, DNN_in_c=None, dp=0.2, as_ts=2, ss_ts=2):
        super().__init__(device=device, dp=dp, as_ts=as_ts, ss_ts=ss_ts,
                         input_mode='embedding', use_aux=False, use_knowledge=True,
                         use_attsipred=True, use_conv=True, use_initial_attention=True,
                         use_concentration=True, DNN_in_c=DNN_in_c)


class Base0_abla_knowledge(_Base0Pad25Flatten):
    def __init__(self, device, DNN_in_c=None, dp=0.2, as_ts=2, ss_ts=2):
        super().__init__(device=device, dp=dp, as_ts=as_ts, ss_ts=ss_ts,
                         input_mode='embedding', use_aux=False, use_knowledge=False,
                         use_attsipred=True, use_conv=True, use_initial_attention=True,
                         use_concentration=True, DNN_in_c=DNN_in_c)


class Base0_abla_rdkitgvae(_Base0Pad25Flatten):
    def __init__(self, device, DNN_in_c=None, dp=0.2, as_ts=2, ss_ts=2):
        super().__init__(device=device, dp=dp, as_ts=as_ts, ss_ts=ss_ts,
                         input_mode='onehot', use_aux=False, use_knowledge=True,
                         use_attsipred=True, use_conv=True, use_initial_attention=True,
                         use_concentration=True, DNN_in_c=DNN_in_c)


class Base0_abla_initialmap(_Base0Pad25Flatten):
    def __init__(self, device, DNN_in_c=None, dp=0.2, as_ts=2, ss_ts=2):
        super().__init__(device=device, dp=dp, as_ts=as_ts, ss_ts=ss_ts,
                         input_mode='embedding', use_aux=False, use_knowledge=True,
                         use_attsipred=True, use_conv=True, use_initial_attention=False,
                         use_concentration=True, DNN_in_c=DNN_in_c)


class Base0_abla_conv(_Base0Pad25Flatten):
    def __init__(self, device, DNN_in_c=None, dp=0.2, as_ts=2, ss_ts=2):
        super().__init__(device=device, dp=dp, as_ts=as_ts, ss_ts=ss_ts,
                         input_mode='embedding', use_aux=False, use_knowledge=True,
                         use_attsipred=True, use_conv=False, use_initial_attention=True,
                         use_concentration=True, DNN_in_c=DNN_in_c)
'''


# ══════════════════════════════════════════════════════════════
#  mRNA-enhanced model: cross-attention(guide→mRNA) + 热力学标量
#  追加到 model.py 末尾即可
# ══════════════════════════════════════════════════════════════

MRNA_RNAFM_DIM = 640
MRNA_THERMO_DIM = 5


class CrossAttention(nn.Module):
    """guide (query) 去 attend mRNA (key/value)。
    query: [B, Lq, d] (guide, Lq=25);  kv: [B, Lk, d] (mRNA, Lk=65)
    返回: [B, Lq, d], attn_weights [B, Lq, Lk]
    """
    def __init__(self, dim=16, n_heads=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, kv, key_padding_mask=None):
        out, attn_w = self.attn(query, kv, kv, key_padding_mask=key_padding_mask)
        return self.norm(query + out), attn_w


class Base0_mrna(_Base0Pad25Flatten):
    """在 Base0_abla_attsipred 基础上增加 mRNA 信息。

    - use_mrna_rnafm: guide 用 cross-attention 去 attend 局部 mRNA 的 RNA-FM 表征
    - use_mrna_thermo: 拼接 5 个 mRNA 热力学标量 (accessibility / local folding dG)

    消融时改这两个开关:
      Base0_mrna(use_mrna_rnafm=False, use_mrna_thermo=True)  # 只热力学
      Base0_mrna(use_mrna_rnafm=True,  use_mrna_thermo=False) # 只RNA-FM
      Base0_mrna(use_mrna_rnafm=True,  use_mrna_thermo=True)  # 全加
    """
    def __init__(self, device, DNN_in_c=None, dp=0.2, as_ts=2, ss_ts=2,
                 use_mrna_rnafm=True, use_mrna_thermo=True,
                 use_aux=True, use_knowledge=True, use_concentration=True,
                 cell_embed_dim=8, use_se=True):
        # 先按 Base0_abla_attsipred 的配置初始化父类
        # (DNN_in_c 先占位为 1, 下面重算)
        super().__init__(device=device, dp=dp, as_ts=as_ts, ss_ts=ss_ts,
                         input_mode='embedding', use_aux=use_aux, use_knowledge=use_knowledge,
                         use_attsipred=False, use_conv=True, use_initial_attention=True,
                         use_concentration=use_concentration, DNN_in_c=1,
                         cell_embed_dim=cell_embed_dim)

        self.use_mrna_rnafm = use_mrna_rnafm
        self.use_mrna_thermo = use_mrna_thermo
        self.use_se = use_se

        if use_mrna_rnafm:
            self.mrna_proj = nn.Sequential(
                nn.Linear(MRNA_RNAFM_DIM, 16),
                nn.LeakyReLU(),
            )
            self.cross_attn = CrossAttention(dim=16, n_heads=2, dropout=dp)

        # 重算 DNN 输入维度: 父类基础 + mRNA 增量
        base_dim = self._expected_input_dim()   # use_attsipred=False, 已不含16
        extra = 0
        if use_mrna_rnafm:
            extra += 16              # cross-attention 输出 mean-pool → 16
        if use_mrna_thermo:
            extra += MRNA_THERMO_DIM  # 5
        self.DNN_in_c = int(base_dim + extra)

        # 用新维度重建 bn/se/DNN (覆盖父类里用占位维度建的)
        self.bn = nn.BatchNorm1d(self.DNN_in_c)
        self.se_block = SELayer(channel=self.DNN_in_c, reduction=8) if use_se else nn.Identity()
        self.DNN = build_dnn(self.DNN_in_c, dp)

    def _guide_branch_with_repr(self, as_in):
        """复制父类 guide 分支逻辑, 但额外返回 per-position 表征 as_x 供 cross-attention 用。"""
        as_x = self._as_initial_map(as_in)              # [b,25,16]
        conv_out = self._as_conv_model(as_x)            # [b,85]
        z = as_x.permute((1, 0, 2))
        z_all = []
        for t in range(self._as_ts):
            z = self._as_pos_embed(z, t)
            z = self._as_encoder_layer(z)
            z_all.append(z)
        z = torch.cat(z_all, dim=2).permute((1, 0, 2))
        att_out = self._as_map_2(z).squeeze(2)          # [b,25]
        return [conv_out, att_out], as_x

    def forward(self, inputs):
        as_in = self._select_input(inputs, 'as')
        ss_in = self._select_input(inputs, 'ss')

        parts = []

        # guide 分支 (拿到 as_x 做 cross-attention)
        guide_parts, as_x = self._guide_branch_with_repr(as_in)
        parts += guide_parts

        # ===== cross-attention: guide attend mRNA =====
        if self.use_mrna_rnafm:
            mrna = inputs['mrna_rnafm'].to(torch.float32)   # [b,65,640]
            mrna_proj = self.mrna_proj(mrna)                # [b,65,16]
            pad_mask = (mrna.abs().sum(dim=-1) == 0)        # [b,65] True=padding
            # 若整条都是padding(无mRNA), mask会全True导致NaN, 这里兜底
            all_pad = pad_mask.all(dim=1)                   # [b]
            pad_mask = pad_mask & (~all_pad).unsqueeze(1)   # 全padding的行强制不mask
            cross_out, _ = self.cross_attn(as_x, mrna_proj, key_padding_mask=pad_mask)
            cross_pool = cross_out.mean(dim=1)              # [b,16]
            # 全padding的行输出置0
            cross_pool = cross_pool * (~all_pad).unsqueeze(1).float()
            parts.append(cross_pool)

        # ss 分支 (照旧)
        parts += self._strand_branch(ss_in, 'ss')

        # energy
        as_energy_out = self._energy_embed(inputs['as_enenrgy_index'].long()).float().squeeze(2)
        ss_energy_out = self._energy_embed(inputs['ss_enenrgy_index'].long()).float().squeeze(2)
        parts += [as_energy_out, ss_energy_out]

        # knowledge
        if self.use_knowledge:
            parts += self._knowledge_features(inputs, 'as')
            parts += self._knowledge_features(inputs, 'ss')

        # concentration
        if self.use_concentration:
            parts.append(inputs['concentration_feature'].to(torch.float32))

        # ===== mRNA 热力学标量 =====
        if self.use_mrna_thermo:
            parts.append(inputs['mrna_thermo'].to(torch.float32))   # [b,5]

        # cell-type embedding
        if self.cell_embed_dim > 0 and 'cell_type_idx' in inputs:
            cell_idx = inputs['cell_type_idx'].squeeze(-1).long()
            parts.append(self.cell_type_embed(cell_idx))

        input_out = torch.cat(parts, dim=1)
        if input_out.size(1) != self.DNN_in_c:
            raise RuntimeError(f'DNN input dim mismatch: got {input_out.size(1)}, expected {self.DNN_in_c}')
        out = self.se_block(self.bn(input_out))
        return self.DNN(out).squeeze(1)


# 消融用的子类 (方便命令行 --model 直接指定)
class Base0_mrna_thermo_only(Base0_mrna):
    def __init__(self, device, DNN_in_c=None, dp=0.2, as_ts=2, ss_ts=2, cell_embed_dim=8):
        super().__init__(device, DNN_in_c, dp, as_ts, ss_ts,
                         use_mrna_rnafm=False, use_mrna_thermo=True,
                         cell_embed_dim=cell_embed_dim)


class Base0_mrna_rnafm_only(Base0_mrna):
    def __init__(self, device, DNN_in_c=None, dp=0.2, as_ts=2, ss_ts=2, cell_embed_dim=8):
        super().__init__(device, DNN_in_c, dp, as_ts, ss_ts,
                         use_mrna_rnafm=True, use_mrna_thermo=False,
                         cell_embed_dim=cell_embed_dim)


class Base0_mrna_full(Base0_mrna):
    def __init__(self, device, DNN_in_c=None, dp=0.2, as_ts=2, ss_ts=2, cell_embed_dim=8):
        super().__init__(device, DNN_in_c, dp, as_ts, ss_ts,
                         use_mrna_rnafm=True, use_mrna_thermo=True,
                         cell_embed_dim=cell_embed_dim)


class Base0_mrna_full_noknow(Base0_mrna):
    """Final Base0 mRNA-full model without handcrafted sequence features."""
    def __init__(self, device, DNN_in_c=None, dp=0.2, as_ts=2, ss_ts=2, cell_embed_dim=8):
        super().__init__(device, DNN_in_c, dp, as_ts, ss_ts,
                         use_mrna_rnafm=True, use_mrna_thermo=True,
                         use_knowledge=False, cell_embed_dim=cell_embed_dim)


class Base0_mrna_full_no_cell(Base0_mrna):
    """Final Base0 mRNA-full model without cell-type embedding."""
    def __init__(self, device, DNN_in_c=None, dp=0.2, as_ts=2, ss_ts=2):
        super().__init__(device, DNN_in_c, dp, as_ts, ss_ts,
                         use_mrna_rnafm=True, use_mrna_thermo=True,
                         cell_embed_dim=0)


class Base0_mrna_full_no_concentration(Base0_mrna):
    """Final Base0 mRNA-full model without concentration features."""
    def __init__(self, device, DNN_in_c=None, dp=0.2, as_ts=2, ss_ts=2, cell_embed_dim=8):
        super().__init__(device, DNN_in_c, dp, as_ts, ss_ts,
                         use_mrna_rnafm=True, use_mrna_thermo=True,
                         use_concentration=False, cell_embed_dim=cell_embed_dim)


class Base0_mrna_full_no_se(Base0_mrna):
    """Final Base0 mRNA-full model without SE feature recalibration."""
    def __init__(self, device, DNN_in_c=None, dp=0.2, as_ts=2, ss_ts=2, cell_embed_dim=8):
        super().__init__(device, DNN_in_c, dp, as_ts, ss_ts,
                         use_mrna_rnafm=True, use_mrna_thermo=True,
                         use_se=False, cell_embed_dim=cell_embed_dim)


class Base0_mrna_full_no_mod_aux(Base0_mrna):
    """Final Base0 mRNA-full model without per-position modification auxiliary features."""
    def __init__(self, device, DNN_in_c=None, dp=0.2, as_ts=2, ss_ts=2, cell_embed_dim=8):
        super().__init__(device, DNN_in_c, dp, as_ts, ss_ts,
                         use_mrna_rnafm=True, use_mrna_thermo=True,
                         use_aux=False, cell_embed_dim=cell_embed_dim)

class Base0_abla_attsipred_noknow(_Base0Pad25Flatten):
    """与 Base0_abla_attsipred 完全相同, 仅去掉 knowledge features (k-mer/GC/PSSM)"""
    def __init__(self, device, DNN_in_c=None, dp=0.2, as_ts=2, ss_ts=2):
        super().__init__(device=device, dp=dp, as_ts=as_ts, ss_ts=ss_ts,
                         input_mode='embedding', use_aux=True, use_knowledge=False,
                         use_attsipred=False, use_conv=True, use_initial_attention=True,
                         use_concentration=True, DNN_in_c=DNN_in_c)
