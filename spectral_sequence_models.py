import torch
import torch.nn as nn
from base_components import Transformer

class SpectralSeqBackbone(nn.Module):
    """
    使用 [CLS] Token 的光谱 Transformer 主干网络
    """
    def __init__(self, in_len, d_model, depth, heads, mlp_dim, dropout, mode='CAF'):
        super().__init__()
        # 1. 将每个波段的单个值嵌入到高维空间
        self.band_embedding = nn.Linear(1, d_model)
        
        # 2. 创建可学习的 [CLS] Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 3. 创建位置编码，长度为 光谱长度 + 1 ([CLS] Token)
        self.pos_embedding = nn.Parameter(torch.randn(1, in_len + 1, d_model))
        
        dim_head = d_model // heads
        self.transformer = Transformer(
            dim=d_model, 
            depth=depth, 
            heads=heads, 
            dim_head=dim_head,
            mlp_head=mlp_dim,
            dropout=dropout,
            num_channel=in_len + 1, # 序列长度变为 L+1
            mode=mode
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: (B, L)
        B, L = x.shape
        
        # (B, L, 1) -> (B, L, d_model)
        x = self.band_embedding(x.unsqueeze(-1))
        
        # 将 [CLS] Token 扩展到 batch size 并拼接到序列最前面
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, L+1, d_model)
        
        # 添加位置编码 (已修正)
        # 动态切片以匹配当前输入的实际长度 L+1
        x = x + self.pos_embedding[:, :(L + 1)]
        
        # 通过 Transformer
        x = self.transformer(x)
        
        # 提取 [CLS] Token 的输出并进行归一化，作为最终的特征表示
        cls_output = self.norm(x[:, 0]) # (B, d_model)
        return cls_output


class GatedWeightedFusion(nn.Module):
    """
    门控加权融合单元，实现 f = g * f_leaf + (1-g) * f_soil
    现在会同时返回融合后@workspace 的特征和门控权重 g
    """
    def __init__(self, in_dim):
        super().__init__()
        # 门控网络，学习权重 g
        self.gating_network = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),
            nn.Sigmoid()
        )

    def forward(self, f_leaf, f_soil):
        # f_leaf, f_soil shape: (B, in_dim)
        x_cat = torch.cat([f_leaf, f_soil], dim=1)
        
        # 计算权重 g
        g = self.gating_network(x_cat) # (B, in_dim)
        
        # 执行加权融合
        fused_features = g * f_leaf + (1 - g) * f_soil
        return fused_features, g


class SpectralSeqFusion(nn.Module):
    """
    符合新架构的序列融合模型
    新增联动：将门控权重 g 作为额外信息输入到回归头
    """
    def __init__(
        self,
        spectral_dim,
        d_model,
        depth,
        heads,
        mlp_dim,
        dropout,
        num_classes=1,
        mode='CAF',
        input_branch='both',
        use_gated_fusion=True,
        use_post_fusion_mlp=True,
        use_gate_weight_feature=True
    ):
        super().__init__()
        if input_branch not in {'both', 'leaf', 'soil'}:
            raise ValueError("input_branch must be one of: 'both', 'leaf', 'soil'")

        self.input_branch = input_branch
        self.use_gated_fusion = use_gated_fusion
        self.use_gate_weight_feature = use_gate_weight_feature

        # 双分支主干网络
        self.leaf_branch = None
        self.soil_branch = None
        if input_branch in {'both', 'leaf'}:
            self.leaf_branch = SpectralSeqBackbone(
                in_len=spectral_dim, d_model=d_model, depth=depth, heads=heads,
                mlp_dim=mlp_dim, dropout=dropout, mode=mode
            )
        if input_branch in {'both', 'soil'}:
            self.soil_branch = SpectralSeqBackbone(
                in_len=spectral_dim, d_model=d_model, depth=depth, heads=heads,
                mlp_dim=mlp_dim, dropout=dropout, mode=mode
            )

        # 门控加权融合模块
        self.fusion_module = GatedWeightedFusion(d_model) if use_gated_fusion and input_branch == 'both' else None

        # 融合后的 MLP 层
        if use_post_fusion_mlp:
            self.post_fusion_mlp = nn.Sequential(
                # 建议增加 LayerNorm 和 Dropout，使其更健壮
                nn.LayerNorm(d_model),
                nn.Linear(d_model, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, d_model)
            )
        else:
            self.post_fusion_mlp = nn.Identity()

        # --- 修改回归头以接收额外信息 ---
        # 新的输入维度 = 融合特征维度(d_model) + 门控权重维度(d_model)
        regression_input_dim = d_model * 2 if use_gate_weight_feature else d_model

        # 更深的 MLP 回归头
        self.regression_head = nn.Sequential(
            nn.LayerNorm(regression_input_dim),
            nn.Linear(regression_input_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model), # 减少维度
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, leaf_spectra, soil_spectra):
        # 1. 根据消融配置提取和融合特征
        if self.input_branch == 'leaf':
            fused = self.leaf_branch(leaf_spectra)
            g = torch.ones_like(fused)
        elif self.input_branch == 'soil':
            fused = self.soil_branch(soil_spectra)
            g = torch.zeros_like(fused)
        else:
            z_leaf = self.leaf_branch(leaf_spectra)      # (B, d_model)
            z_soil = self.soil_branch(soil_spectra)      # (B, d_model)

            if self.use_gated_fusion:
                # 使用门控加权融合，同时获取门控权重 g
                fused, g = self.fusion_module(z_leaf, z_soil)   # fused: (B, d_model), g: (B, d_model)
            else:
                # 关闭门控融合时使用固定均值融合，g 仅作为占位输出用于记录。
                fused = 0.5 * (z_leaf + z_soil)
                g = torch.full_like(fused, 0.5)
        
        # 2. 通过融合后的 MLP
        fused_processed = self.post_fusion_mlp(fused)          # (B, d_model)

        # 3. 可选：将处理后的融合特征与门控权重 g 拼接
        if self.use_gate_weight_feature:
            regression_input = torch.cat([fused_processed, g], dim=1) # (B, d_model * 2)
        else:
            regression_input = fused_processed

        # 4. 通过回归头得到最终预测
        output = self.regression_head(regression_input)
        return output, g
