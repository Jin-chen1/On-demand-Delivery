"""
Attention-based Policy Network for Delivery Dispatching

使用Transformer架构处理变长订单序列，解决以下问题：
1. 稀疏信号：通过Attention机制自动聚焦有效订单
2. 排列不变性：Self-Attention天然具有排列不变性
3. 订单-骑手匹配：Cross-Attention学习订单与骑手的关联

架构设计：
┌─────────────────────────────────────────────────────────────┐
│                    Attention Policy Network                  │
├─────────────────────────────────────────────────────────────┤
│  订单序列 ──► OrderEncoder ──► Self-Attention ──┐           │
│                                                  │           │
│  骑手序列 ──► CourierEncoder ──► Self-Attention ─┼─► Cross  │
│                                                  │   Attention│
│  全局特征 ──► GlobalEncoder ────────────────────┘           │
│                                                              │
│                    ▼                                         │
│              Policy Head (动作概率)                          │
│              Value Head (状态价值)                           │
└─────────────────────────────────────────────────────────────┘

依赖：pip install stable-baselines3 sb3-contrib torch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Type
import logging

logger = logging.getLogger(__name__)

# 检查SB3是否可用
try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.policies import ActorCriticPolicy
    from gymnasium import spaces
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.warning("Stable-Baselines3不可用，Attention网络将无法使用")


class PositionalEncoding(nn.Module):
    """
    位置编码（可选）
    
    为序列添加位置信息，帮助模型理解顺序
    """
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: (batch, seq_q, d_model)
            key: (batch, seq_k, d_model)
            value: (batch, seq_v, d_model)
            mask: (batch, seq_q, seq_k) or None
        
        Returns:
            (batch, seq_q, d_model)
        """
        batch_size = query.size(0)
        
        # 线性变换并分头
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # 应用mask（屏蔽填充位置）
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, seq_q, seq_k)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        context = torch.matmul(attn_weights, V)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 输出投影
        output = self.W_o(context)
        
        return output


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) or None
        
        Returns:
            (batch, seq_len, d_model)
        """
        # Self-Attention + Residual
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # FFN + Residual
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x


class OrderEncoder(nn.Module):
    """
    订单编码器
    
    将原始订单特征编码为高维表示
    """
    
    def __init__(self, 
                 input_dim: int = 9,  # 每个订单的原始特征维度
                 d_model: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=100, dropout=dropout)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        
        self.d_model = d_model
    
    def forward(self, 
                order_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            order_features: (batch, num_orders, input_dim)
            mask: (batch, num_orders) - True表示有效订单
        
        Returns:
            (batch, num_orders, d_model)
        """
        # 投影到d_model维度
        x = self.input_proj(order_features)
        
        # 位置编码（可选，因为订单本身没有固定顺序）
        # x = self.pos_encoding(x)
        
        # 构建attention mask
        if mask is not None:
            # (batch, num_orders) -> (batch, num_orders, num_orders)
            attn_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        else:
            attn_mask = None
        
        # Transformer编码
        for layer in self.layers:
            x = layer(x, attn_mask)
        
        return x


class CourierEncoder(nn.Module):
    """
    骑手编码器
    
    将原始骑手特征编码为高维表示
    """
    
    def __init__(self,
                 input_dim: int = 4,  # 每个骑手的原始特征维度
                 d_model: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        
        self.d_model = d_model
    
    def forward(self,
                courier_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            courier_features: (batch, num_couriers, input_dim)
            mask: (batch, num_couriers) - True表示有效骑手
        
        Returns:
            (batch, num_couriers, d_model)
        """
        x = self.input_proj(courier_features)
        
        if mask is not None:
            attn_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        else:
            attn_mask = None
        
        for layer in self.layers:
            x = layer(x, attn_mask)
        
        return x


class CrossAttentionLayer(nn.Module):
    """
    交叉注意力层
    
    让订单"查询"骑手，学习订单-骑手的匹配关系
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
                order_repr: torch.Tensor,
                courier_repr: torch.Tensor,
                order_mask: Optional[torch.Tensor] = None,
                courier_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            order_repr: (batch, num_orders, d_model)
            courier_repr: (batch, num_couriers, d_model)
            order_mask: (batch, num_orders)
            courier_mask: (batch, num_couriers)
        
        Returns:
            (batch, num_orders, d_model) - 融合了骑手信息的订单表示
        """
        # 构建cross-attention mask
        if order_mask is not None and courier_mask is not None:
            cross_mask = order_mask.unsqueeze(2) & courier_mask.unsqueeze(1)
        else:
            cross_mask = None
        
        # Cross-Attention: 订单查询骑手
        attn_output = self.cross_attn(order_repr, courier_repr, courier_repr, cross_mask)
        order_repr = self.norm1(order_repr + self.dropout(attn_output))
        
        # FFN
        ffn_output = self.ffn(order_repr)
        order_repr = self.norm2(order_repr + self.dropout(ffn_output))
        
        return order_repr


class AttentionFeaturesExtractor(BaseFeaturesExtractor if SB3_AVAILABLE else nn.Module):
    """
    Attention-based特征提取器
    
    用于Stable-Baselines3的自定义特征提取
    """
    
    def __init__(self, 
                 observation_space: 'spaces.Box',
                 features_dim: int = 128,
                 max_orders: int = 50,
                 max_couriers: int = 50,
                 order_feature_dim: int = 9,
                 courier_feature_dim: int = 4,
                 global_feature_dim: int = 10,
                 d_model: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Args:
            observation_space: 观测空间
            features_dim: 输出特征维度
            max_orders: 最大订单数
            max_couriers: 最大骑手数
            order_feature_dim: 每个订单的特征维度
            courier_feature_dim: 每个骑手的特征维度
            global_feature_dim: 全局特征维度
            d_model: Transformer隐藏维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            dropout: Dropout率
        """
        if SB3_AVAILABLE:
            super().__init__(observation_space, features_dim)
        else:
            super().__init__()
            self._features_dim = features_dim
        
        self.max_orders = max_orders
        self.max_couriers = max_couriers
        self.order_feature_dim = order_feature_dim
        self.courier_feature_dim = courier_feature_dim
        self.global_feature_dim = global_feature_dim
        self.d_model = d_model
        
        # 订单编码器
        self.order_encoder = OrderEncoder(
            input_dim=order_feature_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 骑手编码器
        self.courier_encoder = CourierEncoder(
            input_dim=courier_feature_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 交叉注意力
        self.cross_attention = CrossAttentionLayer(d_model, num_heads, dropout)
        
        # 全局特征编码器
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 聚合层：将变长序列聚合为固定维度
        self.order_aggregator = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        
        self.courier_aggregator = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        
        # 最终融合层
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim)
        )
        
        logger.info(f"AttentionFeaturesExtractor初始化完成")
        logger.info(f"  d_model: {d_model}, num_heads: {num_heads}, num_layers: {num_layers}")
        logger.info(f"  输出特征维度: {features_dim}")
    
    def _parse_observation(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        解析扁平化的观测向量，提取各部分特征
        
        观测向量结构（与StateEncoder一致）：
        - 时间特征: TIME_DIM维
        - 全局特征: GLOBAL_DIM维
        - 订单特征: max_orders * ORDER_FEATURE_DIM维
        - 骑手特征: max_couriers * COURIER_FEATURE_DIM维
        - 热力图: grid_size * grid_size * 2维
        
        注意：维度常量从StateEncoder导入，确保同步
        """
        # 导入StateEncoder的维度常量，确保与编码器同步
        from .state_representation import StateEncoder
        
        batch_size = observations.shape[0]
        
        # 使用StateEncoder的类常量，避免硬编码
        time_dim = StateEncoder.TIME_DIM
        global_dim = StateEncoder.GLOBAL_DIM
        order_dim = self.max_orders * self.order_feature_dim
        courier_dim = self.max_couriers * self.courier_feature_dim
        
        idx = 0
        
        # 时间特征
        time_features = observations[:, idx:idx + time_dim]
        idx += time_dim
        
        # 全局特征
        global_features = observations[:, idx:idx + global_dim]
        idx += global_dim
        
        # 订单特征 -> reshape为 (batch, max_orders, order_feature_dim)
        order_features = observations[:, idx:idx + order_dim]
        order_features = order_features.view(batch_size, self.max_orders, self.order_feature_dim)
        idx += order_dim
        
        # 骑手特征 -> reshape为 (batch, max_couriers, courier_feature_dim)
        courier_features = observations[:, idx:idx + courier_dim]
        courier_features = courier_features.view(batch_size, self.max_couriers, self.courier_feature_dim)
        idx += courier_dim
        
        # 热力图（暂时不使用，可以后续添加CNN处理）
        # heatmap_features = observations[:, idx:]
        
        # 合并时间和全局特征
        combined_global = torch.cat([time_features, global_features], dim=-1)
        
        # 生成mask（基于订单/骑手特征是否全为0）
        # 如果一个订单的所有特征都是0，认为是填充
        order_mask = (order_features.abs().sum(dim=-1) > 1e-6)  # (batch, max_orders)
        courier_mask = (courier_features.abs().sum(dim=-1) > 1e-6)  # (batch, max_couriers)
        
        return {
            'global_features': combined_global,
            'order_features': order_features,
            'courier_features': courier_features,
            'order_mask': order_mask,
            'courier_mask': courier_mask
        }
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            observations: (batch, obs_dim) 扁平化的观测向量
        
        Returns:
            (batch, features_dim) 提取的特征
        """
        # 解析观测
        parsed = self._parse_observation(observations)
        
        # 编码全局特征
        global_repr = self.global_encoder(parsed['global_features'])  # (batch, d_model)
        
        # 编码订单序列
        order_repr = self.order_encoder(
            parsed['order_features'],
            parsed['order_mask']
        )  # (batch, max_orders, d_model)
        
        # 编码骑手序列
        courier_repr = self.courier_encoder(
            parsed['courier_features'],
            parsed['courier_mask']
        )  # (batch, max_couriers, d_model)
        
        # 交叉注意力：订单查询骑手
        order_repr = self.cross_attention(
            order_repr, courier_repr,
            parsed['order_mask'], parsed['courier_mask']
        )  # (batch, max_orders, d_model)
        
        # 聚合订单表示（masked mean pooling）
        order_mask = parsed['order_mask'].unsqueeze(-1).float()  # (batch, max_orders, 1)
        order_sum = (self.order_aggregator(order_repr) * order_mask).sum(dim=1)
        order_count = order_mask.sum(dim=1).clamp(min=1)
        order_pooled = order_sum / order_count  # (batch, d_model)
        
        # 聚合骑手表示
        courier_mask = parsed['courier_mask'].unsqueeze(-1).float()
        courier_sum = (self.courier_aggregator(courier_repr) * courier_mask).sum(dim=1)
        courier_count = courier_mask.sum(dim=1).clamp(min=1)
        courier_pooled = courier_sum / courier_count  # (batch, d_model)
        
        # 融合所有特征
        combined = torch.cat([global_repr, order_pooled, courier_pooled], dim=-1)
        features = self.fusion(combined)  # (batch, features_dim)
        
        return features


def create_attention_policy_kwargs(
    max_orders: int = 50,
    max_couriers: int = 50,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    features_dim: int = 128,
    dropout: float = 0.1
) -> Dict[str, Any]:
    """
    创建使用Attention特征提取器的policy_kwargs
    
    用于传递给PPO/MaskablePPO的policy_kwargs参数
    
    注意：特征维度从StateEncoder的类常量获取，确保同步
    
    Args:
        max_orders: 最大订单数
        max_couriers: 最大骑手数
        d_model: Transformer隐藏维度
        num_heads: 注意力头数
        num_layers: Transformer层数
        features_dim: 输出特征维度
        dropout: Dropout率
    
    Returns:
        policy_kwargs字典
    
    Example:
        >>> policy_kwargs = create_attention_policy_kwargs()
        >>> model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs)
    """
    # 导入StateEncoder的维度常量
    from .state_representation import StateEncoder
    
    return {
        'features_extractor_class': AttentionFeaturesExtractor,
        'features_extractor_kwargs': {
            'features_dim': features_dim,
            'max_orders': max_orders,
            'max_couriers': max_couriers,
            # 使用StateEncoder的类常量，确保维度同步
            'order_feature_dim': StateEncoder.ORDER_FEATURE_DIM,
            'courier_feature_dim': StateEncoder.COURIER_FEATURE_DIM,
            'global_feature_dim': StateEncoder.TIME_DIM + StateEncoder.GLOBAL_DIM,
            'd_model': d_model,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'dropout': dropout
        },
        'net_arch': dict(pi=[128, 64], vf=[128, 64])  # 策略网络和价值网络的隐藏层
    }


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试特征提取器
    batch_size = 4
    obs_dim = 860  # 与StateEncoder一致
    
    # 创建假观测
    observations = torch.randn(batch_size, obs_dim)
    
    # 创建特征提取器
    from gymnasium import spaces
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    
    extractor = AttentionFeaturesExtractor(
        observation_space=obs_space,
        features_dim=128,
        max_orders=50,
        max_couriers=50
    )
    
    # 前向传播
    features = extractor(observations)
    print(f"输入形状: {observations.shape}")
    print(f"输出形状: {features.shape}")
    print("测试通过！")
