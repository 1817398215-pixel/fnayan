# data_utils.py
# 数据处理工具模块

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class FusionDataset(Dataset):
    """融合数据集类 - 仅包含叶片光谱和土壤光谱"""
    def __init__(self, leaf_spectra, soil_spectra, chlorophyll):
        self.leaf_spectra = torch.FloatTensor(leaf_spectra)
        self.soil_spectra = torch.FloatTensor(soil_spectra)
        self.chlorophyll = torch.FloatTensor(chlorophyll).unsqueeze(1)

    def __len__(self):
        return len(self.chlorophyll)

    def __getitem__(self, idx):
        return self.leaf_spectra[idx], self.soil_spectra[idx], self.chlorophyll[idx]


def load_excel_data(data_path):
    """从Excel文件加载数据 - 移除氮含量列"""
    print(f"Loading data from: {data_path}")
    
    # 只支持Excel文件
    if not data_path.endswith(('.xlsx', '.xls')):
        raise ValueError("请使用Excel文件（.xlsx或.xls）。当前代码只支持表格格式数据")
    
    # 加载Excel文件
    print("加载Excel表格文件...")
    df = pd.read_excel(data_path)
    print(f"Excel数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print("\n数据样本:")
    print(df.head())
    
    # 转换为numpy数组
    sheet_data = df.values
    
    # 分析数据结构
    if len(sheet_data.shape) == 2:
        n_rows, n_cols = sheet_data.shape
        print(f"\n表格格式: {n_rows}行 × {n_cols}列")
        
        # 显示前几行数据样本
        print("\n数据样本（前5行，前15列）:")
        for i in range(min(5, n_rows)):
            row_data = sheet_data[i, :min(15, n_cols)]
            print(f"行{i+1}: {[f'{val:.3f}' if isinstance(val, (int, float)) else str(val) for val in row_data]}")
        
        # 新的数据结构（移除氮含量）：
        # 第1列: 时期编号
        # 第2列: 叶绿素
        # 第3-206列: 植物叶片光谱数据 (204列)
        # 第207列: 编号
        # 第208-411列: 土壤光谱信号 (204列)
        
        if n_cols >= 411:  # 确保有足够的列（减少了一列）
            print(f"\n检测到新结构（{n_cols}列）- 已移除氮含量列")
            
            group_info = sheet_data[:, 0].flatten()           # 第1列：时期编号
            chlorophyll = sheet_data[:, 1].flatten()          # 第2列：叶绿素
            
            # 提取植物叶片光谱数据（第3-206列，共204列）
            leaf_spectra = sheet_data[:, 2:206]              # 第3-206列：植物叶片光谱
            
            # 提取土壤光谱信号（第208-411列，共204列）
            soil_spectra = sheet_data[:, 207:411]            # 第208-411列：土壤光谱
            
            
            # 注意：跳过第207列（编号列）
            soil_spectra = sheet_data[:, 207:411]            # 第208-411列：土壤光谱
            
            # 第207列作为额外的编号信息（可选使用）
            sample_ids = sheet_data[:, 206]                  # 第207列：编号
            
            print(f"数据提取成功:")
            print(f"  时期编号: {group_info.shape}, 范围: [{np.min(group_info)}, {np.max(group_info)}]")
            print(f"  叶绿素: {chlorophyll.shape}, 范围: [{np.min(chlorophyll):.3f}, {np.max(chlorophyll):.3f}]")
            print(f"  叶片光谱: {leaf_spectra.shape}")
            print(f"  土壤光谱: {soil_spectra.shape}")
            print(f"  样本编号: {sample_ids.shape}, 范围: [{np.min(sample_ids)}, {np.max(sample_ids)}]")
            
            # 验证数据完整性
            unique_groups = np.unique(group_info)
            print(f"  唯一时期数: {len(unique_groups)} (值: {unique_groups})")
            
        else:
            raise ValueError(f"数据列数不足（{n_cols}列），期望至少411列")
    
    else:
        raise ValueError("数据格式不正确，期望二维表格格式")
    
    return leaf_spectra, soil_spectra, chlorophyll, group_info


def split_data_by_group(group_info, test_size=0.2, val_size=0.2, random_state=42):
    """根据分组信息进行数据划分"""
    np.random.seed(random_state)
    
    # 获取唯一的分组
    unique_groups = np.unique(group_info)
    n_groups = len(unique_groups)
    
    # 计算每个划分的组数
    n_test_groups = max(1, int(n_groups * test_size))
    n_val_groups = max(1, int(n_groups * val_size))
    n_train_groups = n_groups - n_test_groups - n_val_groups
    
    print(f"Total groups: {n_groups}, Train groups: {n_train_groups}, Val groups: {n_val_groups}, Test groups: {n_test_groups}")
    
    # 随机打乱组的顺序
    shuffled_groups = np.random.permutation(unique_groups)
    
    # 分配组到不同的集合
    train_groups = shuffled_groups[:n_train_groups]
    val_groups = shuffled_groups[n_train_groups:n_train_groups + n_val_groups]
    test_groups = shuffled_groups[n_train_groups + n_val_groups:]
    
    # 根据组分配创建索引
    train_indices = np.where(np.isin(group_info, train_groups))[0]
    val_indices = np.where(np.isin(group_info, val_groups))[0]
    test_indices = np.where(np.isin(group_info, test_groups))[0]
    
    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}, Test samples: {len(test_indices)}")
    
    return train_indices, val_indices, test_indices


def create_datasets(X_train_leaf, X_val_leaf, X_test_leaf,
                    X_train_soil, X_val_soil, X_test_soil,
                    y_train, y_val, y_test):
    """创建数据加载器 - 移除氮元素参数"""
    # 确保所有数据都是数值类型
    y_train = np.array(y_train, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    
    train_dataset = FusionDataset(X_train_leaf, X_train_soil, y_train)
    val_dataset = FusionDataset(X_val_leaf, X_val_soil, y_val)
    test_dataset = FusionDataset(X_test_leaf, X_test_soil, y_test)

    return train_dataset, val_dataset, test_dataset