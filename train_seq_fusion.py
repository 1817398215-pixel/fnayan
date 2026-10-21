# train_seq_fusion.py
# 训练光谱序列融合模型的脚本

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import savgol_filter # 新增
import sys # 新增：用于在出错时退出

from models import SpectralSeqFusion as Model # 确保导入的是新模型
from training_utils import train_epoch, validate_epoch, test_model, visualize_results, regression_metrics, visualize_scatter_plot


def add_bool_argument(parser, name, default, help_text):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f'--{name}', dest=name, action='store_true', help=help_text)
    group.add_argument(f'--no-{name}', dest=name, action='store_false', help=f'Disable {name}')
    parser.set_defaults(**{name: default})


def safe_filename(name):
    return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in name)


def make_run_name(args):
    if args.run_name:
        return safe_filename(args.run_name)

    parts = [f'fusion_seq_{args.mode}', args.input_branch]
    if not args.use_sg_filter:
        parts.append('no_sg')
    if not args.use_gated_fusion:
        parts.append('no_gated_fusion')
    if not args.use_post_fusion_mlp:
        parts.append('no_post_mlp')
    if not args.use_gate_weight_feature:
        parts.append('no_gate_weight_feature')
    return safe_filename('_'.join(parts))


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train a spectral fusion model.')
    parser.add_argument('--run_name', type=str, default='', help='Name used for saving ablation outputs')
    parser.add_argument('--mode', type=str, default='CAF', choices=['ViT', 'CAF'], help='Model mode: ViT or CAF')
    parser.add_argument('--input_branch', type=str, default='both', choices=['both', 'leaf', 'soil'], help='Input branch for ablation')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--dim', type=int, default=256, help='Dimension of the model')
    parser.add_argument('--depth', type=int, default=8, help='Depth of the model')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--mlp_dim', type=int, default=512, help='Dimension of the MLP')
    parser.add_argument('--dropout', type=float, default=0.25, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    add_bool_argument(parser, 'use_group_split', False, 'Use Group K-Folds for splitting data')
    add_bool_argument(parser, 'use_sg_filter', False, 'Use Savitzky-Golay filter for smoothing')
    add_bool_argument(parser, 'use_gated_fusion', True, 'Use learnable gated weighted fusion')
    add_bool_argument(parser, 'use_post_fusion_mlp', True, 'Use post-fusion MLP')
    add_bool_argument(parser, 'use_gate_weight_feature', True, 'Concatenate gate weights into regression head')
    add_bool_argument(parser, 'test_only', False, 'Only run the test, no training')
    args = parser.parse_args()
    run_name = make_run_name(args)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Run name: {run_name}')
    print(
        "Ablation switches: "
        f"input_branch={args.input_branch}, "
        f"mode={args.mode}, "
        f"use_sg_filter={args.use_sg_filter}, "
        f"use_gated_fusion={args.use_gated_fusion}, "
        f"use_post_fusion_mlp={args.use_post_fusion_mlp}, "
        f"use_gate_weight_feature={args.use_gate_weight_feature}"
    )

    # 加载和预处理数据
    data = pd.read_excel('fanyan.xlsx')
    
    X_leaf = data.iloc[:, 5:209].values
    X_soil = data.iloc[:, 209:413].values
    y = data.iloc[:, 2].values.reshape(-1, 1)

    # 应用 Savitzky-Golay 滤波器去噪
    if args.use_sg_filter:
        print("应用 Savitzky-Golay 滤波器进行数据平滑...")
        # window_length 必须是奇数，polyorder 必须小于 window_length
        X_leaf = savgol_filter(X_leaf, window_length=11, polyorder=2, axis=1)
        X_soil = savgol_filter(X_soil, window_length=11, polyorder=2, axis=1)

    # 数据标准化
    scaler_X_leaf = StandardScaler()
    scaler_X_soil = StandardScaler()
    scaler_y = StandardScaler()

    X_leaf_scaled = scaler_X_leaf.fit_transform(X_leaf)
    X_soil_scaled = scaler_X_soil.fit_transform(X_soil)
    y_scaled = scaler_y.fit_transform(y)

    # ------------------- 数据划分（关键修改） -------------------
    if args.use_group_split:
        print("正在使用 GroupKFold 按'时期'进行严格的数据划分...")
        
        # 从第一列解析出'时期'作为组标签
        composite_ids = data.iloc[:, 0].astype(str).values
        try:
            # 提取下划线前的内容作为组ID
            groups = np.array([s.split('_')[0] for s in composite_ids])
        except Exception as e:
            print(f"错误：无法从第一列解析组ID。请确保格式为 '时期_叶片编号'。错误: {e}")
            sys.exit(1) # 如果解析失败则退出

        unique_groups = np.unique(groups)
        n_unique_groups = len(unique_groups)
        print(f"成功解析出 {n_unique_groups} 个唯一的时期: {unique_groups}")

        if n_unique_groups < 2:
            print("错误：时期（组）少于2个，无法使用 GroupKFold 进行有意义的划分。")
            sys.exit(1)

        # 使用 GroupKFold，n_splits 不能大于组的数量。
        # 对于4个时期，这将执行4折交叉验证，每次用1个时期做测试。
        gkf = GroupKFold(n_splits=n_unique_groups) 
        
        # 我们只取第一折的结果进行一次训练/测试
        train_indices, test_indices = next(gkf.split(X_leaf_scaled, y_scaled, groups=groups))
        
        # 1. 先划分出与训练/验证集完全独立的测试集
        X_leaf_train_val, X_leaf_test = X_leaf_scaled[train_indices], X_leaf_scaled[test_indices]
        X_soil_train_val, X_soil_test = X_soil_scaled[train_indices], X_soil_scaled[test_indices]
        y_train_val, y_test = y_scaled[train_indices], y_scaled[test_indices]
        
        # 打印验证信息
        train_val_groups = np.unique(groups[train_indices])
        test_groups = np.unique(groups[test_indices])
        print(f"\n用于训练/验证的时期: {train_val_groups}")
        print(f"用于测试的时期: {test_groups}")
        print(f"测试集样本数: {len(y_test)}")
        assert len(np.intersect1d(train_val_groups, test_groups)) == 0, "错误：训练集和测试集之间存在数据泄漏！"
        print("验证成功：训练/验证集与测试集所用时期完全独立。")

        # 2. 再从刚刚的训练/验证集中划分出最终的训练集和验证集
        print("\n正在从训练/验证集中划分出最终的训练集和验证集...")
        X_leaf_train, X_leaf_val, X_soil_train, X_soil_val, y_train, y_val = train_test_split(
            X_leaf_train_val, X_soil_train_val, y_train_val, test_size=0.2, random_state=42
        )
        print(f"最终训练集样本数: {len(y_train)}, 验证集样本数: {len(y_val)}")

    else:
        # 保持原有的随机划分逻辑作为对比
        print("警告：正在使用随机划分，可能因数据相似性导致评估结果虚高。")
        X_leaf_train_full, X_leaf_test, X_soil_train_full, X_soil_test, y_train_full, y_test = train_test_split(
            X_leaf_scaled, X_soil_scaled, y_scaled, test_size=0.2, random_state=42
        )
        # 进一步划分出验证集
        X_leaf_train, X_leaf_val, X_soil_train, X_soil_val, y_train, y_val = train_test_split(
            X_leaf_train_full, X_soil_train_full, y_train_full, test_size=0.2, random_state=42
        )
    # ----------------------------------------------------------

    # 转换为Tensor
    train_data = TensorDataset(torch.FloatTensor(X_leaf_train), torch.FloatTensor(X_soil_train), torch.FloatTensor(y_train))
    val_data = TensorDataset(torch.FloatTensor(X_leaf_val), torch.FloatTensor(X_soil_val), torch.FloatTensor(y_val))
    test_data = TensorDataset(torch.FloatTensor(X_leaf_test), torch.FloatTensor(X_soil_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # 初始化模型
    model = Model(
        spectral_dim=204,
        d_model=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        mode=args.mode,
        input_branch=args.input_branch,
        use_gated_fusion=args.use_gated_fusion,
        use_post_fusion_mlp=args.use_post_fusion_mlp,
        use_gate_weight_feature=args.use_gate_weight_feature
    ).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 创建保存路径
    save_dir = 'spectralformer_chl_models'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'best_model_{run_name}.pth')
    config_save_path = os.path.join(save_dir, f'config_{run_name}.json')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        config = vars(args).copy()
        config['resolved_run_name'] = run_name
        json.dump(config, f, ensure_ascii=False, indent=2)

    if not args.test_only:
        print("开始训练...")
        # 训练和验证循环
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        #patience = 15
        #patience_counter = 0

        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = validate_epoch(model, val_loader, criterion, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f'轮次 {epoch}/{args.epochs}, 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_X_leaf': scaler_X_leaf,
                    'scaler_X_soil': scaler_X_soil,
                    'scaler_y': scaler_y
                }, save_path)
                print(f'  最佳模型已保存，验证损失: {best_val_loss:.6f}')
                #patience_counter = 0
            #else:
                #patience_counter += 1

            #if patience_counter >= patience:
                #print(f'验证损失连续 {patience} 轮没有改善，触发早停机制。')
                #break
        
        print("训练完成!")

        # 保存训练/验证损失历史
    # 测试最佳模型
    checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    scaler_y_loaded = checkpoint['scaler_y']

    # 接收测试集返回的真实值、预测值和权重 g
    y_pred, y_true, g_vals = test_model(model, test_loader, device)
    
    # 逆标准化
    y_true = scaler_y_loaded.inverse_transform(y_true)
    y_pred = scaler_y_loaded.inverse_transform(y_pred)

    # 计算和打印指标
    r2, rmse, mae, rrmse = regression_metrics(y_true, y_pred)
    print(f"测试结果 - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, RRMSE: {rrmse:.2f}%")

    # --- 保存预测结果和权重 g 到 CSV ---
    # 由于 g 是一个长度为 d_model (如 256) 的向量，我们计算其在特征维度上的平均值
    # 得到一个 [0, 1] 之间的标量，代表该样本整体的门控倾向
    g_mean = np.mean(g_vals, axis=1)
    
    results_df = pd.DataFrame({
        'True_Chlorophyll': y_true.flatten(),
        'Predicted_Chlorophyll': y_pred.flatten(),
        'Gate_Weight_Leaf_Ratio_g': g_mean.flatten() # 越接近 1 代表越偏向叶片，越接近 0 代表越偏向土壤
    })
    csv_save_path = os.path.join(save_dir, f'predictions_with_weights_{run_name}.csv')
    results_df.to_csv(csv_save_path, index=False, encoding='utf-8-sig')
    print(f"\n预测详情及门控权重已保存至: {csv_save_path}")

    # 绘制和保存结果
    model_name = run_name
    # 调用新的散点图函数
    visualize_scatter_plot(y_true, y_pred, model_name, save_dir)
    # visualize_results(y_true, y_pred, model_name, save_dir) # 您可以注释掉旧的绘图函数

    # 保存训练/验证损失历史
    if not args.test_only:
        loss_history_path = os.path.join(save_dir, f'loss_history_{run_name}.csv')
        pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(loss_history_path, index=False)


if __name__ == '__main__':
    main()
