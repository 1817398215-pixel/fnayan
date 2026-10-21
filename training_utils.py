# training_utils.py
# 训练工具模块 - 移除氮元素分支

import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def regression_metrics(y_true, y_pred):
    """计算回归评估指标"""
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # MAE
    mae = np.mean(np.abs(y_true - y_pred))

    # Relative RMSE
    rrmse = (rmse / np.mean(y_true) * 100) if np.mean(y_true) != 0 else 0

    return r2, rmse, mae, rrmse


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch - 移除氮元素参数"""
    model.train()
    total_loss = 0

    for leaf_spectra, soil_spectra, chlorophyll in train_loader:
        leaf_spectra, soil_spectra, chlorophyll = \
            leaf_spectra.to(device), soil_spectra.to(device), chlorophyll.to(device)

        optimizer.zero_grad()
        # 接收模型返回的预测值和权重 g (训练时我们只用预测值算 loss，忽略 g)
        outputs, _ = model(leaf_spectra, soil_spectra)
        loss = criterion(outputs, chlorophyll)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate_epoch(model, val_loader, criterion, device):
    """验证一个epoch - 移除氮元素参数"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for leaf_spectra, soil_spectra, chlorophyll in val_loader:
            leaf_spectra, soil_spectra, chlorophyll = \
                leaf_spectra.to(device), soil_spectra.to(device), chlorophyll.to(device)
            # 同样解包
            outputs, _ = model(leaf_spectra, soil_spectra)
            loss = criterion(outputs, chlorophyll)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def test_model(model, test_loader, device):
    """测试模型 - 移除氮元素参数"""
    model.eval()
    predictions = []
    true_values = []
    g_values = [] # 用于收集权重 g

    with torch.no_grad():
        for leaf_spectra, soil_spectra, chlorophyll in test_loader:
            leaf_spectra, soil_spectra = \
                leaf_spectra.to(device), soil_spectra.to(device)
            
            # 提取预测值和权重 g
            outputs, g = model(leaf_spectra, soil_spectra)
            predictions.extend(outputs.cpu().numpy())
            true_values.extend(chlorophyll.numpy())
            g_values.extend(g.cpu().numpy()) # 收集

    return np.array(predictions), np.array(true_values), np.array(g_values)


def visualize_scatter_plot(y_true, y_pred, model_name, save_dir):
    """
    生成并保存在论文中常见的真实值与预测值对比的散点图。
    """
    plt.style.use('default')
    plt.figure(figsize=(8, 8))
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    plt.scatter(y_true, y_pred, c='steelblue', edgecolor='black', alpha=0.8, label='Yield(g/plant)')

    min_val = min(y_true.min(), y_pred.min()) * 0.9
    max_val = max(y_true.max(), y_pred.max()) * 1.1
    plt.plot([min_val, max_val], [min_val, max_val], 'g--', label='1:1 line')

    r2, rmse, mae, rrmse = regression_metrics(y_true, y_pred)

    metrics_text = (
        f'$R^2$={r2:.3f}\n'
        f'RMSE={rmse:.4f}\n'      # 从 .2f 改为 .4f
        f'RRMSE={rrmse:.2f}%\n'
        f'MAE={mae:.4f}'          # 从 .2f 改为 .4f
    )
    plt.text(0.95, 0.05, metrics_text, transform=plt.gca().transAxes,
             fontsize=14, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='none', alpha=0.8))

    plt.xlabel('True value', fontsize=14)
    plt.ylabel('Predicted value', fontsize=14)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(False)

    save_path = os.path.join(save_dir, f'scatter_plot_{model_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"散点图已保存至: {save_path}")
    plt.show()


def visualize_results(y_true, y_pred, model_name, save_dir):
    """可视化预测结果"""
    plt.figure(figsize=(15, 10))

    # Scatter plot
    plt.subplot(2, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, s=20)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('True Chlorophyll')
    plt.ylabel('Predicted Chlorophyll')
    plt.title('Prediction vs True')
    plt.grid(True, alpha=0.3)

    # Residual plot
    plt.subplot(2, 3, 2)
    residuals = y_pred - y_true
    plt.scatter(y_true, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('True Chlorophyll')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)

    # Histogram
    plt.subplot(2, 3, 3)
    plt.hist(y_true, bins=30, alpha=0.5, label='True', density=True, color='blue')
    plt.hist(y_pred, bins=30, alpha=0.5, label='Predicted', density=True, color='orange')
    plt.xlabel('Chlorophyll')
    plt.ylabel('Density')
    plt.title('Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Error distribution
    plt.subplot(2, 3, 4)
    plt.hist(residuals, bins=30, alpha=0.7, color='green')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)

    # Time series plot
    plt.subplot(2, 3, 5)
    indices = np.arange(len(y_true))
    plt.plot(indices, y_true, 'b-', label='True', alpha=0.7)
    plt.plot(indices, y_pred, 'r-', label='Predicted', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Chlorophyll')
    plt.title('True vs Predicted Sequence')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Performance metrics text
    plt.subplot(2, 3, 6)
    plt.axis('off')
    r2, rmse, mae, rrmse = regression_metrics(y_true, y_pred)
    metrics_text = f"""
    Performance Metrics:

    R² Score: {r2:.4f}
    RMSE: {rmse:.4f}
    MAE: {mae:.4f}
    RRMSE: {rrmse:.2f}%

    Sample Count: {len(y_true)}
    True Range: [{y_true.min():.2f}, {y_true.max():.2f}]
    Pred Range: [{y_pred.min():.2f}, {y_pred.max():.2f}]
    """
    plt.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

    plt.suptitle(f'SpectralFormer Fusion - {model_name} Results (Leaf + Soil Only)', fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'results_{model_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return r2, rmse, mae, rrmse