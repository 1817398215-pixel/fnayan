import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ================== 字体设置 ==================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.unicode_minus"] = False

# ================== 保存目录 ==================
save_dir = "picture"
os.makedirs(save_dir, exist_ok=True)

# ================== 读取数据 ==================
file_path = "叶片.xlsx"
df = pd.read_excel(file_path, header=None)

df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna(axis=1, how="any")

X_leaf = df.iloc[:, :204].values
wavelengths = np.linspace(411, 989, 204)

# ================== SG 滤波 ==================
X_leaf_sg = savgol_filter(X_leaf, window_length=11, polyorder=2, axis=1)

# ================== 两种 Z-score ==================

# 1️⃣ 按样本 Z-score（sample-wise）
X_leaf_z_sample = (
    X_leaf_sg
    - X_leaf_sg.mean(axis=1, keepdims=True)
) / X_leaf_sg.std(axis=1, keepdims=True)

# 2️⃣ 按波段 Z-score（band-wise）
X_leaf_z_band = (
    X_leaf_sg
    - X_leaf_sg.mean(axis=0, keepdims=True)
) / X_leaf_sg.std(axis=0, keepdims=True)

# ================== 绘图函数 ==================
def plot_compare(X, title, filename, ylabel):
    plt.figure(figsize=(6, 4))
    for i in range(X.shape[0]):
        plt.plot(wavelengths, X[i], linewidth=0.4)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.show()

# ================== 对比图 ==================

# 图1：按样本 Z-score
plot_compare(
    X_leaf_z_sample,
    "Leaf spectra (SG + sample-wise Z-score)",
    "Leaf_Zscore_samplewise.png",
    "Normalized reflectance"
)

# 图2：按波段 Z-score
plot_compare(
    X_leaf_z_band,
    "Leaf spectra (SG + band-wise Z-score)",
    "Leaf_Zscore_bandwise.png",
    "Normalized reflectance"
)
