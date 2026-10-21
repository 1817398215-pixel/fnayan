import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ================== 全局字体设置 ==================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.unicode_minus"] = False

# ================== 保存目录 ==================
save_dir = "picture"
os.makedirs(save_dir, exist_ok=True)

# ================== 读取数据 ==================
file_path = "叶片.xlsx"   # 修改为你的文件路径
df = pd.read_excel(file_path, header=None)

df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna(axis=1, how="any")

X_leaf = df.iloc[:, :204].values
X_soil = df.iloc[:, 204:408].values

# 保存原始光谱
X_leaf_raw = X_leaf.copy()
X_soil_raw = X_soil.copy()

wavelengths = np.linspace(411, 989, 204)

# ================== SG 滤波 ==================
X_leaf = savgol_filter(X_leaf, window_length=11, polyorder=2, axis=1)
X_soil = savgol_filter(X_soil, window_length=11, polyorder=2, axis=1)

# ================== Z-score（按样本） ==================
X_leaf_z = (X_leaf - X_leaf.mean(axis=1, keepdims=True)) / X_leaf.std(axis=1, keepdims=True)
X_soil_z = (X_soil - X_soil.mean(axis=1, keepdims=True)) / X_soil.std(axis=1, keepdims=True)

# ================== 画图函数（统一风格） ==================
def plot_spectra(X, title, ylabel, filename):
    plt.figure(figsize=(6, 4))
    for i in range(X.shape[0]):
        plt.plot(wavelengths, X[i], linewidth=0.4)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.show()

# ================== 绘图 ==================

# 1️⃣ 叶片 原始光谱
plot_spectra(
    X_leaf_raw,
    "Leaf spectra (raw)",
    "Reflectance",
    "Leaf_raw.png"
)

# 2️⃣ 土壤 原始光谱
plot_spectra(
    X_soil_raw,
    "Soil spectra (raw)",
    "Reflectance",
    "Soil_raw.png"
)

# 3️⃣ 叶片 SG 滤波
plot_spectra(
    X_leaf,
    "Leaf spectra (SG filtered, window=11, poly=2)",
    "Reflectance",
    "Leaf_SG.png"
)

# 4️⃣ 叶片 SG + Z-score
plot_spectra(
    X_leaf_z,
    "Leaf spectra (SG + Z-score)",
    "Normalized reflectance",
    "Leaf_SG_Zscore.png"
)

# 5️⃣ 土壤 SG 滤波
plot_spectra(
    X_soil,
    "Soil spectra (SG filtered, window=11, poly=2)",
    "Reflectance",
    "Soil_SG.png"
)

# 6️⃣ 土壤 SG + Z-score
plot_spectra(
    X_soil_z,
    "Soil spectra (SG + Z-score)",
    "Normalized reflectance",
    "Soil_SG_Zscore.png"
)
