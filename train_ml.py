# train_ml_dual_branch.py
# 双分支（叶片/土壤）传统机器学习对比：两路模型 + 融合回归（Ridge）
# 特点：支持 GroupKFold 严格按时期划分，并避免融合阶段泄漏（使用折外预测）

import os
import sys
import json
import argparse
import random
import numpy as np
import pandas as pd

from scipy.signal import savgol_filter

from sklearn.model_selection import train_test_split, GroupKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

import joblib

from training_utils import regression_metrics, visualize_scatter_plot


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def safe_sg_filter(X: np.ndarray, window_length: int = 11, polyorder: int = 2, axis: int = 1):
    n = X.shape[axis]
    wl = int(window_length)

    if wl > n:
        wl = n
    if wl % 2 == 0:
        wl -= 1

    min_wl = max(polyorder + 2, 5)
    if min_wl % 2 == 0:
        min_wl += 1

    if wl < min_wl:
        print(f"警告：SG窗口({wl})不合法或过小（n={n}），跳过 SG。")
        return X

    return savgol_filter(X, window_length=wl, polyorder=polyorder, axis=axis)


def parse_groups_from_first_col(df: pd.DataFrame, col_idx: int = 0, sep: str = "_") -> np.ndarray:
    composite_ids = df.iloc[:, col_idx].astype(str).values
    try:
        groups = np.array([s.split(sep)[0] for s in composite_ids])
    except Exception as e:
        raise ValueError(f"无法从第{col_idx+1}列解析组ID（sep='{sep}'）。错误: {e}")
    return groups


def build_pipeline_and_grid(model_name: str, seed: int):
    """
    给每个分支构建 pipeline + param_grid
    PLSR/SVR 需要 scaler；RF 不需要 scaler（passthrough）
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", PLSRegression())
    ])

    if model_name == "PLSR":
        grid = {
            "scaler": [StandardScaler()],
            "regressor": [PLSRegression()],
            "regressor__n_components": list(range(1, 31)),
        }
    elif model_name == "SVR":
        grid = {
            "scaler": [StandardScaler()],
            "regressor": [SVR()],
            "regressor__C": [0.1, 1, 10, 100],
            "regressor__gamma": ["scale", "auto", 0.01, 0.1],
        }
    elif model_name == "RF":
        grid = {
            "scaler": ["passthrough"],
            "regressor": [RandomForestRegressor(random_state=seed, n_jobs=-1)],
            "regressor__n_estimators": [100, 200, 300],
            "regressor__max_depth": [None, 10, 20, 30],
        }
    else:
        raise ValueError("model_name must be one of PLSR/SVR/RF")

    return pipe, grid


def main():
    parser = argparse.ArgumentParser(description="Dual-branch ML for spectral data: leaf branch + soil branch + fusion.")
    parser.add_argument("--model_leaf", type=str, default="PLSR", choices=["PLSR", "SVR", "RF"], help="Leaf branch model")
    parser.add_argument("--model_soil", type=str, default="PLSR", choices=["PLSR", "SVR", "RF"], help="Soil branch model")

    parser.add_argument("--use_group_split", action="store_true", help="Use GroupKFold for strict split by period/group")
    parser.add_argument("--test_fold_idx", type=int, default=0, help="When group split, which outer fold as TEST (0-based)")
    parser.add_argument("--group_sep", type=str, default="_", help="Separator for parsing group from first column")

    parser.add_argument("--use_sg_filter", action="store_true", help="Use Savitzky-Golay filter")
    parser.add_argument("--sg_window", type=int, default=11)
    parser.add_argument("--sg_poly", type=int, default=2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--excel_path", type=str, default="fanyan.xlsx")
    parser.add_argument("--save_dir", type=str, default="ml_models_results_dual")

    # 融合层超参（Ridge）
    parser.add_argument("--ridge_alpha", type=float, default=1.0, help="Ridge alpha for fusion regressor")

    args = parser.parse_args()
    set_seed(args.seed)

    print(f"--- 双分支对比实验开始 ---")
    print(f"Leaf 分支模型: {args.model_leaf} | Soil 分支模型: {args.model_soil}")

    if not os.path.exists(args.excel_path):
        print(f"错误：找不到数据文件 {args.excel_path}")
        sys.exit(1)

    data = pd.read_excel(args.excel_path)

    # 你的列切片（保持不变）
    X_leaf = data.iloc[:, 5:209].values
    X_soil = data.iloc[:, 209:413].values
    y = data.iloc[:, 2].values.astype(float)

    if args.use_sg_filter:
        print(f"应用 SG 滤波：window={args.sg_window}, poly={args.sg_poly}")
        X_leaf = safe_sg_filter(X_leaf, args.sg_window, args.sg_poly, axis=1)
        X_soil = safe_sg_filter(X_soil, args.sg_window, args.sg_poly, axis=1)

    # ---------- 外层划分：train_val / test ----------
    groups = None
    if args.use_group_split:
        print("使用 GroupKFold 外层按时期隔离测试集...")
        groups = parse_groups_from_first_col(data, col_idx=0, sep=args.group_sep)
        unique_groups = np.unique(groups)
        n_splits_outer = min(len(unique_groups), 4)
        if n_splits_outer < 2:
            print("错误：组数少于2，无法 GroupKFold。")
            sys.exit(1)

        outer_gkf = GroupKFold(n_splits=n_splits_outer)
        splits = list(outer_gkf.split(X_leaf, y, groups=groups))
        if args.test_fold_idx < 0 or args.test_fold_idx >= len(splits):
            print(f"错误：test_fold_idx 超范围 [0, {len(splits)-1}]")
            sys.exit(1)

        train_val_idx, test_idx = splits[args.test_fold_idx]

        X_leaf_tr, X_leaf_te = X_leaf[train_val_idx], X_leaf[test_idx]
        X_soil_tr, X_soil_te = X_soil[train_val_idx], X_soil[test_idx]
        y_tr, y_te = y[train_val_idx], y[test_idx]
        groups_tr = groups[train_val_idx]

        print(f"训练/调优组: {np.unique(groups_tr)}")
        print(f"测试组: {np.unique(groups[test_idx])}")

    else:
        print("警告：随机划分（不保证时期独立）...")
        idx_all = np.arange(len(y))
        tr_idx, te_idx = train_test_split(idx_all, test_size=0.2, random_state=args.seed)

        X_leaf_tr, X_leaf_te = X_leaf[tr_idx], X_leaf[te_idx]
        X_soil_tr, X_soil_te = X_soil[tr_idx], X_soil[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        groups_tr = None

    print(f"训练/验证集: {len(y_tr)} | 测试集: {len(y_te)}")

    # ---------- 内层：分别对两分支做调参 ----------
    leaf_pipe, leaf_grid = build_pipeline_and_grid(args.model_leaf, args.seed)
    soil_pipe, soil_grid = build_pipeline_and_grid(args.model_soil, args.seed)

    if args.use_group_split:
        n_unique = len(np.unique(groups_tr))
        n_splits_inner = min(n_unique, 5)
        if n_splits_inner < 2:
            print("错误：训练集内组数少于2，无法做 GroupKFold 调参。")
            sys.exit(1)
        inner_cv = GroupKFold(n_splits=n_splits_inner)

        leaf_gs = GridSearchCV(
            leaf_pipe, leaf_grid, cv=inner_cv,
            scoring="neg_mean_squared_error", n_jobs=-1, verbose=1
        )
        soil_gs = GridSearchCV(
            soil_pipe, soil_grid, cv=inner_cv,
            scoring="neg_mean_squared_error", n_jobs=-1, verbose=1
        )

        print("\n[Leaf 分支] 正在调参...")
        leaf_gs.fit(X_leaf_tr, y_tr, groups=groups_tr)
        print("[Soil 分支] 正在调参...")
        soil_gs.fit(X_soil_tr, y_tr, groups=groups_tr)

        best_leaf = leaf_gs.best_estimator_
        best_soil = soil_gs.best_estimator_

        print(f"\nLeaf 最佳参数: {leaf_gs.best_params_}")
        print(f"Soil 最佳参数: {soil_gs.best_params_}")

        # ---------- 融合：用“折外预测”训练融合层，避免泄漏 ----------
        # 训练集折外预测（每个样本的预测来自不包含它的训练折）
        print("\n生成训练集折外预测用于融合（避免泄漏）...")
        leaf_oof = cross_val_predict(best_leaf, X_leaf_tr, y_tr, cv=inner_cv, groups=groups_tr, n_jobs=-1).ravel()
        soil_oof = cross_val_predict(best_soil, X_soil_tr, y_tr, cv=inner_cv, groups=groups_tr, n_jobs=-1).ravel()

    else:
        leaf_gs = GridSearchCV(leaf_pipe, leaf_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)
        soil_gs = GridSearchCV(soil_pipe, soil_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)

        print("\n[Leaf 分支] 正在调参...")
        leaf_gs.fit(X_leaf_tr, y_tr)
        print("[Soil 分支] 正在调参...")
        soil_gs.fit(X_soil_tr, y_tr)

        best_leaf = leaf_gs.best_estimator_
        best_soil = soil_gs.best_estimator_

        print(f"\nLeaf 最佳参数: {leaf_gs.best_params_}")
        print(f"Soil 最佳参数: {soil_gs.best_params_}")

        # 非 group 情况：用普通5折生成折外预测
        print("\n生成训练集折外预测用于融合（避免泄漏）...")
        leaf_oof = cross_val_predict(best_leaf, X_leaf_tr, y_tr, cv=5, n_jobs=-1).ravel()
        soil_oof = cross_val_predict(best_soil, X_soil_tr, y_tr, cv=5, n_jobs=-1).ravel()

    # ---------- 训练融合层 ----------
    X_fuse_tr = np.column_stack([leaf_oof, soil_oof])   # (n, 2)
    fuse = Ridge(alpha=args.ridge_alpha, random_state=args.seed)
    fuse.fit(X_fuse_tr, y_tr)

    # ---------- 测试集预测 ----------
    leaf_te_pred = best_leaf.predict(X_leaf_te).ravel()
    soil_te_pred = best_soil.predict(X_soil_te).ravel()

    X_fuse_te = np.column_stack([leaf_te_pred, soil_te_pred])
    y_pred = fuse.predict(X_fuse_te).ravel()
    y_true = y_te.ravel()

    r2, rmse, mae, rrmse = regression_metrics(y_true, y_pred)
    print(f"\n最终融合测试结果 - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, RRMSE: {rrmse:.2f}%")

    # ---------- 保存 ----------
    os.makedirs(args.save_dir, exist_ok=True)

    tag = f"dual_leaf{args.model_leaf}_soil{args.model_soil}"
    if args.use_group_split:
        tag += f"_groupfold{args.test_fold_idx}"
    if args.use_sg_filter:
        tag += f"_sg{args.sg_window}p{args.sg_poly}"
    tag += f"_ridge{args.ridge_alpha}"

    # 散点图
    visualize_scatter_plot(y_true, y_pred, tag, args.save_dir)

    # 保存模型（两分支 + 融合层）
    joblib.dump(best_leaf, os.path.join(args.save_dir, f"best_leaf_{tag}.joblib"))
    joblib.dump(best_soil, os.path.join(args.save_dir, f"best_soil_{tag}.joblib"))
    joblib.dump(fuse,      os.path.join(args.save_dir, f"best_fuse_{tag}.joblib"))

    # 保存参数（转字符串，避免 JSON 序列化报错）
    def jsonable_params(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[k] = v
            else:
                out[k] = str(v)
        return out

    params = {
        "leaf_best_params": jsonable_params(leaf_gs.best_params_),
        "soil_best_params": jsonable_params(soil_gs.best_params_),
        "ridge_alpha": args.ridge_alpha
    }
    with open(os.path.join(args.save_dir, f"best_params_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    metrics = {
        "tag": tag,
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "rrmse_percent": float(rrmse),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "use_group_split": bool(args.use_group_split),
        "model_leaf": args.model_leaf,
        "model_soil": args.model_soil
    }
    with open(os.path.join(args.save_dir, f"metrics_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "leaf_pred": leaf_te_pred,
        "soil_pred": soil_te_pred
    }).to_csv(os.path.join(args.save_dir, f"predictions_{tag}.csv"),
              index=False, encoding="utf-8-sig")

    print(f"\n保存完成，目录：{args.save_dir}")
    print(f"--- 双分支对比实验结束 ---")


if __name__ == "__main__":
    main()
