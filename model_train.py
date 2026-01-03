# =================================================
# model_train.py - 強化版：目標轉換與自動路徑偵測
# =================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
# 機器學習模型與調參工具
from xgboost import XGBRegressor
import optuna
# 模型評估指標
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error 
import joblib

# 設定 Seaborn 風格
sns.set_style("whitegrid")

# =================================================
# 1. 繪圖小幫手 (Visualizer Class)
# =================================================
class ModelVisualizer:
    """專門負責實驗視覺化與圖片歸檔的類別"""
    def __init__(self, timestamp, plot_dir):
        self.timestamp = timestamp
        self.plot_dir = plot_dir

    def plot_validation_curve(self, y_val, preds, val_score, mape):
        """圖 A: 驗證集預測走勢圖"""
        plt.figure(figsize=(12, 5))
        plt.plot(y_val.index, y_val, label='Actual', color='blue', marker='o', markersize=4)
        plt.plot(y_val.index, preds, label='Predicted', color='red', linestyle='--', marker='x', markersize=4)
        plt.title(f"Validation Period: Actual vs Predicted\n(RMSE: {val_score:.4f}, MAPE: {mape:.2%})")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
        save_path = f"{self.plot_dir}/val_{self.timestamp}_rmse_{val_score:.2f}.png"
        plt.savefig(save_path)
        print(f"📊 驗證走勢圖已儲存: {save_path}")

    def plot_feature_importance(self, model, feature_names):
        """圖 B: 特徵重要性圖"""
        plt.figure(figsize=(10, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        top_feat_names = feature_names[indices].tolist()
        
        plt.title(f"Top 15 Feature Importances_{self.timestamp}")
        plt.bar(range(len(top_feat_names)), importances[indices], color='green')
        plt.xticks(range(len(top_feat_names)), top_feat_names, rotation=90)
        plt.tight_layout()
        
        save_path = f"{self.plot_dir}/fi_{self.timestamp}.png"
        plt.savefig(save_path)
        print(f"📊 特徵重要性圖已儲存: {save_path}")
        return top_feat_names

    def plot_correlation_heatmap(self, df, top_features, target_col):
        """圖 C: 相關係數熱力圖"""
        plt.figure(figsize=(12, 10))
        plot_cols = top_features + [target_col]
        # 過濾掉不在 df 中的欄位
        plot_cols = [c for c in plot_cols if c in df.columns]
        corr_matrix = df[plot_cols].corr()
        
        plt.title(f"Feature Correlation Heatmap_{self.timestamp}", fontsize=15)
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, square=True)
        plt.tight_layout()
        
        save_path = f"{self.plot_dir}/heatmap_{self.timestamp}.png"
        plt.savefig(save_path)
        print(f"📊 相關係數熱力圖已儲存: {save_path}")

# =================================================
# 2. 主訓練流程
# =================================================
def train_and_predict(df_features, submission_file='sample_submission.csv', use_optuna=False):
    print("🚀 [Training] 啟動模型訓練生產線...")
    
    # --- 1. 自動偵測工作目錄與路徑 ---
    current_path = os.getcwd()
    # 如果是在 archive 下執行，修正路徑前綴
    is_in_archive = os.path.basename(current_path) == "archive"
    prefix = "" if is_in_archive else "archive/"
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    # 不再強制加 archive/，讓它根據執行位置決定 experiments 資料夾在哪
    plot_dir = "experiments/plots" 
    os.makedirs(plot_dir, exist_ok=True)
    viz = ModelVisualizer(timestamp, plot_dir)

    # 搜尋考卷檔案
    possible_paths = [submission_file, "submission.csv", "sample_submission.csv", 
                      "../sample_submission.csv", "../data/sample_submission.csv"]
    found_submission = None
    for p in possible_paths:
        if os.path.exists(p):
            found_submission = p
            break
    
    if not found_submission:
        raise FileNotFoundError(f"❌ 找不到考卷檔案，請檢查路徑。目前目錄: {current_path}")
    
    print(f"✅ 成功找到考卷: {found_submission}")
    submit_df = pd.read_csv(found_submission)
    target_ids = submit_df['date'].values 
    target_col = '0056_close_y' 

    # --- 2. 目標值轉換 (預測漲跌 Diff) ---
    if 'date' in df_features.columns:
        df_features = df_features.set_index('date')
    
    # 計算每日價差作為目標
    df_features['target_diff'] = df_features[target_col].diff()
    
    # 切分考試集與歷史資料
    X_test = df_features.loc[df_features.index.isin(target_ids)].copy()
    X_train_full_raw = df_features.loc[~df_features.index.isin(target_ids)].dropna().copy()
    
    # 紀錄歷史最後一天的真實價格
    last_real_price = df_features.loc[~df_features.index.isin(target_ids), target_col].iloc[-1]
    
    y_train_full = X_train_full_raw['target_diff']
    # 特徵中移除目標價格與價差
    X_train_full = X_train_full_raw.drop(columns=[target_col, 'target_diff'], errors='ignore')
    X_test = X_test.drop(columns=[target_col, 'target_diff'], errors='ignore')

    # 切分訓練與驗證
    split_idx = int(len(X_train_full) * 0.8)
    X_train, y_train = X_train_full.iloc[:split_idx], y_train_full.iloc[:split_idx]
    X_val, y_val = X_train_full.iloc[split_idx:], y_train_full.iloc[split_idx:]
    y_val_real_prices = X_train_full_raw.loc[X_val.index, target_col]

    # --- 3. 模型訓練 (Optuna) ---
    if use_optuna:
        print("🤖 [Optuna] 搜尋預測『漲跌動能』的最佳參數...")
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'random_state': 42, 'n_jobs': -1
            }
            m = XGBRegressor(**params)
            m.fit(X_train, y_train)
            # 驗證時還原價格計算 RMSE
            p_diff = m.predict(X_val)
            p_real = X_train_full_raw[target_col].shift(1).loc[X_val.index] + p_diff
            return np.sqrt(mean_squared_error(y_val_real_prices, p_real))

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        model = XGBRegressor(**study.best_params)
    else:
        model = XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6, random_state=42)

    # --- 4. 驗證與視覺化 ---
    model.fit(X_train, y_train)
    val_p_diff = model.predict(X_val)
    # 還原價格：前日價格 + 預測漲跌
    val_p_real = X_train_full_raw[target_col].shift(1).loc[X_val.index] + val_p_diff
    
    score = np.sqrt(mean_squared_error(y_val_real_prices, val_p_real))
    mape = mean_absolute_percentage_error(y_val_real_prices, val_p_real)

    viz.plot_validation_curve(y_val_real_prices, val_p_real, score, mape)
    top_feats = viz.plot_feature_importance(model, X_train.columns)
    viz.plot_correlation_heatmap(df_features, top_feats, target_col)

    # --- 5. 產出預測 ---
    model.fit(X_train_full, y_train_full)
    test_diffs = model.predict(X_test)
    
    # 累加還原考試集價格
    final_preds = []
    curr_p = last_real_price
    for d in test_diffs:
        curr_p += d
        final_preds.append(curr_p)

    submit_df[submit_df.columns[1]] = final_preds
    submit_df.to_csv('submission.csv', index=False)
    print(f"🎉 預測完成！RMSE: {score:.4f}")
    
    return model, score