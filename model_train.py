# =================================================
# 1. 
# =================================================
# 套件導入與環境設定
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

# 設定 Seaborn 風格
sns.set_style("whitegrid")

# =================================================
# 2. 繪圖小幫手 (Visualizer Class)
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
        # plt.show()

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
        # plt.show()
        return top_feat_names

    def plot_correlation_heatmap(self, df, top_features, target_col):
        """圖 C: 相關係數熱力圖 (使用 Seaborn)"""
        plt.figure(figsize=(12, 10))
        # 組合前 15 名特徵與目標價格欄位
        plot_cols = top_features + [target_col]
        corr_matrix = df[plot_cols].corr()
        
        plt.title(f"Feature Correlation Heatmap_{self.timestamp}", fontsize=15)
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap="coolwarm", 
            linewidths=0.5, 
            square=True
        )
        plt.tight_layout()
        
        save_path = f"{self.plot_dir}/heatmap_{self.timestamp}.png"
        plt.savefig(save_path)
        print(f"📊 相關係數熱力圖已儲存: {save_path}")
        # plt.show()

# =================================================
# 3. 主訓練流程 (Main Training Logic)
# =================================================
def train_and_predict(df_features, submission_file='sample_submission.csv', use_optuna=False):
    """
    接收特徵工程後的資料，訓練 XGBoost 模型。
    參數 use_optuna=True 時，會啟動自動調參模式。
    """
    print("🚀 [Training] 啟動模型訓練生產線...")
    
    # --- 初始設定 ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    plot_dir = "experiments/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # 初始化繪圖工具
    viz = ModelVisualizer(timestamp, plot_dir)

    # 檢查考卷路徑
    if not os.path.exists(submission_file):
        submission_file = 'submission.csv'
    
    # --- 資料處理 ---
    submit_df = pd.read_csv(submission_file)
    target_ids = submit_df['date'].values 

    # 設定目標欄位 (主角)
    target_col = '0056_close_y' 

    # 為了方便切分，先將 date 設為 index
    if 'date' in df_features.columns:
        df_features_indexed = df_features.set_index('date')
    else:
        df_features_indexed = df_features.copy()
    
    # --- 切分 訓練集 (歷史資料) vs 考試集 (未來要預測的) ---
    X_test = df_features_indexed.loc[df_features_indexed.index.isin(target_ids)] # 這是最後要交卷的
    X_train_full = df_features_indexed.loc[~df_features_indexed.index.isin(target_ids)] # 這是所有的歷史資料
    
    # 分離答案
    y_train_full = X_train_full[target_col]
    X_train_full = X_train_full.drop(columns=[target_col], errors='ignore')
    X_test = X_test.drop(columns=[target_col], errors='ignore')
    
    print(f"📚 歷史資料總數: {X_train_full.shape}")
    print(f"📝 預測資料集: {X_test.shape}")
    # 切分訓練與驗證集
    split_point = int(len(X_train_full) * 0.8)
    X_train, y_train = X_train_full.iloc[:split_point], y_train_full.iloc[:split_point]
    X_val, y_val = X_train_full.iloc[split_point:], y_train_full.iloc[split_point:]

    print(f"   👉 實際訓練用: {X_train.shape}, 驗證用: {X_val.shape}")
    
    # --- 模型定義與調參 ---
    if use_optuna:
        print("🤖 [Optuna] 啟動自動化參數搜尋...")
        def objective(trial):
            # 讓 AI 隨機嘗試這些參數
            params = {
                # 1. 【核心戰術】以慢打快：更多樹，但每棵樹學少一點
                'n_estimators': trial.suggest_int('n_estimators', 1500, 3500), # 拉高上限
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05), # 降低學習率
                
                # 2. 深度控制：給它一點點空間，從 3-6 放寬到 3-7
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                
                # 3. 正則化 (維持剛才的 Log 模式，這很棒)
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                
                # 4. 稍微調低 min_child_weight (原本 1-10 有點太嚴格，改 1-5)
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                
                # 其他維持不變
                'subsample': trial.suggest_float('subsample', 0.6, 0.85),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.85),
                'n_jobs': -1,
                'random_state': 42
            }
            
            # 訓練一個臨時模型
            temp_model = XGBRegressor(**params)
            temp_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            # 算分數
            return np.sqrt(mean_squared_error(y_val, temp_model.predict(X_val)))

        # 開始跑 20 次實驗 (你可以改 n_trials=50 會更準)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        
        print(f"🎉 找到最佳參數: {study.best_params}")
        print(f"📉 最佳分數 (RMSE): {study.best_value:.4f}")
        val_score = study.best_value
        model = XGBRegressor(**study.best_params, n_jobs=-1, random_state=42)
    else:
        print("🤝 使用手動預設參數模式...")

        model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        val_score = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))

    # --- 視覺化診斷 (採用 ModelVisualizer) ---
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, preds)

    # 依序執行繪圖任務 A, B, C
    viz.plot_validation_curve(y_val, preds, val_score, mape)
    top_feats = viz.plot_feature_importance(model, X_train.columns)
    viz.plot_correlation_heatmap(df_features, top_feats, target_col)

    # --- 最終產出 ---
    print("🚀 使用完整歷史資料重新訓練 (Full Retrain)...")
    model.fit(X_train_full, y_train_full)
    
    print("🔮 正在進行最終預測...")
    predictions = model.predict(X_test)

    pred_df = pd.DataFrame({'date': X_test.index, 'prediction': predictions})
    final_submission = submit_df[['date']].merge(pred_df, on='date', how='left')
    target_submit_col = [c for c in submit_df.columns if c != 'date'][0]
    final_submission[target_submit_col] = final_submission['prediction']
    final_submission[['date', target_submit_col]].to_csv('submission.csv', index=False)
    
    print(f"🎉 考卷已填寫完成: submission.csv")
    return model, val_score