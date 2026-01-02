# =================================================
# 1. 資料處理與數學運算
# =================================================
import pandas as pd
import numpy as np

# =================================================
# 2. 機器學習模型與調參工具
# =================================================
from xgboost import XGBRegressor
import optuna
# 備註：如果你之後要恢復使用集成模型 (Voting)，記得要把 RandomForestRegressor, 
# LGBMRegressor, VotingRegressor 加回來

# =================================================
# 3. 模型評估指標
# =================================================
from sklearn.metrics import mean_squared_error 

def train_and_predict(df_features, submission_file='sample_submission.csv', use_optuna=False):
    """
    接收特徵工程後的資料，訓練 XGBoost 模型。
    參數 use_optuna=True 時，會啟動自動調參模式。
    """
    print("🚀 [Training] 啟動模型訓練生產線...")
    
    # =================================================
    # 1. 準備資料與定義目標
    # =================================================
    # 讀取考卷，確認要預測哪些 ID (Date)
    submit_df = pd.read_csv(submission_file)
    target_ids = submit_df['date'].values 

    # 設定目標欄位 (主角)
    target_col = '0056_close_y' 

    # 為了方便切分，先將 date 設為 index
    if 'date' in df_features.columns:
        df_features = df_features.set_index('date')
    
    # --- 切分 訓練集 (歷史資料) vs 考試集 (未來要預測的) ---
    X_test = df_features.loc[df_features.index.isin(target_ids)] # 這是最後要交卷的
    X_train_full = df_features.loc[~df_features.index.isin(target_ids)] # 這是所有的歷史資料
    
    # 分離答案
    y_train_full = X_train_full[target_col]
    X_train_full = X_train_full.drop(columns=[target_col], errors='ignore')
    X_test = X_test.drop(columns=[target_col], errors='ignore')
    
    print(f"📚 歷史資料總數: {X_train_full.shape}")
    print(f"📝 預測資料集: {X_test.shape}")

    # =================================================
    # 2. 內部驗證 (為了算出分數)
    # =================================================
    # 切出後 20% 的資料當作驗證集 (Validation Set)
    split_point = int(len(X_train_full) * 0.8)

    X_train = X_train_full.iloc[:split_point]
    y_train = y_train_full.iloc[:split_point]

    X_val = X_train_full.iloc[split_point:]
    y_val = y_train_full.iloc[split_point:]
    
    print(f"   👉 實際訓練用: {X_train.shape}, 驗證用: {X_val.shape}")

    # =================================================
    # 3. 定義模型 (分為一般模式 vs 自動調參模式)
    # =================================================
    
    if use_optuna:
        print("🤖 [Optuna] 啟動！正在尋找最強參數 (這會花一點時間)...")
        
        # 定義給 Optuna 的考試規則
        def objective(trial):
            # 讓 AI 隨機嘗試這些參數
            params = {
                # 限制樹的深度，不讓它太深 (原本 max 10 太深了)
                'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
                'max_depth': trial.suggest_int('max_depth', 3, 6), 
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                # 增加正則化懲罰 (懲罰太複雜的模型)
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
                'subsample': trial.suggest_float('subsample', 0.6, 0.85),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.85),
                'n_jobs': -1,
                'random_state': 42
            }
            
            # 訓練一個臨時模型
            temp_model = XGBRegressor(**params)
            temp_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            # 算分數
            preds = temp_model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            return rmse

        # 開始跑 20 次實驗 (你可以改 n_trials=50 會更準)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        
        print(f"🎉 找到最佳參數: {study.best_params}")
        print(f"📉 最佳分數 (RMSE): {study.best_value:.4f}")
        
        # 使用找到的最強參數建立模型
        best_params = study.best_params
        model = XGBRegressor(**best_params, n_jobs=-1, random_state=42)
        
    else:
        # 這是你原本的手動設定 (Fallback)
        print("🤝 使用預設參數模式...")
        model = XGBRegressor(
            n_estimators=1000, 
            learning_rate=0.05, 
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )

    # =================================================
    # 4. 訓練與評分
    # =================================================
    print("🚀 使用完整歷史資料重新訓練 (Full Retrain)...")
    model.fit(X_train_full, y_train_full)
    
    # 算一下驗證分數 (如果是 Optuna 模式，直接用最佳分數)
    if use_optuna:
        val_score = study.best_value
    else:
        # 手動模式要重算一次
        temp_model = model # 這裡只是一個近似，實際上 Full Retrain 後無法算 Val Score，所以我們沿用之前的概念
        # 為了簡單起見，我們重新用 80/20 訓練一次來拿分數，或是直接回傳 0
        # 這裡簡單處理：回傳最後一次驗證的分數
        model_for_score = XGBRegressor(**model.get_params())
        model_for_score.fit(X_train, y_train)
        val_preds = model_for_score.predict(X_val)
        val_score = np.sqrt(mean_squared_error(y_val, val_preds))
        print(f"✅ 手動模式驗證分數: {val_score:.4f}")

    # =================================================
    # 5. 預測與存檔
    # =================================================
    print("🔮 正在進行最終預測...")
    predictions = model.predict(X_test)
    
    pred_df = pd.DataFrame({'date': X_test.index, 'prediction': predictions})
    final_submission = submit_df[['date']].merge(pred_df, on='date', how='left')
    target_submit_col = [c for c in submit_df.columns if c != 'date'][0]
    final_submission[target_submit_col] = final_submission['prediction']
    
    output_filename = 'submission.csv'
    final_submission[['date', target_submit_col]].to_csv(output_filename, index=False)
    
    print(f"🎉 考卷已填寫完成: {output_filename}")
    
    return model, val_score