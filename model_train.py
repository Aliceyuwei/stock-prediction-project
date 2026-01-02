# =================================================
# 1. 資料處理與數學運算
# =================================================
import pandas as pd
import numpy as np

# =================================================
# 2. 機器學習模型
# =================================================
from xgboost import XGBRegressor
# 備註：如果你之後要恢復使用集成模型 (Voting)，記得要把 RandomForestRegressor, 
# LGBMRegressor, VotingRegressor 加回來

# =================================================
# 3. 模型評估指標
# =================================================
from sklearn.metrics import mean_squared_error 

def train_and_predict(df_features, submission_file='sample_submission.csv'):
    """
    接收特徵工程後的資料，訓練 XGBoost 模型 (含驗證與重新訓練)，並產出提交檔案。
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
    # 3. 定義模型
    # =================================================
    print("🤝 正在組建模型 (XGBoost)...")
    
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
    print("🏋️ 開始訓練模型 (這可能需要幾秒鐘)...")
    model.fit(X_train, y_train)
    
    # 計算驗證分數
    val_predictions = model.predict(X_val)
    val_score = np.sqrt(mean_squared_error(y_val, val_predictions))
    print(f"✅ 模型驗證分數 (RMSE): {val_score:.4f}")

    # =================================================
    # 5. [進階] 為了交卷，用「全部」資料再訓練一次 (Retrain)
    # =================================================
    print("🚀 使用完整歷史資料重新訓練，以達到最佳預測效果...")
    model.fit(X_train_full, y_train_full)

    # =================================================
    # 6. 最終預測與存檔
    # =================================================
    print("🔮 正在進行最終預測...")
    predictions = model.predict(X_test)
    
    # 建立預測結果表
    pred_df = pd.DataFrame({
        'date': X_test.index,
        'prediction': predictions
    })
    
    # 合併回原本的考卷格式
    final_submission = submit_df[['date']].merge(pred_df, on='date', how='left')
    target_submit_col = [c for c in submit_df.columns if c != 'date'][0]
    final_submission[target_submit_col] = final_submission['prediction']
    
    # 存檔
    output_filename = 'submission.csv'
    final_submission = final_submission[['date', target_submit_col]]
    final_submission.to_csv(output_filename, index=False)
    
    print(f"🎉 恭喜！考卷已填寫完成，檔案位於: {output_filename}")
    
    return model, val_score