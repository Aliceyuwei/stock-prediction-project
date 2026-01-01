import pandas as pd
import numpy as np
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def train_and_predict(df_features, submission_file='sample_submission.csv'):
    """
    接收特徵工程後的資料，訓練集成模型，並產出提交檔案。
    
    參數:
        df_features (pd.DataFrame): 包含特徵與目標的完整資料表
        submission_file (str): 老師給的考卷檔案路徑 (用來確認要預測哪些日期)
        
    回傳:
        model: 訓練好的集成模型物件
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
    
    # --- 切分 訓練集 vs 考試集 ---
    # 考試集 (Test): 只要是 sample_submission 裡出現的日期，就是我們要考的
    X_test = df_features.loc[df_features.index.isin(target_ids)]
    
    # 訓練集 (Train): 剩下的所有資料，都拿來給 AI 讀書
    X_train = df_features.loc[~df_features.index.isin(target_ids)]
    
    # --- 分離 特徵 (X) 與 答案 (y) ---
    # 訓練集：把答案拿出來
    y_train = X_train[target_col]
    X_train = X_train.drop(columns=[target_col], errors='ignore')
    
    # 考試集：也要把(空的)答案欄位拿掉，以免影響預測
    X_test = X_test.drop(columns=[target_col], errors='ignore')
    
    print(f"📚 訓練資料集: {X_train.shape}")
    print(f"📝 預測資料集: {X_test.shape} (應與 sample_submission 列數相同)")

    # =================================================
    # 2. 定義夢幻隊伍 (Ensemble Model)
    # =================================================
    print("🤝 正在組建集成模型 (XGBoost + Random Forest + LightGBM)...")
    
    # 專家 A: XGBoost
    xgb = XGBRegressor(
        n_estimators=1000, 
        learning_rate=0.05, 
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )

    # 專家 B: Random Forest
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    # 專家 C: LightGBM
    lgbm = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    # 投票器 (集成模型)
    model = VotingRegressor(
        estimators=[
            ('xgb', xgb), 
            ('rf', rf), 
            ('lgbm', lgbm)
        ],
        weights=[2, 1, 1] # 權重配置
    )
    
    # =================================================
    # 3. 正式訓練
    # =================================================
    print("🏋️ 開始訓練模型 (這可能需要幾秒鐘)...")
    model.fit(X_train, y_train)
    print("✅ 模型訓練完成！")
    
    # =================================================
    # 4. 預測與填寫考卷
    # =================================================
    print("🔮 正在進行最終預測...")
    predictions = model.predict(X_test)
    
    # 建立預測結果表 (暫存)
    pred_df = pd.DataFrame({
        'date': X_test.index,
        'prediction': predictions
    })
    
    # 合併回原本的考卷格式 (確保順序不錯亂)
    final_submission = submit_df[['date']].merge(pred_df, on='date', how='left')
    
    # 填入答案
    # 這裡會自動抓取 sample_submission 的第二個欄位名稱 (通常是 0056_close_y)
    target_submit_col = [c for c in submit_df.columns if c != 'date'][0]
    final_submission[target_submit_col] = final_submission['prediction']
    
    # =================================================
    # 5. 存檔
    # =================================================
    output_filename = 'submission.csv'
    # 只保留老師要求的欄位
    final_submission = final_submission[['date', target_submit_col]]
    final_submission.to_csv(output_filename, index=False)
    
    print(f"🎉 恭喜！考卷已填寫完成，檔案位於: {output_filename}")
    print("前 5 筆預測結果預覽：")
    print(final_submission.head())
    
    return model