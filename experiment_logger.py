import pandas as pd
import os
from datetime import datetime

# 設定紀錄檔的路徑
LOG_FILE = 'experiments/history.csv'

def log_experiment(model_name, params, rmse, mape, note=""):
    """
    將實驗結果記錄到 CSV 檔案中。
    
    參數:
        model_name (str): 模型名稱 (例如 "XGBoost", "Ensemble")
        params (str or dict): 重要參數設定 (例如 "lr=0.05, depth=6")
        rmse (float): 測試集 RMSE 分數
        mape (float): 測試集 MAPE 分數
        note (str): 備註 (例如 "嘗試拿掉 MACD 特徵")
    """
    
    # 1. 準備要寫入的一筆資料
    new_record = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Model': model_name,
        'Params': str(params), # 轉成字串以免格式跑掉
        'RMSE': rmse,
        'MAPE': mape,
        'Note': note
    }
    
    # 2. 檢查檔案是否存在
    if os.path.exists(LOG_FILE):
        # 如果有檔案，就讀進來，把新資料加在後面
        df = pd.read_csv(LOG_FILE)
        # 使用 pd.concat 取代 append (因為 append 即將被廢棄)
        new_df = pd.DataFrame([new_record])
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        # 如果沒有檔案，就直接建立一個新的
        df = pd.DataFrame([new_record])
    
    # 3. 存檔
    df.to_csv(LOG_FILE, index=False)
    print(f"📝 實驗紀錄已儲存至: {LOG_FILE}")
    print(f"   (本次成績 - MAPE: {mape:.2%})")