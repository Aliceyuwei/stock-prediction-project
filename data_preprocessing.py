import pandas as pd
import numpy as np

def clean_data(df):
    """
    清潔工：負責處理缺失值、異常值、排序
    """
    print("🧹 [Preprocessing] 開始清洗資料...")
    df = df.copy()

    # 1. 確保日期排序 (很重要！)
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)

    # 2. 處理無限大 (inf) -> 轉成 NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # 3. 補缺值 (先 ffill 再 fillna 0)
    df = df.ffill().fillna(0)

    print("✅ 資料清洗完成！沒有 NaN 了。")
    return df