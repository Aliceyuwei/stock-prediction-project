import pandas as pd
import numpy as np

def add_technical_indicators(df):
    """
    特徵工程 (純時間版)：
    因為考試集 (Test) 與訓練集 (Train) 之間有巨大的時間斷層，
    且我們不知道考試期間的「昨日股價」，所以不能用 Lag/RSI/MACD。
    
    我們改用「時間特徵」來捕捉趨勢與季節性。
    """
    df = df.copy()
    
    # 1. 確保已排序
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)
    else:
        df = df.sort_index()

    # 2. 產生時間相關特徵
    if 'date' in df.columns:
        # A. 長期趨勢 (Trend)
        df['Date_Int'] = df['date']
        
        # B. 週期性特徵 (Seasonality)
        df['Day_Mod_5'] = df['date'] % 5   # 猜測星期幾
        df['Day_Mod_20'] = df['date'] % 20 # 猜測月週期
        df['Day_Mod_60'] = df['date'] % 60 # 猜測季週期
        
        # Sin/Cos 特徵
        df['Sin_Week'] = np.sin(2 * np.pi * df['date'] / 5)
        df['Cos_Week'] = np.cos(2 * np.pi * df['date'] / 5)
        df['Sin_Month'] = np.sin(2 * np.pi * df['date'] / 20)
        df['Cos_Month'] = np.cos(2 * np.pi * df['date'] / 20)

    # 3. 處理每支股票
    close_cols = [c for c in df.columns if 'close' in c.lower()]
    print(f"📊 [Feature Engineering] 轉為純時間特徵模式 ({len(close_cols)} 支股票)...")


    # 根據圖表，2352, 9945, 1101 是前三名的關鍵股票
    top_features = ['2352_close', '9945_close', '1101_close']
    
    for col in top_features:
        if col in df.columns:
            # 1. 漲跌幅 (Momentum): 今天比昨天漲跌多少 %
            # 這能幫助模型理解「趨勢」，而不只是「價格」
            df[f'{col}_Return'] = df[col].pct_change()
            
            # 2. 乖離率 (Bias): 股價距離 5 日均線多遠
            # 這是很強的技術指標
            ma5 = df[col].rolling(window=5).mean()
            df[f'{col}_Bias'] = (df[col] - ma5) / ma5

    # 4. 清理
    df = df.fillna(0)
    
    print(f"✅ 特徵工程 (時間版) 完成！欄位數: {len(df.columns)}")
    return df