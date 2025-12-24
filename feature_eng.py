import pandas as pd
import numpy as np

def calculate_rsi(series, period=14):
    """
    輔助函式：計算 RSI (相對強弱指標)
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    """
    輔助函式：計算 MACD
    回傳: DIF, DEM, OSC
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dem = dif.ewm(span=signal, adjust=False).mean()
    osc = dif - dem
    return dif, dem, osc

def add_technical_indicators(df):
    """
    主函式：為每一支股票加上 MA, RSI, MACD 特徵
    """
    df = df.copy()
    
    # 1. 找出所有收盤價欄位
    close_cols = [c for c in df.columns if 'close' in c.lower()]
    
    print(f"📊 開始特徵工程 (MA, RSI, MACD)，共處理 {len(close_cols)} 支股票...")

    for col in close_cols:
        prefix = col.split('_')[0] 
        
        # --- A. MA ---
        df[f'{prefix}_MA_5'] = df[col].rolling(window=5).mean()
        df[f'{prefix}_MA_10'] = df[col].rolling(window=10).mean()
        
        # --- B. RSI ---
        df[f'{prefix}_RSI_14'] = calculate_rsi(df[col], period=14)
        
        # --- C. MACD ---
        dif, dem, osc = calculate_macd(df[col])
        df[f'{prefix}_MACD_DIF'] = dif
        df[f'{prefix}_MACD_DEM'] = dem
        df[f'{prefix}_MACD_OSC'] = osc
        
        # --- D. 乖離率 ---
        df[f'{prefix}_Bias_5'] = (df[col] - df[f'{prefix}_MA_5']) / df[f'{prefix}_MA_5']

        # # --- D. 收益率 (Returns) - 最重要！ ---
        # df[f'{prefix}_Return_1'] = df[col].pct_change(periods=1)
        
        # # --- E. 波動率 (Volatility/Std) ---
        # # 過去 20 天的標準差，代表風險大小
        # df[f'{prefix}_Std_20'] = df[col].rolling(window=20).std()
        
        # # --- F. 布林通道位置 (Bollinger Band Position) ---
        # ma_20 = df[col].rolling(window=20).mean()
        # std_20 = df[col].rolling(window=20).std()
        # upper = ma_20 + (2 * std_20)
        # lower = ma_20 - (2 * std_20)
        # # 避免分母為 0
        # denominator = (upper - lower)
        # # 如果上下軌距離太近(趨近0)，就給 0.5 (中間值)，不然會報錯
        # df[f'{prefix}_BB_Pos'] = np.where(denominator == 0, 0.5, (df[col] - lower) / denominator)

    # 2. 補值
    df = df.bfill().fillna(0)
    
    print(f"✅ 特徵工程完成！總欄位數: {len(df.columns)}")
    return df