import pandas as pd
import numpy as np
import re

# ========================================================
# 1. 相對強弱指標 (RSI)
# ========================================================
def calculate_rsi(series, period=14):
    """
    計算 RSI (Relative Strength Index)
    """
    # 1. 計算每日價格變動
    delta = series.diff()
    
    # 2. 區分「漲幅」與「跌幅」
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # 3. 計算 RS 與 RSI
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ========================================================
# 2. 平滑異同移動平均線 (MACD)
# ========================================================
def calculate_macd(series, fast=12, slow=26, signal=9):
    """
    計算 MACD (Moving Average Convergence Divergence)
    """
    # 1. 計算 EMA
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    
    # 2. 計算 DIF
    dif = ema_fast - ema_slow
    
    # 3. 計算 DEM (訊號線)
    dem = dif.ewm(span=signal, adjust=False).mean()
    
    # 4. 計算 OSC (柱狀圖)
    osc = dif - dem
    return dif, dem, osc

# ========================================================
# 3. 主整合函式 (Manager)
# ========================================================
def add_technical_indicators(df):
    """
    特徵工程主控台：自動加上 MA, RSI, MACD 以及進階指標 (Bias, Return, Volatility, BBP)
    """
    df = df.copy()
    
    # 自動抓取所有包含 'close' 的欄位
    close_cols = [c for c in df.columns if 'close' in c.lower()]
    print(f"📊 [Feature Engineering] 偵測到 {len(close_cols)} 支股票，開始計算全套技術指標...")

    for col in close_cols:
        # 使用 Regex 解析股票代號
        match = re.search(r'\d+', col)
        if match:
            prefix = match.group()
        else:
            print(f"⚠️ 跳過無法解析代號的欄位: {col}")
            continue
        
        # -------------------------------------------------------
        # 1. 基礎與趨勢指標
        # -------------------------------------------------------
        
        # --- A. MA (均線) ---
        ma_5 = df[col].rolling(window=5).mean()
        ma_20 = df[col].rolling(window=20).mean() # 新增 MA20 給乖離率和布林通道用
        
        df[f'{prefix}_MA_5'] = ma_5

        # --- B. RSI ---
        df[f'{prefix}_RSI'] = calculate_rsi(df[col], period=14)
        
        # --- C. MACD ---
        dif, dem, osc = calculate_macd(df[col])
        df[f'{prefix}_MACD_DIF'] = dif
        df[f'{prefix}_MACD_DEM'] = dem
        df[f'{prefix}_MACD_OSC'] = osc
        
        # -------------------------------------------------------
        # 2. 進階指標 (New Features) 🔥
        # -------------------------------------------------------

        # --- D. 乖離率 (Bias Ratio) ---
        # 意義：股價離均線太遠會「回歸」。正乖離太大賣出，負乖離太大買進。
        df[f'{prefix}_Bias_5'] = (df[col] - ma_5) / ma_5
        df[f'{prefix}_Bias_20'] = (df[col] - ma_20) / ma_20

        # --- E. 收益率 (Returns) ---
        # 意義：動能指標，看昨今兩天的漲跌幅
        df[f'{prefix}_Return_1'] = df[col].pct_change()
        
        # --- F. 波動率 (Volatility) ---
        # 意義：風險指標，計算過去 20 天漲跌幅的標準差
        df[f'{prefix}_Vol_20'] = df[col].pct_change().rolling(window=20).std()
        
        # --- G. 布林通道位置 (BBP) ---
        # 意義：股價在通道內的相對位置。 >1 代表超強勢(或超買)，<0 代表超弱勢(或超賣)
        std_20 = df[col].rolling(window=20).std()
        upper_band = ma_20 + (2 * std_20)
        lower_band = ma_20 - (2 * std_20)
        
        # (股價 - 下軌) / (上軌 - 下軌)，加 1e-9 防止分母為 0
        df[f'{prefix}_BBP'] = (df[col] - lower_band) / (upper_band - lower_band + 1e-9)
        
    # 補值：因為 MA20 會讓前 20 筆變 NaN，建議用 0 填補
    df = df.fillna(0)
    
    print(f"✅ 特徵工程完成！目前的欄位數: {len(df.columns)}")
    return df

# ========================================================
# 4. 自我測試區塊
# ========================================================
if __name__ == "__main__":
    print("🧪 [Test Mode] 模組載入成功！請在 Notebook 中呼叫 add_technical_indicators 使用。")