import pandas as pd
import os

# 定義函式，讓外部可以呼叫
def load_and_merge_data(data_path='./data/'):
    """
    讀取 0056 並合併 10 支成分股的收盤價
    """
    print(f"🚀 [data_loader] 開始從 {data_path} 讀取資料...")

    # ==========================================
    # 1. 讀取主要檔案 (0056)
    # ==========================================
    
    # 檢查檔案是否存在
    if not os.path.exists(data_path + '0056.csv'):
        print(f"❌ 錯誤：找不到 {data_path}0056.csv")
        return None

    df_train = pd.read_csv(data_path + '0056.csv')
    df_train = df_train.sort_values('date')

    # ==========================================
    # 2. 讀取其他檔案
    # ==========================================
    stock_list = ['1101', '2327', '2352', '2385', '2449', '2915', '3005', '3532', '6176', '9945']

    for stock_code in stock_list:
        file_path = data_path + f"{stock_code}.csv"
        
        if os.path.exists(file_path):
            df_feature = pd.read_csv(file_path)
            # 日期排序
            df_feature = df_feature.sort_values('date')
            # 補缺值 (建議用新寫法 ffill)
            df_feature = df_feature.ffill().fillna(0)
            
            # 抓出收盤價欄位名稱
            # 這裡加個 try 以防萬一沒有 close 欄位
            try:
                [close_col] = df_feature.filter(like='close').columns
                
                # 只取需要的欄位
                df_temp = df_feature[['date', close_col]].copy()
                
                # 合併
                df_train = pd.merge(df_train, df_temp, on='date', how='left')
            except ValueError:
                print(f"⚠️ {stock_code} 找不到 close 欄位，跳過。")
        else:
            print(f"⚠️ 找不到 {stock_code}.csv，跳過。")

    print(f"✅ 資料合併完成！資料大小：{df_train.shape}")
    
    # 🔑 最重要的一步：把結果回傳出去！
    return df_train