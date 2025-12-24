import pandas as pd
import os

def load_and_merge_data(data_path='./data/'):
    """
    純粹的搬運工：只負責讀取 CSV 並合併，不處理缺失值
    """
    print(f"🚀 [Loader] 開始讀取資料...")

    # 1. 讀取主角
    if not os.path.exists(data_path + '0056.csv'):
        print(f"❌ 找不到 {data_path}0056.csv")
        return None

    df_train = pd.read_csv(data_path + '0056.csv')
    
    # 2. 讀取配角並合併
    stock_list = ['1101', '2327', '2352', '2385', '2449', '2915', '3005', '3532', '6176', '9945']
    
    for stock_code in stock_list:
        file_path = data_path + f"{stock_code}.csv"
        
        if os.path.exists(file_path):
            df_feature = pd.read_csv(file_path)
            
            try:
                # 只抓 close，不做任何補值
                [close_col] = df_feature.filter(like='close').columns
                df_temp = df_feature[['date', close_col]].copy()
                df_train = pd.merge(df_train, df_temp, on='date', how='left')
            except ValueError:
                pass
                
    print(f"✅ 資料載入完成 (尚未清洗)！大小：{df_train.shape}")
    return df_train