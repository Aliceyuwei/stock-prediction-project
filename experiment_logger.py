import pandas as pd
import os
from datetime import datetime

class ExperimentLogger:
    def __init__(self, log_file='experiments/training_log.csv'):
        self.log_file = log_file
        # 確保 experiments 資料夾存在
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def log(self, score, model_name="XGBoost", params=None, features=None, note=""):
        """
        記錄一次訓練結果
        :param score: 這次的模型分數 (例如 RMSE, MAE)
        :param model_name: 模型名稱
        :param params: 模型參數 (Dict 格式)
        :param features: 使用的特徵列表 (List 格式)
        :param note: 給自己的筆記 (例如: "新增了乖離率特徵")
        """
        
        # 1. 準備要寫入的資料
        entry = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Score': score,
            'Model': model_name,
            'Note': note,
            # 把複雜的參數轉成字串，以免 CSV 格式跑掉
            'Params': str(params) if params else "",
            'Feature_Count': len(features) if features is not None else 0,
            'Feature_List': str(features) if features is not None else ""
        }

        # 2. 讀取或建立 CSV
        if os.path.exists(self.log_file):
            df = pd.read_csv(self.log_file)
            # 使用 pd.concat 來新增資料 (取代 append)
            new_df = pd.DataFrame([entry])
            df = pd.concat([df, new_df], ignore_index=True)
        else:
            df = pd.DataFrame([entry])

        # 3. 存檔
        df.to_csv(self.log_file, index=False)
        print(f"📝 實驗紀錄已儲存至: {self.log_file}")