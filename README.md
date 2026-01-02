# 📈 Stock Prediction AI: 0056 ETF Price Forecasting

> 透過機器學習預測 0056 元大高股息 ETF 收盤價。
> 結合特徵工程 (Feature Engineering) 與 XGBoost 模型，達成 RMSE 39.02 的優異成績。

## 🏆 Project Highlights (專案亮點)

* **Performance**: 最終成績 **RMSE 39.02**，成功超越 Baseline Good (39.95)。
* **Modular Pipeline**: 將資料處理流程模組化，實現「實驗室 (Notebook)」與「產線 (Script)」分離的專業架構。
* **Experiment Tracking**: 內建實驗紀錄系統，自動追蹤每次訓練的參數與分數。
* **Feature Engineering**: 發現關鍵成分股 (2352 佳世達) 的動能指標 (Return/Bias) 對預測有決定性影響。

---

## 📂 Project Structure (檔案架構)

本專案採用模組化設計，由 `main.ipynb` 作為中控台，呼叫各個功能模組：

```text
📁 stock-prediction-project/
│
├── 📜 main.ipynb             # [中控台] 唯一的執行入口。負責參數設定、呼叫模組、視覺化分析。
│
├── 🛠️ src (核心模組)
│   ├── data_loader.py        # 負責讀取與合併原始 CSV 資料
│   ├── data_preprocessing.py # 負責資料清洗 (處理空值、排序)
│   ├── feature_eng.py        # 負責特徵工程 (計算 RSI, MACD, 週期特徵, 乖離率)
│   ├── model_train.py        # 負責模型訓練、驗證切分、產出預測結果
│   └── experiment_logger.py  # 負責自動寫入實驗紀錄 (CSV)
│
├── 📁 experiments/           # [實驗紀錄區]
│   ├── training_log.csv      # 自動記錄每次實驗的分數、筆記與特徵
│   └── models/               # 存放訓練好的模型 (.pkl)，方便重複使用
│
├── 📁 data/                  # [資料區] (Git Ignored)
│   └── (原始 .csv 檔案)
│
└── 📜 requirements.txt       # 專案依賴套件列表
