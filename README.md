# 📈 0056 ETF 股價預測與自動化監控系統

> **目標**：透過機器學習預測 0056 (元大高股息) 收盤價，並實現每日自動化訓練與視覺化監控。
> **表現**：最終成績 **RMSE 39.02**，超越 Baseline Good (39.95)。

## 🌟 核心特色

* **全自動排程**：整合 **Apache Airflow 3.x**，每日下午 3:00 自動觸發模型更新。
* **互動式看板**：使用 **Streamlit** 打造視覺化介面，即時掌握預測趨勢。
* **專業架構**：採用實驗室 (Notebook) 與生產環境 (Python Scripts) 分離的設計。
* **特徵工程**：導入 RSI、MACD、以及關鍵成分股 (如 2352 佳世達) 的動能指標。

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
```

---

## 🚀 How to Run (如何執行)

### 1. 安裝環境
請確保已安裝 Python 3.8+ 以及相關套件：

```bash
pip install -r requirements.txt
