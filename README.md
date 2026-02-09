# 📈 0056 ETF 股價預測與自動化監控系統

> * **目標**：透過機器學習預測 0056 (元大高股息) 收盤價，並實現每日自動化訓練與視覺化監控。
> * **表現**：最終成績 **RMSE 39.02**，超越 Baseline Good (39.95)。

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
├── 🚀 自動化排程區 (Airflow DAGs)
│   └── 📁 dags/
│       └── 📜 stock_workflow.py  # 定義每日下午 3:00 的預測自動化流程
│
├── 📊 experiments/           # [數據與實驗紀錄]
│   ├── training_log.csv      # 自動記錄每次實驗的分數、筆記與特徵
│   └── models/               # 存放訓練好的模型 (.pkl)，方便重複使用
│
├── 🛠️ 核心功能模組 (.py)
│   ├── 📈 app.py                # Streamlit 視覺化監控看板入口
│   ├── 🧠 model_train.py        # 模型訓練、驗證與產出結果的核心邏輯
│   ├── 📥 data_loader.py         # 負責讀取與合併 0056 及成分股 CSV
│   ├── 🧹 data_preprocessing.py  # 資料清洗、排序與缺失值處理
│   ├── 🧪 feature_eng.py         # 計算技術指標 (RSI, MACD, 乖離率)
│   └── 📝 experiment_logger.py   # 自動化實驗紀錄系統
│
└── 📜 環境與文件
    ├── 📜 main.ipynb            # [實驗室] 研究開發用 Notebook
    ├── 📜 README.md             # 專案說明文件
    ├── 📜 .gitignore            # Git 忽略清單
    └── 📜 requirements.txt      # 專案依賴套件清單
```

---

## 🚀 How to Run (如何執行)

### 1. 安裝環境
請確保已安裝 Python 3.8+ 以及相關套件：

```bash
pip install -r requirements.txt
```

### 2. 啟動互動式看板 (Streamlit)
執行以下指令即可開啟 Web 介面查看 0056 預測趨勢：

```bash
streamlit run app.py
```

### 3. 啟動自動化排程 (Airflow 3.x)
若要啟動每日下午 3:00 的自動化訓練流程，請執行：

```bash
# 設定家目錄為當前專案路徑 & 啟動 Airflow (Standalone 模式)
export AIRFLOW_HOME=$(pwd) && airflow standalone
```
登入憑證：密碼請參閱專案根目錄下的 simple_auth_manager_passwords.json.generated 檔案
