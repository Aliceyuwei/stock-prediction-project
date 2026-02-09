from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# 將專案路徑加入系統
sys.path.append("/Users/alice/Downloads/Intelligence/stock_project")

# 導入原本的邏輯
from model_train import train_and_predict

default_args = {
    "owner": "alice",
    "start_date": datetime(2026, 2, 9),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "0056_prediction_pipeline",
    default_args=default_args,
    schedule="0 15 * * *",  # 每天下午 3 點執行
    catchup=False,
) as dag:

    # 任務 1：執行預測並更新 submission.csv
    run_prediction = PythonOperator(
        task_id="run_stock_prediction",
        python_callable=train_and_predict,  # 呼叫你原本 model_train.py 裡的 function
    )

    run_prediction
