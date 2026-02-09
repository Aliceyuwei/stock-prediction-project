import streamlit as st
import pandas as pd
import plotly.express as px
import os

# 1. 導入你寫好的搬運工函數 (請確保名稱與 data_loader.py 內一致)
from data_loader import load_and_merge_data

# 設定網頁標題與佈局
st.set_page_config(page_title="股票預測專案看板", layout="wide")

# --- 側邊欄設計 ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2422/2422796.png", width=100)
st.sidebar.title("控制面板")

menu = st.sidebar.radio("功能選單", ["數據探索 (EDA)", "預測結果展示", "模型訓練狀態"])


# --- 資料讀取 ---
# 使用 st.cache_data 避免每次切換選單都要重新讀取檔案，提高效率
@st.cache_data
def get_cached_data():
    return load_and_merge_data("./data/")


df = get_cached_data()

# --- 主畫面邏輯 ---
st.title("📈 0056 股票預測分析系統")

if df is not None:
    if menu == "數據探索 (EDA)":
        st.header("🔍 原始數據與特徵探索")

        # 顯示指標卡 (Metrics)
        m1, m2, m3 = st.columns(3)
        m1.metric("資料總筆數", f"{len(df)} 筆")
        m2.metric("特徵數量", f"{len(df.columns)-1} 個")
        m3.metric("目標股票", "0056.TW")

        st.divider()

        # 左右佈局：左邊看表格，右邊看圖表
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("📋 數據預覽")
            st.dataframe(df.tail(15), height=400)

        with col2:
            st.subheader("📈 趨勢分析")
            # 排除 date 欄位後讓使用者選擇
            target_col = st.selectbox(
                "選擇要觀測的股票欄位", [c for c in df.columns if c != "date"]
            )
            fig = px.line(
                df, x="date", y=target_col, title=f"{target_col} 歷史價格走勢"
            )
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

    elif menu == "預測結果展示":
        st.header("🔮 預測結果對比")

        if os.path.exists("submission.csv"):
            sub_df = pd.read_csv("submission.csv")
            st.success("成功讀取最近一次預測結果！")

            if "date" in sub_df.columns:
                # 繪製圖表時，排除 date 欄位，只畫價格
                chart_data = (
                    sub_df.drop(columns=["date"])
                    if "date" in sub_df.columns
                    else sub_df
                )

                st.subheader("趨勢圖表")
                # 使用 Plotly 畫圖會比 st.line_chart 更清晰
                fig_res = px.line(
                    sub_df,
                    x="date",
                    y=[c for c in sub_df.columns if c != "date"],
                    title="預測與實際數值對比",
                )
                st.plotly_chart(fig_res, use_container_width=True)

            st.subheader("📋 詳細數據表")
            st.dataframe(sub_df)
        else:
            st.info(
                "💡 目前尚未偵測到 submission.csv。請先執行模型訓練以產生預測數據。"
            )

    elif menu == "模型訓練狀態":
        st.header("🧪 實驗紀錄 (Experiments)")
        # 讀取你的 experiments 資料夾
        if os.path.exists("experiments"):
            exp_files = os.listdir("experiments")
            st.write(f"目前共有 {len(exp_files)} 筆實驗紀錄。")
            st.json(exp_files)  # 簡單列出檔案名
        else:
            st.warning("找不到 experiments 資料夾。")

else:
    st.error("❌ 無法載入資料，請確認 data 資料夾中包含 0056.csv 與相關副股檔案。")

# --- 頁尾 ---
st.caption("Developed by Alice | Environment: Miniforge3 (ds_study)")
