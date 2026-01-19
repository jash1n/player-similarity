import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from urllib.parse import urlparse, parse_qs



import numpy as np

# =====================
# ページ設定
# =====================
st.set_page_config(
    page_title="選手類似度統合システム",
    layout="wide"
)

st.title("選手発掘システム（TF-IDF × スタッツ × ヒートマップ）")

# =====================
# CSV パス
# =====================
TFIDF_CSV = "data/特徴語マトリックス.csv"
STATS_CSV = "data/スタッツマトリックス.csv"
HEATMAP_CSV = "data/ヒートマップマトリックス.csv"
HEATMAP_IMG_DIR = Path("data/ヒートマップ")

# =====================
# 関数
# =====================
@st.cache_data
def load_matrix(path):
    df = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def minmax_normalize(df):
    min_val = df.min().min()
    max_val = df.max().max()
    if max_val - min_val == 0:
        return df * 0
    return (df - min_val) / (max_val - min_val)

@st.cache_data
def load_youtube_links(path="data/youtube_link.csv"):
    df = pd.read_csv(path)
    return dict(zip(df["Player"], df["Link"]))

youtube_links = load_youtube_links()

def show_youtube(player_name):
    link = youtube_links.get(player_name)

    if link:
        st.video(link)
    else:
        st.info(f"YouTube動画が登録されていません: {player_name}")

# =====================
# データ読み込み
# =====================
try:
    tfidf = load_matrix(TFIDF_CSV)
    stats = load_matrix(STATS_CSV)
    heatmap = load_matrix(HEATMAP_CSV)
except Exception as e:
    st.error(f"CSV読み込みエラー: {e}")
    st.stop()

# =====================
# 共通選手抽出
# =====================
domestic_players = sorted(
    set(tfidf.index) & set(stats.index) & set(heatmap.index)
)
overseas_players = sorted(
    set(tfidf.columns) & set(stats.columns) & set(heatmap.columns)
)

tfidf = tfidf.loc[domestic_players, overseas_players]
stats = stats.loc[domestic_players, overseas_players]
heatmap = heatmap.loc[domestic_players, overseas_players]

# =====================
# 正規化
# =====================
tfidf_n = minmax_normalize(tfidf)
stats_n = minmax_normalize(stats)
heatmap_n = minmax_normalize(heatmap)

# =====================
# 重み設定
# =====================
st.sidebar.header("⚖ 重み設定")

w_tfidf = st.sidebar.slider("TF-IDF", 0.0, 1.0, 0.33, 0.01)
w_stats = st.sidebar.slider("スタッツ", 0.0, 1.0, 0.33, 0.01)
w_heat = st.sidebar.slider("ヒートマップ", 0.0, 1.0, 0.34, 0.01)

total = w_tfidf + w_stats + w_heat
w_tfidf /= total
w_stats /= total
w_heat /= total

# =====================
# 統合類似度
# =====================
final_similarity = (
    tfidf_n * w_tfidf +
    stats_n * w_stats +
    heatmap_n * w_heat
)

# =====================
# 検索方向スイッチ
# =====================
st.sidebar.header("🔁 検索方向")

mode = st.sidebar.radio(
    "検索方向を選択",
    ["国内 → 海外", "海外 → 国内"]
)

# =====================
# 類似選手検索
# =====================
st.subheader("類似選手検索")

TOP_N = 7

if mode == "国内 → 海外":
    base_players = domestic_players
    sim_matrix = final_similarity
    base_label = "国内選手"
    target_label = "海外選手"
else:
    base_players = overseas_players
    sim_matrix = final_similarity.T
    base_label = "海外選手"
    target_label = "国内選手"

player = st.selectbox(f"{base_label}を選択", base_players)

result = (
    sim_matrix.loc[player]
    .sort_values(ascending=False)
    .head(TOP_N)
    .reset_index()
)

result.columns = [target_label, "類似度"]
result["類似度"] = result["類似度"].round(3)

st.dataframe(
    result,
    width=700,
    height=280,
    hide_index=True
)
with st.expander("🎥 選手ハイライト動画"):
    st.markdown(f"### 🎯 選択選手：{player}")
    show_youtube(player)

    st.markdown("### 🔍 類似選手")

    cols = st.columns(2)
    for i, p in enumerate(result[target_label]):
        with cols[i % 2]:
            st.markdown(f"**{p}**")
            show_youtube(p)


with st.expander("🗺 選手ヒートマップ（選択＋類似選手）"):
    st.markdown(f"### 🎯 選択選手：{player}")

    base_img = HEATMAP_IMG_DIR / f"{player}.png"
    if base_img.exists():
        st.image(base_img, width=350)
    else:
        st.warning(f"ヒートマップ画像が見つかりません: {player}")

    st.markdown("### 🔍 類似選手")

    cols = st.columns(4)
    for i, p in enumerate(result[target_label]):
        img_path = HEATMAP_IMG_DIR / f"{p}.png"
        with cols[i % 4]:
            if img_path.exists():
                st.image(img_path, caption=p, width=250)
            else:
                st.warning(p)


# =====================
# 類似度マトリックス表示
# =====================
with st.expander("📊 統合類似度マトリックスを見る"):
    st.dataframe(
        final_similarity,
        height=600
    )
with st.expander("🔥 総合類似度ヒートマップ"):
    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(
        final_similarity,
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "類似度"},
        xticklabels=True,
        yticklabels=True
    )

    ax.set_xlabel("海外選手")
    ax.set_ylabel("国内選手")

    st.pyplot(fig)

