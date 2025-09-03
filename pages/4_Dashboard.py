# pages/4_Dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time

st.set_page_config(page_title="Real-time Monitoring Dashboard", page_icon="📈", layout="wide")
sns.set_style("whitegrid")

# ============================================================
# 1) Load artifacts
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    ohe = joblib.load("onehot_encoder.pkl")
    return model, vectorizer, ohe

model, vectorizer, ohe = load_artifacts()

from app import clean_text, transform_features, predict_proba, predict_labels_from_proba

# ============================================================
# 2) Sidebar
# ============================================================
st.sidebar.header("⚙️ Dashboard Settings")
update_interval = st.sidebar.slider("Update Interval (seconds)", 1, 10, 3)
batch_size = st.sidebar.slider("Batch Size (simulasi streaming)", 5, 50, 20)

st.title("📈 Real-time Monitoring Dashboard")
st.markdown(
    """
    Dashboard ini menampilkan **performansi model secara real-time**.  
    Bisa dipakai untuk memantau:
    - Jumlah prediksi (positif vs negatif)
    - Distribusi probabilitas
    - Tren akumulatif dari waktu ke waktu
    """
)

# ============================================================
# 3) Upload Data untuk Monitoring
# ============================================================
up = st.file_uploader("📂 Upload CSV untuk simulasi real-time monitoring", type="csv")

if up is not None:
    df = pd.read_csv(up)
    needed_cols = ["verified_reviews", "variation"]
    if not all(c in df.columns for c in needed_cols):
        st.error("File CSV harus punya kolom: verified_reviews & variation")
    else:
        st.success(f"Data loaded: {df.shape[0]} rows")

        # ============================================================
        # 4) Simulasi Streaming dengan State
        # ============================================================
        if "step" not in st.session_state:
            st.session_state.step = 0
            st.session_state.history = pd.DataFrame(columns=["time", "positive", "negative", "avg_prob"])

        start_btn = st.button("▶️ Start Streaming")
        stop_btn = st.button("⏹ Stop Streaming")

        if start_btn:
            st.session_state.running = True
        if stop_btn:
            st.session_state.running = False

        placeholder = st.empty()

        while st.session_state.get("running", False) and st.session_state.step < len(df):
            step = st.session_state.step
            batch = df.iloc[step: step + batch_size]

            if len(batch) == 0:
                break

            # Prediksi batch
            probs = predict_proba(batch["verified_reviews"].astype(str).tolist(),
                                  batch["variation"].astype(str).tolist())
            preds = predict_labels_from_proba(probs)

            pos_count = (preds == 1).sum()
            neg_count = (preds == 0).sum()
            avg_prob = np.mean(probs)

            # Tambahkan ke history
            ts = time.strftime("%H:%M:%S")
            new_row = pd.DataFrame([{"time": ts, "positive": pos_count, "negative": neg_count, "avg_prob": avg_prob}])
            st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)

            # Update UI
            with placeholder.container():
                st.subheader("📊 Batch Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Positive Predictions", pos_count)
                col2.metric("Negative Predictions", neg_count)
                col3.metric("Avg Probability", f"{avg_prob:.2f}")

                st.subheader("📈 Cumulative Trends")
                fig, ax = plt.subplots(1, 2, figsize=(12,5))

                # Trend Pos/Neg
                sns.lineplot(x="time", y="positive", data=st.session_state.history, ax=ax[0], label="Positive")
                sns.lineplot(x="time", y="negative", data=st.session_state.history, ax=ax[0], label="Negative")
                ax[0].set_title("Predictions Over Time")
                ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)

                # Trend Avg Probability
                sns.lineplot(x="time", y="avg_prob", data=st.session_state.history, ax=ax[1], color="purple")
                ax[1].set_ylim(0,1)
                ax[1].set_title("Average Positive Probability Over Time")
                ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)

                st.pyplot(fig)

                st.subheader("📋 Prediction History (Last 10 Batches)")
                st.dataframe(st.session_state.history.tail(10))

            st.session_state.step += batch_size
            time.sleep(update_interval)
