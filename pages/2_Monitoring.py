# pages/2_Monitoring.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import ks_2samp

st.set_page_config(page_title="Monitoring & Threshold Tuning", page_icon="📊", layout="wide")

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

# Reuse clean_text & transform_features dari app utama
from app import clean_text, transform_features, predict_proba, predict_labels_from_proba

# ============================================================
# 2) Sidebar
# ============================================================
st.sidebar.header("⚙️ Settings")
st.sidebar.markdown("Halaman ini untuk **monitoring data drift** & **tuning threshold**.")

# ============================================================
# 3) Upload Reference & New Data
# ============================================================
st.title("📊 Monitoring Data Drift & Threshold Tuning")

st.markdown(
    """
    - **Reference Data** = dataset lama (digunakan untuk baseline model).  
    - **New Data** = dataset baru (akan dibandingkan distribusinya dengan Reference).  
    - **Threshold Tuning** = menemukan ambang optimal berdasarkan ROC curve.
    """
)

col1, col2 = st.columns(2)

with col1:
    ref_file = st.file_uploader("📂 Upload Reference Data (CSV)", type="csv", key="ref")
with col2:
    new_file = st.file_uploader("📂 Upload New Data (CSV)", type="csv", key="new")

if ref_file and new_file:
    df_ref = pd.read_csv(ref_file)
    df_new = pd.read_csv(new_file)

    st.subheader("🔍 Data Overview")
    st.write("Reference Data", df_ref.head())
    st.write("New Data", df_new.head())

    # ============================================================
    # 4) Drift Detection - Probabilities
    # ============================================================
    st.markdown("## 📈 Drift Detection (Prediction Probabilities)")
    
    # Pastikan ada kolom
    needed_cols = ["verified_reviews", "variation"]
    if all(c in df_ref.columns for c in needed_cols) and all(c in df_new.columns for c in needed_cols):
        # Dapatkan probabilitas prediksi
        probs_ref = predict_proba(df_ref["verified_reviews"].astype(str).tolist(),
                                  df_ref["variation"].astype(str).tolist())
        probs_new = predict_proba(df_new["verified_reviews"].astype(str).tolist(),
                                  df_new["variation"].astype(str).tolist())

        # Plot distribusi
        fig, ax = plt.subplots(figsize=(8,5))
        sns.kdeplot(probs_ref, label="Reference", shade=True)
        sns.kdeplot(probs_new, label="New", shade=True)
        ax.set_title("Distribusi Probabilitas Positif")
        ax.set_xlim(0,1)
        ax.legend()
        st.pyplot(fig)

        # KS test (Kolmogorov-Smirnov) untuk deteksi drift
        ks_stat, ks_pval = ks_2samp(probs_ref, probs_new)
        st.info(f"KS Test: statistic = {ks_stat:.3f}, p-value = {ks_pval:.3f}")
        if ks_pval < 0.05:
            st.error("❌ Terjadi Data Drift! Distribusi berbeda signifikan (p < 0.05).")
        else:
            st.success("✅ Tidak ada drift signifikan (p ≥ 0.05).")
    else:
        st.warning("Kolom `verified_reviews` & `variation` tidak lengkap di file CSV.")

    # ============================================================
    # 5) Threshold Tuning (Jika Ada Label)
    # ============================================================
    st.markdown("## 🎯 Threshold Tuning (ROC Curve)")
    if "feedback" in df_ref.columns:
        y_true = df_ref["feedback"].astype(int).values
        probs = predict_proba(df_ref["verified_reviews"].astype(str).tolist(),
                              df_ref["variation"].astype(str).tolist())

        fpr, tpr, thresholds = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6,6))
        ax.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
        ax.plot([0,1], [0,1], "k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        st.pyplot(fig)

        # Optimal threshold (Youden's J statistic)
        J = tpr - fpr
        ix = np.argmax(J)
        best_thresh = thresholds[ix]
        st.success(f"Optimal Threshold (Youden’s J): {best_thresh:.3f}")
    else:
        st.info("Tambahkan kolom `feedback` di Reference Data untuk ROC analysis & threshold tuning.")
