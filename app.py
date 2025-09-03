import os
import re
import string
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Opsional
try:
    import emoji
except Exception:
    emoji = None

# SHAP opsional
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False
    
# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="Alexa Reviews Sentiment App",
    page_icon="🔊",
    layout="wide"
)
sns.set_style("whitegrid")

# ============================================================
# 1) Utilities: Text Cleaning (harus SAMA dengan training)
# ============================================================
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# Download saat pertama kali
nltk.download("stopwords")
nltk.download("wordnet")

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """Harus konsisten dengan pipeline training."""
    text = re.sub(r'[^\x00-\x7F]+', ' ', str(text).lower())
    text = ''.join([c for c in text if c not in string.punctuation])
    text = re.sub(r"\d+", " ", text)
    if emoji is not None:
        try:
            text = emoji.replace_emoji(text, replace=" ")
        except Exception:
            pass
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]

    # handle negation simple: "not good" -> "not_good"
    new_words = []
    skip = False
    for i, w in enumerate(words):
        if w == "not" and i+1 < len(words):
            new_words.append("not_" + words[i+1])
            skip = True
        elif skip:
            skip = False
            continue
        else:
            new_words.append(w)
    words = new_words

    words = [LEMMATIZER.lemmatize(w) for w in words]
    return " ".join(words)

# ============================================================
# 2) Load Artifacts
# ============================================================

@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    ohe = joblib.load("onehot_encoder.pkl")
    return model, vectorizer, ohe

with st.spinner("Loading model & vectorizer..."):
    model, vectorizer, ohe = load_artifacts()

# Helper untuk get OHE categories
def get_variation_choices():
    try:
        cats = ohe.categories_[0].tolist()
        cats = [c if isinstance(c, str) else str(c) for c in cats]
        return cats
    except Exception:
        return []

VARIATION_CHOICES = get_variation_choices()

# ============================================================
# 3) Prediction Helpers
# ============================================================

def transform_features(review_texts: list[str], variations: list[str]):
    """
    Input:
        review_texts: list of raw strings
        variations: list of strings (harus 1-1 dengan review_texts)
    Output:
        sparse matrix gabungan TF-IDF + Onehot(variation)
    """
    # Clean texts
    clean_texts = [clean_text(t) for t in review_texts]
    X_text = vectorizer.transform(clean_texts)
    X_var = ohe.transform(pd.Series(variations).to_frame())
    from scipy.sparse import hstack
    X_final = hstack([X_text, X_var])
    return X_final, clean_texts

def predict_proba(review_texts: list[str], variations: list[str]) -> np.ndarray:
    X, _ = transform_features(review_texts, variations)
    # Beberapa model tidak punya predict_proba (tapi semua yang kita pakai punya)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # fallback to decision_function -> sigmoid approx
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # min-max normalize ke 0-1
        smin, smax = scores.min(), scores.max()
        if smax == smin:
            return np.ones_like(scores) * 0.5
        return (scores - smin) / (smax - smin)
    # fallback default
    preds = model.predict(X)
    return preds.astype(float)

def predict_labels_from_proba(p1: np.ndarray, threshold: float) -> np.ndarray:
    return (p1 >= threshold).astype(int)

# ============================================================
# 4) UI Layout
# ============================================================
st.title("🔊 Alexa Reviews Sentiment App")
st.markdown(
    """
Aplikasi ini memprediksi **sentimen** (0 = Negatif, 1 = Positif) dari ulasan Alexa, 
dengan mempertimbangkan **teks ulasan** dan **variation (warna/material)**.
"""
)

with st.sidebar:
    st.header("⚙️ Pengaturan")
    threshold = st.slider("Decision Threshold (Positif jika ≥ threshold)", 0.0, 1.0, 0.5, 0.01)
    show_clean_text = st.checkbox("Tampilkan teks yang sudah dibersihkan", value=False)
    st.markdown("---")
    st.markdown("**Artefak Model**")
    st.write("• best_model.pkl\n• tfidf_vectorizer.pkl\n• onehot_encoder.pkl")

tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📦 Batch Prediction", "📈 Evaluation & Explainability"])

# ============================================================
# 5) Tab 1 - Single Prediction
# ============================================================
with tab1:
    st.subheader("🔍 Single Prediction")
    colL, colR = st.columns([2,1])

    with colL:
        review_text = st.text_area("Tulis ulasan:", height=160, placeholder="Contoh: The Alexa is amazing, responds quickly and the sound is great!")
    with colR:
        if VARIATION_CHOICES:
            variation = st.selectbox("Pilih Variation:", VARIATION_CHOICES, index=0)
        else:
            variation = st.text_input("Ketik Variation (jika OHE tidak punya kategori):", value="Black")

    if st.button("Prediksi Sentimen"):
        if not review_text.strip():
            st.warning("Tolong isi ulasan terlebih dahulu.")
        else:
            p1 = predict_proba([review_text], [variation])[0]
            yhat = int(p1 >= threshold)
            st.success(f"Hasil Prediksi: **{'Positif' if yhat==1 else 'Negatif'}** (p= {p1:.3f}, threshold= {threshold:.2f})")

            if show_clean_text:
                st.caption("Teks setelah cleaning:")
                st.code(clean_text(review_text))

            # Visual: gauge sederhana (bar probability)
            fig, ax = plt.subplots(figsize=(5, 0.6))
            ax.barh([0], [p1], align="center")
            ax.set_xlim(0,1)
            ax.set_yticks([])
            ax.set_title("Probability of Positive")
            for spine in ["top","right","left"]:
                ax.spines[spine].set_visible(False)
            st.pyplot(fig)

            # # SHAP (opsional) - hanya jika tersedia & model kompatibel
            # if SHAP_AVAILABLE and hasattr(model, "predict_proba"):
            #     with st.expander("🔎 SHAP Explanation (opsional)"):
            #         try:
            #             X_sp, clean_list = transform_features([review_text], [variation])
            #             explainer = shap.Explainer(model, feature_names=None)
            #             shap_values = explainer(X_sp)
            #             st.write("SHAP values untuk sampel ini (summary bar):")
            #             fig2 = shap.plots.bar(shap_values[0], show=False)
            #             st.pyplot(bbox_inches='tight', dpi=150)
            #         except Exception as e:
            #             st.info(f"SHAP tidak dapat dijalankan: {e}")
            # SHAP (opsional) - hanya jika tersedia & model kompatibel
            # if SHAP_AVAILABLE and hasattr(model, "predict_proba"):
            #     with st.expander("🔎 SHAP Explanation (opsional)"):
            #         try:
            #             X_sp, clean_list = transform_features([review_text], [variation])

            # # Gunakan LinearExplainer untuk model linear seperti LogisticRegression
            #             background = vectorizer.transform(["neutral sample"]).toarray()
            #             explainer = shap.LinearExplainer(model, background)

            #             shap_values = explainer.shap_values(X_sp.toarray())

            #             st.write("SHAP values untuk sampel ini (summary bar):")
            #             fig2, ax2 = plt.subplots()
            #             shap.summary_plot(
            #                 shap_values,
            #                 features=X_sp.toarray(),
            #                 feature_names=vectorizer.get_feature_names_out(),
            #                 show=False
            #             )
            #             st.pyplot(fig2)
            #         except Exception as e:
            #             st.info(f"SHAP tidak dapat dijalankan: {e}")


# ============================================================
# 6) Tab 2 - Batch Prediction
# ============================================================
with tab2:
    st.subheader("📦 Batch Prediction dari CSV")
    st.markdown("**Format CSV minimal**: kolom `verified_reviews`, `variation`. Opsional: `feedback` untuk evaluasi.")

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        df_in = pd.read_csv(up)
        missing_cols = [c for c in ["verified_reviews", "variation"] if c not in df_in.columns]
        if missing_cols:
            st.error(f"Kolom wajib tidak ditemukan: {missing_cols}")
        else:
            # Prediksi
            with st.spinner("Melakukan prediksi batch..."):
                probs = predict_proba(df_in["verified_reviews"].astype(str).tolist(),
                                      df_in["variation"].astype(str).tolist())
                preds = predict_labels_from_proba(probs, threshold)
                df_out = df_in.copy()
                df_out["proba_positive"] = probs
                df_out["pred_feedback"] = preds

            st.success("Selesai memprediksi.")
            st.dataframe(df_out.head(20), use_container_width=True)

            # Visual distribusi probability & prediksi
            c1, c2 = st.columns(2)
            with c1:
                fig = plt.figure(figsize=(6,4))
                sns.histplot(df_out["proba_positive"], bins=30, kde=True)
                plt.title("Distribusi Probabilitas Positif")
                plt.xlabel("Proba(Positive)")
                st.pyplot(fig)
            with c2:
                fig = plt.figure(figsize=(6,4))
                sns.countplot(x="pred_feedback", data=df_out)
                plt.title("Distribusi Prediksi (0/1)")
                plt.xlabel("Predicted Feedback")
                st.pyplot(fig)

            # Download hasil
            st.download_button(
                "⬇️ Download Prediksi CSV",
                data=df_out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv"
            )

# ============================================================
# 7) Tab 3 - Evaluation & Explainability
# ============================================================
with tab3:
    st.subheader("📈 Evaluasi (opsional, jika ada label) & Explainability")
    st.markdown("Upload file CSV yang **mengandung label** `feedback` untuk melihat metrik evaluasi.")

    up_eval = st.file_uploader("Upload CSV berlabel (verified_reviews, variation, feedback)", type=["csv"], key="eval_csv")
    if up_eval is not None:
        df_eval = pd.read_csv(up_eval)
        needed = ["verified_reviews", "variation", "feedback"]
        miss = [c for c in needed if c not in df_eval.columns]
        if miss:
            st.error(f"Kolom wajib tidak ada: {miss}")
        else:
            # Bersihkan data NaN
            df_eval = df_eval.dropna(subset=["verified_reviews", "variation", "feedback"]).copy()
            y_true = df_eval["feedback"].astype(int).values
            probs = predict_proba(df_eval["verified_reviews"].astype(str).tolist(),
                                  df_eval["variation"].astype(str).tolist())
            preds = predict_labels_from_proba(probs, threshold)

            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

            acc = accuracy_score(y_true, preds)
            prec = precision_score(y_true, preds)
            rec = recall_score(y_true, preds)
            f1 = f1_score(y_true, preds)
            try:
                auc = roc_auc_score(y_true, probs)
            except Exception:
                auc = np.nan

            mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
            mcol1.metric("Accuracy", f"{acc:.3f}")
            mcol2.metric("Precision", f"{prec:.3f}")
            mcol3.metric("Recall", f"{rec:.3f}")
            mcol4.metric("F1-Score", f"{f1:.3f}")
            mcol5.metric("ROC-AUC", f"{auc:.3f}" if not np.isnan(auc) else "-")

            # Confusion Matrix
            cm = confusion_matrix(y_true, preds)
            fig = plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            st.pyplot(fig)

            # Calibration Plot (opsional)
            with st.expander("📉 Calibration Plot (opsional)"):
                try:
                    from sklearn.calibration import calibration_curve
                    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10)
                    fig2 = plt.figure(figsize=(5,4))
                    plt.plot(prob_pred, prob_true, marker="o")
                    plt.plot([0,1], [0,1], "--")
                    plt.xlabel("Predicted probability")
                    plt.ylabel("True probability in bin")
                    plt.title("Calibration Curve")
                    st.pyplot(fig2)
                except Exception as e:
                    st.info(f"Calibration tidak tersedia: {e}")

            # SHAP Global (opsional)
            # if SHAP_AVAILABLE and hasattr(model, "predict_proba"):
            #     with st.expander("🧠 SHAP Global Summary (opsional)"):
            #         try:
            #             # Gunakan subset kecil untuk performa
            #             sample_n = min(500, len(df_eval))
            #             X_sp, _ = transform_features(
            #                 df_eval["verified_reviews"].astype(str).iloc[:sample_n].tolist(),
            #                 df_eval["variation"].astype(str).iloc[:sample_n].tolist()
            #             )
            #             explainer = shap.Explainer(model)
            #             shap_values = explainer(X_sp)
            #             st.write("Ringkasan SHAP:")
            #             shap.summary_plot(shap_values, show=False)
            #             st.pyplot(bbox_inches='tight', dpi=150)
            #         except Exception as e:
            #             st.info(f"Gagal menampilkan SHAP: {e}")

            # # SHAP Global (opsional)
            # if SHAP_AVAILABLE and hasattr(model, "predict_proba"):
            #     with st.expander("🧠 SHAP Global Summary (opsional)"):
            #         try:
            # # Gunakan subset kecil untuk performa
            #             sample_n = min(500, len(df_eval))
            #             X_sp, _ = transform_features(
            #                 df_eval["verified_reviews"].astype(str).iloc[:sample_n].tolist(),
            #                 df_eval["variation"].astype(str).iloc[:sample_n].tolist()
            #             )

            #             background = vectorizer.transform(["neutral example"]).toarray()
            #             explainer = shap.LinearExplainer(model, background)

            #             shap_values = explainer.shap_values(X_sp.toarray())

            #             st.write("Ringkasan SHAP:")
            #             fig3, ax3 = plt.subplots()
            #             shap.summary_plot(
            #                 shap_values,
            #                 features=X_sp.toarray(),
            #                 feature_names=vectorizer.get_feature_names_out(),
            #                 show=False
            #             )
            #             st.pyplot(fig3)
            #         except Exception as e:
            #             st.info(f"Gagal menampilkan SHAP: {e}")


# ============================================================
# 8) Footer
# ============================================================
st.markdown("---")
st.caption("© Alexa Sentiment App — Streamlit demo. Gunakan bersama artefak model yang sudah dilatih sebelumnya.")
