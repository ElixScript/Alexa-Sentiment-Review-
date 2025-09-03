# pages/3_Explainability.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Model Explainability", page_icon="🧠", layout="wide")
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

# Try SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ============================================================
# 2) UI Layout
# ============================================================
st.title("🧠 Model Explainability")
st.markdown(
    """
    Halaman ini membantu menjelaskan **mengapa model membuat prediksi tertentu**.  
    Kita gunakan 2 pendekatan:
    - **Feature Importance Global** dari TF-IDF + OneHot(Variation)
    - **SHAP Values** untuk interpretasi lokal & global (jika tersedia)
    """
)

tab1, tab2 = st.tabs(["📊 Global Feature Importance", "🔍 SHAP Explainability"])

# ============================================================
# 3) Tab 1 - Global Feature Importance
# ============================================================
with tab1:
    st.subheader("📊 Global Feature Importance")

    st.markdown("#### 1. Top Words dari TF-IDF")
    try:
        feature_names = vectorizer.get_feature_names_out()
    except Exception:
        feature_names = []

    # Gabungkan fitur TF-IDF + variation
    if feature_names != []:
        var_names = ohe.get_feature_names_out(["variation"])
        all_features = np.concatenate([feature_names, var_names])
    else:
        all_features = ohe.get_feature_names_out(["variation"])

    if hasattr(model, "coef_"):  # Logistic Regression, Naive Bayes
        coefs = model.coef_[0]
        coef_df = pd.DataFrame({"feature": all_features, "coef": coefs})
        coef_df["abs_coef"] = coef_df["coef"].abs()
        top_pos = coef_df.sort_values("coef", ascending=False).head(15)
        top_neg = coef_df.sort_values("coef", ascending=True).head(15)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top 15 Positive Words** (mengarah ke sentimen positif)")
            fig, ax = plt.subplots(figsize=(6,4))
            sns.barplot(x="coef", y="feature", data=top_pos, palette="Greens_r", ax=ax)
            plt.title("Positive Features")
            st.pyplot(fig)
        with col2:
            st.markdown("**Top 15 Negative Words** (mengarah ke sentimen negatif)")
            fig, ax = plt.subplots(figsize=(6,4))
            sns.barplot(x="coef", y="feature", data=top_neg, palette="Reds_r", ax=ax)
            plt.title("Negative Features")
            st.pyplot(fig)

    elif hasattr(model, "feature_importances_"):  # Random Forest, XGBoost
        importances = model.feature_importances_
        imp_df = pd.DataFrame({"feature": all_features, "importance": importances})
        top_imp = imp_df.sort_values("importance", ascending=False).head(20)

        st.markdown("**Top 20 Features** berdasarkan feature_importances_")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(x="importance", y="feature", data=top_imp, palette="Blues_r", ax=ax)
        plt.title("Feature Importances")
        st.pyplot(fig)

    else:
        st.warning("Model tidak menyediakan coef_ atau feature_importances_. Tidak bisa menampilkan importance.")

# ============================================================
# 4) Tab 2 - SHAP Explainability
# ============================================================
with tab2:
    st.subheader("🔍 SHAP Explainability")

    if not SHAP_AVAILABLE:
        st.warning("SHAP library tidak terinstal. Jalankan `pip install shap` untuk menggunakan fitur ini.")
    elif not hasattr(model, "predict_proba"):
        st.warning("Model tidak mendukung predict_proba sehingga SHAP tidak bisa digunakan.")
    else:
        st.markdown(
            """
            **SHAP** digunakan untuk:
            - Menjelaskan kontribusi setiap fitur pada prediksi individu
            - Menunjukkan fitur global yang paling mempengaruhi model
            """
        )
        # Input text untuk dianalisis
        sample_text = st.text_area("Masukkan 1 ulasan untuk dianalisis dengan SHAP:", 
                                   "The Alexa is amazing and responds quickly!")
        sample_var = st.text_input("Variation:", "Black")

        if st.button("Hitung SHAP untuk sampel ini"):
            from app import transform_features
            X_sp, clean_texts = transform_features([sample_text], [sample_var])
            explainer = shap.Explainer(model, feature_names=None)
            shap_values = explainer(X_sp)

            st.write("**SHAP Explanation untuk sampel ini**")
            fig = shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(bbox_inches="tight", dpi=120)

        with st.expander("📊 SHAP Global Summary (subset 200 sampel random)"):
            st.info("Gunakan dataset dengan kolom `verified_reviews` & `variation` untuk analisis global.")
            up = st.file_uploader("Upload CSV untuk analisis global", type="csv", key="shapcsv")
            if up is not None:
                df_in = pd.read_csv(up)
                needed = ["verified_reviews", "variation"]
                if all(c in df_in.columns for c in needed):
                    sample_df = df_in.sample(min(200, len(df_in)), random_state=42)
                    from app import transform_features
                    X_sp, _ = transform_features(sample_df["verified_reviews"].tolist(),
                                                 sample_df["variation"].tolist())
                    explainer = shap.Explainer(model, feature_names=None)
                    shap_values = explainer(X_sp)

                    st.write("**SHAP Summary Plot (global)**")
                    shap.summary_plot(shap_values, show=False)
                    st.pyplot(bbox_inches="tight", dpi=120)
                else:
                    st.error("Kolom wajib tidak lengkap: butuh verified_reviews & variation.")
