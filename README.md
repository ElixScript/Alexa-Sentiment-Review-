# 🔊 Alexa Sentiment Review App

Project ini membangun **sistem analisis sentimen** untuk ulasan produk Amazon Alexa.  
Pipeline meliputi **EDA, preprocessing teks, modeling (ML), dan deployment dengan Streamlit**.

---

## 📂 Project Structure
```

Alexa-Sentiment-Review-/
│── amazon\_alexa.tsv                # Dataset utama
│── Alexa\_Sentiment\_Analysis.ipynb  # Notebook EDA + Modeling
│── app.py                          # Streamlit app
│── best\_model.pkl                  # Model terlatih (Logistic Regression)
│── tfidf\_vectorizer.pkl            # TF-IDF Vectorizer
│── onehot\_encoder.pkl              # OneHotEncoder untuk kolom variation
│── requirements.txt                # Daftar dependencies
│── **pycache**/                    # Cache python

````

---

## 📊 Exploratory Data Analysis (EDA)
Beberapa insight dari EDA:

- **Distribusi Rating**: mayoritas ulasan memiliki rating tinggi (5⭐).
- **Distribusi Feedback**: dataset imbalance → banyak review positif dibanding negatif.
- **Top Variation**: variasi *Black Dot* dan *Charcoal Fabric* paling dominan.
- **Panjang Review**: review yang panjang cenderung lebih positif.
- **Top N-Grams (Negatif)**:
  - Unigram: `echo`, `amazon`, `device`, `alexa`
  - Bigram: `echo dot`, `doesn work`, `sound quality`
- **WordCloud**:
  - Positif → *love, echo, great, music*
  - Negatif → *problem, device, time, sound, work*

📌 Model dipilih: **Logistic Regression**  
➡ karena recall pada kelas **negatif** tinggi (73%), sehingga lebih baik menangkap review negatif.

---

## ⚙️ Machine Learning Pipeline
1. **Preprocessing Teks**
   - Lowercasing
   - Remove angka, punctuation, emoji
   - Stopword removal
   - Lemmatization
   - Handle negation (`not good → not_good`)

2. **Feature Engineering**
   - **TF-IDF (unigram + bigram)**
   - **OneHotEncoding** pada kolom `variation`
   - Gabungan → `hstack([X_text, X_variation])`

3. **Handling Imbalance**
   - Menggunakan **SMOTE**

4. **Models Evaluated**
   - Logistic Regression ✅ (chosen)
   - Naive Bayes
   - Random Forest
   - XGBoost

---

## 🚀 Streamlit App (Deployment)

### Features:
- **Single Prediction** → ketik review + pilih variation → hasil prediksi sentimen
- **Batch Prediction** → upload CSV (`verified_reviews`, `variation`) → hasil prediksi massal
- **Evaluation & Explainability** → upload CSV berlabel untuk cek akurasi, precision, recall, F1, confusion matrix, calibration plot

📌 File: `app.py`

### Run Locally
```bash
# 1. Aktifkan virtual environment
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Jalankan Streamlit
streamlit run app.py
````

---

## 📦 Requirements

Semua dependencies ada di `requirements.txt`.
Beberapa library utama:

* `pandas, numpy, scikit-learn, imbalanced-learn`
* `matplotlib, seaborn, wordcloud`
* `xgboost`
* `nltk, emoji`
* `streamlit, shap (opsional)`
