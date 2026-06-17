import streamlit as st
import pickle
import numpy as np
import re
import string
import os

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Clickbait Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# LOAD MODEL & VECTORIZER
# ============================================================
@st.cache_resource
def load_model():
    model_path  = 'models/best_model.pkl'
    tfidf_path  = 'models/tfidf_vectorizer.pkl'
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(tfidf_path, 'rb') as f:
            tfidf = pickle.load(f)
        return model, tfidf
    except Exception as e:
        st.error(f'Error loading model: {e}')
        return None, None

# ============================================================
# PREPROCESSING
# ============================================================
INDO_STOPWORDS = set([
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk',
    'pada', 'adalah', 'dalam', 'tidak', 'akan', 'juga', 'atau', 'ada',
    'bisa', 'telah', 'sudah', 'saat', 'agar', 'oleh', 'karena', 'lebih',
    'atas', 'lagi', 'para', 'bagi', 'nya', 'kita', 'mereka', 'kami',
    'saya', 'kamu', 'dia', 'ia', 'anda', 'belum', 'pun', 'hanya',
    'jika', 'maka', 'namun', 'tetapi', 'tapi', 'seperti', 'dapat',
    'hal', 'tersebut', 'hingga', 'bahwa', 'antara', 'serta',
    'setelah', 'sebelum', 'ketika', 'tentang', 'sejak', 'semua', 'sebuah'
])

CLICKBAIT_TRIGGER_WORDS = [
    'ternyata', 'bikin', 'kaget', 'wow', 'heboh', 'viral', 'mengejutkan',
    'tak disangka', 'tak terduga', 'luar biasa', 'fantastis', 'akhirnya',
    'terbongkar', 'terkuak', 'terungkap', 'rahasianya', 'begini', 'inilah',
    'kenapa', 'mengapa', 'simak', 'yuk', 'mau tau', 'penasaran',
    'jangan sampai', 'harus tau', 'wajib tau', 'fakta', 'tips', 'trik',
    'cara', 'mudah', 'cepat', 'gratis', 'terbaik', 'terpopuler', 'paling',
    'terbaru', 'eksklusif', 'spesial'
]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.split() if t not in INDO_STOPWORDS and len(t) > 1]
    return ' '.join(tokens)

def extract_features(text):
    text_str   = str(text)
    text_lower = text_str.lower()
    word_count         = len(text_str.split())
    exclamation_count  = text_str.count('!')
    question_count     = text_str.count('?')
    ellipsis_count     = text_str.count('...')
    caps_count         = sum(1 for w in text_str.split() if w.isupper() and len(w) > 1)
    number_count       = len(re.findall(r'\d+', text_str))
    trigger_count      = sum(1 for tw in CLICKBAIT_TRIGGER_WORDS if tw in text_lower)
    punct_count        = sum(1 for c in text_str if c in string.punctuation)
    punct_ratio        = punct_count / max(len(text_str), 1)
    return np.array([[word_count, exclamation_count, question_count,
                      ellipsis_count, caps_count, number_count,
                      trigger_count, round(punct_ratio, 4)]])

def predict_clickbait(text, model, tfidf):
    from scipy.sparse import hstack, csr_matrix
    cleaned      = clean_text(text)
    tfidf_feat   = tfidf.transform([cleaned])
    ling_feat    = csr_matrix(extract_features(text))
    combined     = hstack([tfidf_feat, ling_feat])
    prediction   = model.predict(combined)[0]
    try:
        probability = model.predict_proba(combined)[0]
        confidence  = probability[prediction]
    except:
        confidence = None
    return prediction, confidence

# ============================================================
# UI
# ============================================================
# Header
st.markdown("""
<div style='text-align:center; padding: 20px 0'>
    <h1>🔍 Indonesian Clickbait Detector</h1>
    <p style='color: gray; font-size: 16px'>
        Deteksi apakah judul berita online termasuk <b>clickbait</b> atau bukan
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Load model
model, tfidf = load_model()

if model is None or tfidf is None:
    st.error("❌ Model tidak berhasil dimuat. Pastikan file model ada di folder `models/`")
    st.stop()

# Input
st.subheader("📝 Masukkan Judul Berita")
user_input = st.text_area(
    label="Judul Berita",
    placeholder="Contoh: Viral! Aksi Pria Ini Bikin Netizen Kaget, Ternyata Ini yang Dilakukannya...",
    height=100,
    label_visibility="collapsed"
)

# Contoh judul
st.markdown("**💡 Coba contoh judul:**")
col1, col2 = st.columns(2)
with col1:
    if st.button("🔴 Contoh Clickbait", use_container_width=True):
        user_input = "VIRAL! Ternyata Begini Cara Artis Ini Sembunyikan Hubungan Gelapnya, Bikin Kaget!!!"
with col2:
    if st.button("🟢 Contoh Non-Clickbait", use_container_width=True):
        user_input = "Bank Indonesia Pertahankan Suku Bunga Acuan di Level 6 Persen"

st.divider()

# Predict button
if st.button("🔍 Deteksi Sekarang", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("⚠️ Masukkan judul berita terlebih dahulu!")
    else:
        with st.spinner("Menganalisis..."):
            prediction, confidence = predict_clickbait(user_input, model, tfidf)

        # Hasil
        if prediction == 1:
            st.markdown("""
            <div style='background:#ffebee; border-left:5px solid #e74c3c;
                        padding:20px; border-radius:8px; margin:10px 0'>
                <h2 style='color:#e74c3c; margin:0'>⚠️ CLICKBAIT</h2>
                <p style='margin:5px 0; color:#666'>Judul ini terindikasi sebagai clickbait</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#e8f5e9; border-left:5px solid #2ecc71;
                        padding:20px; border-radius:8px; margin:10px 0'>
                <h2 style='color:#2ecc71; margin:0'>✅ NON-CLICKBAIT</h2>
                <p style='margin:5px 0; color:#666'>Judul ini tidak terindikasi sebagai clickbait</p>
            </div>
            """, unsafe_allow_html=True)

        if confidence is not None:
            st.metric("🎯 Confidence", f"{confidence*100:.1f}%")
            st.progress(float(confidence))

        # Analisis fitur linguistik
        with st.expander("🔬 Lihat Analisis Fitur Linguistik"):
            feats = extract_features(user_input)[0]
            feat_names = ['Jumlah Kata', 'Tanda Seru (!)', 'Tanda Tanya (?)',
                          'Titik-titik (...)', 'Kata KAPITAL', 'Angka',
                          'Kata Pemicu Clickbait', 'Rasio Tanda Baca']
            feat_df = {
                'Fitur': feat_names,
                'Nilai': feats.tolist(),
                'Indikator': ['⚠️' if (i == 1 and v > 0) or (i == 2 and v > 0) or
                               (i == 6 and v > 0) or (i == 4 and v > 0)
                               else '✅' for i, v in enumerate(feats)]
            }
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

st.divider()

# Info model
with st.expander("ℹ️ Tentang Model"):
    st.markdown("""
    **Model:** Stacking Ensemble (Naive Bayes + SVM + XGBoost) dengan Logistic Regression sebagai meta-learner

    **Fitur:**
    - TF-IDF (10,000 fitur, unigram + bigram)
    - 8 Fitur Linguistik: panjang judul, tanda baca, kata pemicu emosi, dll

    **Dataset:** CLICK-ID — 14,699 judul berita Indonesia dari 11 portal berita

    **Optimasi:** Bayesian Optimization (Optuna) dengan 50 trials

    **Peneliti:** Humaida Athifah | NIM: 241730040
    """)
