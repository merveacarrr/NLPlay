# -*- coding: utf-8 -*-
"""
Streamlit Web Arayüzü: NLP Ön İşleme ve Vektörleştirme Pipeline
"""
import streamlit as st
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np

# Pipeline fonksiyonlarını içe aktar
from main import preprocess_texts, vectorize_texts, plot_wordcloud, plot_top_words_bar, check_text_quality, read_texts_from_file

st.set_page_config(page_title="NLP Ön İşleme Pipeline", layout="wide")
st.title("NLP Ön İşleme ve Vektörleştirme Pipeline")
st.write("""
Bu uygulama ile metinlerinizi dinamik olarak ön işleyebilir, vektörleştirebilir ve sonuçları görselleştirebilirsiniz.
""")

# --- Sol panel: Ayarlar ---
st.sidebar.header("Ayarlar")
language = st.sidebar.selectbox("Dil", ["english", "turkish"])
do_tokenize = st.sidebar.checkbox("Tokenization", value=True)
do_lowercase = st.sidebar.checkbox("Lowercase", value=True)
remove_stopwords = st.sidebar.checkbox("Stopword Temizliği", value=True)
do_lemmatization = st.sidebar.checkbox("Lemmatization", value=True)
use_pos_tagging = st.sidebar.checkbox("POS Tagging ile Lemmatization", value=True)
vector_method = st.sidebar.selectbox("Vektörleştirme Yöntemi", ["tfidf", "count"])

# --- Metin girişi veya dosya yükleme ---
st.subheader("Metin Girişi veya Dosya Yükleme")
input_mode = st.radio("Giriş Yöntemi", ["Metin Kutusu", "Dosya Yükle"])
corpus = []
if input_mode == "Metin Kutusu":
    user_text = st.text_area("Her satıra bir metin gelecek şekilde metinlerinizi girin:", height=200)
    if user_text.strip():
        corpus = [line.strip() for line in user_text.split('\n') if line.strip()]
elif input_mode == "Dosya Yükle":
    uploaded_file = st.file_uploader("TXT veya CSV dosyası yükleyin", type=["txt", "csv"])
    if uploaded_file:
        content = uploaded_file.read().decode("utf-8")
        if uploaded_file.name.endswith(".txt"):
            corpus = [line.strip() for line in content.split('\n') if line.strip()]
        elif uploaded_file.name.endswith(".csv"):
            import csv
            reader = csv.reader(io.StringIO(content))
            for row in reader:
                corpus.extend([cell.strip() for cell in row if cell.strip()])

if corpus:
    st.success(f"{len(corpus)} metin yüklendi.")
    # --- Ön işleme ---
    result = preprocess_texts(
        corpus,
        language=language,
        do_tokenize=do_tokenize,
        do_lowercase=do_lowercase,
        remove_stopwords=remove_stopwords,
        do_lemmatization=do_lemmatization,
        use_pos_tagging=use_pos_tagging
    )
    processed = result['processed_texts']
    stats = result['stats']

    # --- İstatistikler ve uyarılar ---
    st.subheader("Metin İstatistikleri")
    for i, s in enumerate(stats):
        st.write(f"{i+1}. Metin: Toplam {s['original_word_count']} kelime, {s['stopwords_removed']} stopword çıkarıldı, {s['final_word_count']} kelime kaldı.")
    for w in check_text_quality(stats):
        st.warning(w)

    # --- İşlenmiş metinler ---
    with st.expander("İşlenmiş Metinler (Tüm Adımlar Sonrası)"):
        for i, t in enumerate(processed):
            st.write(f"{i+1}. {t}")

    # --- Vektörleştirme ---
    st.subheader("Vektörleştirme Sonuçları")
    feature_names, matrix = vectorize_texts(processed, method=vector_method)
    st.write(f"**{vector_method.upper()} Feature Names:**", feature_names)
    st.write(f"**{vector_method.upper()} Matrix:**")
    st.dataframe(matrix)

    # --- Görselleştirme ---
    st.subheader("Görselleştirme")
    # WordCloud
    st.write("**Word Cloud**")
    wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(processed))
    fig_wc, ax_wc = plt.subplots(figsize=(10,5))
    ax_wc.imshow(wc, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)
    # Bar chart
    st.write("**En Sık Kelimeler (Bar Chart)**")
    word_scores = matrix.sum(axis=0)
    top_n = min(10, len(feature_names))
    top_idx = np.argsort(word_scores)[::-1][:top_n]
    fig_bar, ax_bar = plt.subplots(figsize=(10,5))
    ax_bar.bar([feature_names[i] for i in top_idx], word_scores[top_idx])
    ax_bar.set_title('En Sık Kelimeler')
    ax_bar.set_xlabel('Kelime')
    ax_bar.set_ylabel('Ağırlık')
    st.pyplot(fig_bar)
else:
    st.info("Lütfen metin girin veya dosya yükleyin.") 