# -*- coding: utf-8 -*-
"""
NLP Ön İşleme ve Vektörleştirme Pipeline'ı
- Dinamik ön işleme (tokenization, lowercasing, stopword removal, lemmatization, POS tagging)
- Türkçe ve İngilizce destek
- TF-IDF ve CountVectorizer vektörleştirme
- WordCloud ve bar chart görselleştirme
- İstatistiksel özet ve uyarı sistemi
- Dosyadan okuma (txt/csv)
- Kapsamlı yorum satırları
"""

import os
import re
import sys
import csv
import warnings
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# NLTK ve opsiyonel spaCy importları
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Vektörleştirme için sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Türkçe lemmatization için spaCy (isteğe bağlı)
try:
    import spacy
    nlp_tr = spacy.blank('tr')
except ImportError:
    nlp_tr = None
    warnings.warn("spaCy ve Türkçe model yüklü değil, Türkçe lemmatization devre dışı.")

# ----------------------
# Dosyadan okuma fonksiyonu
# ----------------------
def read_texts_from_file(filepath):
    """txt veya csv dosyasından metinleri okur."""
    texts = []
    if filepath.endswith('.txt'):
        with open(filepath, encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    elif filepath.endswith('.csv'):
        with open(filepath, encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                texts.extend([cell.strip() for cell in row if cell.strip()])
    else:
        raise ValueError('Desteklenmeyen dosya formatı!')
    return texts

# ----------------------
# Dinamik Ön İşleme Pipeline'ı
# ----------------------
def preprocess_texts(
    texts,
    language='english',
    do_tokenize=True,
    do_lowercase=True,
    remove_stopwords=True,
    do_lemmatization=True,
    use_pos_tagging=True
):
    """
    texts: list of str
    language: 'english' or 'turkish'
    Returns: dict with processed_texts, stats
    """
    stats = []
    processed_texts = []
    if language == 'turkish' and nlp_tr is None:
        warnings.warn("Türkçe lemmatization için spaCy gerekli!")
    lemmatizer = WordNetLemmatizer()
    for text in texts:
        orig_len = len(text.split())
        # Lowercase
        if do_lowercase:
            text = text.lower()
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', ' ', text)
        # Tokenize
        if do_tokenize:
            tokens = word_tokenize(text, language='turkish' if language=='turkish' else 'english')
        else:
            tokens = text.split()
        # Remove stopwords
        stopword_count = 0
        if remove_stopwords:
            try:
                sw = set(stopwords.words(language))
            except:
                sw = set()
            tokens_ = []
            for t in tokens:
                if t not in sw:
                    tokens_.append(t)
                else:
                    stopword_count += 1
            tokens = tokens_
        # Lemmatization
        if do_lemmatization:
            if language == 'english':
                if use_pos_tagging:
                    from nltk import pos_tag
                    def get_wordnet_pos(treebank_tag):
                        if treebank_tag.startswith('J'):
                            return 'a'
                        elif treebank_tag.startswith('V'):
                            return 'v'
                        elif treebank_tag.startswith('N'):
                            return 'n'
                        elif treebank_tag.startswith('R'):
                            return 'r'
                        else:
                            return 'n'
                    pos_tags = pos_tag(tokens)
                    tokens = [lemmatizer.lemmatize(w, get_wordnet_pos(p)) for w, p in pos_tags]
                else:
                    tokens = [lemmatizer.lemmatize(w) for w in tokens]
            elif language == 'turkish' and nlp_tr is not None:
                doc = nlp_tr(' '.join(tokens))
                tokens = [token.lemma_ for token in doc]
        processed_texts.append(' '.join(tokens))
        stats.append({
            'original_word_count': orig_len,
            'stopwords_removed': stopword_count,
            'final_word_count': len(tokens)
        })
    return {'processed_texts': processed_texts, 'stats': stats}

# ----------------------
# Vektörleştirme Fonksiyonu
# ----------------------
def vectorize_texts(texts, method='tfidf', max_features=30):
    """
    method: 'tfidf' | 'count'
    Returns: feature_names, matrix
    """
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=max_features)
    elif method == 'count':
        vectorizer = CountVectorizer(max_features=max_features)
    else:
        raise ValueError('method tfidf veya count olmalı!')
    X = vectorizer.fit_transform(texts)
    return vectorizer.get_feature_names_out(), X.toarray()

# ----------------------
# Görselleştirme Fonksiyonları
# ----------------------
def plot_wordcloud(texts, title='Word Cloud'):
    """WordCloud görselleştirmesi."""
    text = ' '.join(texts)
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

def plot_top_words_bar(feature_names, matrix, top_n=10, title='En Sık Kelimeler'):
    """Bar chart ile en yüksek ağırlıklı kelimeler."""
    import numpy as np
    word_scores = matrix.sum(axis=0)
    top_idx = np.argsort(word_scores)[::-1][:top_n]
    plt.figure(figsize=(10,5))
    plt.bar([feature_names[i] for i in top_idx], word_scores[top_idx])
    plt.title(title)
    plt.xlabel('Kelime')
    plt.ylabel('Ağırlık')
    plt.show()

# ----------------------
# Uyarı Sistemi
# ----------------------
def check_text_quality(stats, min_words=3):
    """Çok kısa veya anlamlı kelime kalmayan metinler için uyarı üretir."""
    warnings = []
    for i, s in enumerate(stats):
        if s['final_word_count'] < min_words:
            warnings.append(f"Uyarı: {i+1}. metinde çok az anlamlı kelime kaldı!")
    return warnings

# ----------------------
# Ana Çalışma Akışı (örnek)
# ----------------------
if __name__ == '__main__':
    print("=" * 60)
    print("NLP ÖN İŞLEME VE VEKTÖRLEŞTİRME PIPELINE")
    print("=" * 60)
    
    # 1. Metinleri hazırla veya dosyadan oku
    corpus = [
        "Artificial Intelligence is the future.",
        "AI is changing the world.",
        "AI is a branch of computer science.",
        "Machine learning is a subset of AI.",
        "Deep learning enables powerful AI applications.",
        "Natural language processing is a field of AI.",
        "AI impacts many industries.",
        "Ethics in AI is important.",
        "AI can automate tasks.",
        "AI systems learn from data."
    ]
    # Alternatif: corpus = read_texts_from_file('veri.txt')

    print(f"\n1. ORİJİNAL METİNLER ({len(corpus)} adet):")
    print("-" * 40)
    for i, text in enumerate(corpus, 1):
        print(f"{i}. {text}")

    # 2. Ön işleme
    print(f"\n2. ÖN İŞLEME ADIMLARI:")
    print("-" * 40)
    print("✓ Tokenization")
    print("✓ Lowercasing") 
    print("✓ Stopword temizliği")
    print("✓ Lemmatization (POS tagging ile)")
    
    result = preprocess_texts(
        corpus,
        language='english',
        do_tokenize=True,
        do_lowercase=True,
        remove_stopwords=True,
        do_lemmatization=True,
        use_pos_tagging=True
    )
    processed = result['processed_texts']
    stats = result['stats']

    print(f"\n3. İŞLENMİŞ METİNLER:")
    print("-" * 40)
    for i, text in enumerate(processed, 1):
        print(f"{i}. {text}")

    # 3. İstatistiksel özet ve uyarılar
    print(f"\n4. METİN İSTATİSTİKLERİ:")
    print("-" * 40)
    for i, s in enumerate(stats):
        print(f"{i+1}. Metin: Toplam {s['original_word_count']} kelime, {s['stopwords_removed']} stopword çıkarıldı, {s['final_word_count']} kelime kaldı.")
    
    warnings_list = check_text_quality(stats)
    if warnings_list:
        print(f"\nUYARILAR:")
        print("-" * 40)
        for w in warnings_list:
            print(f"⚠ {w}")

    # 4. Vektörleştirme (TF-IDF)
    print(f"\n5. TF-IDF VEKTÖRLEŞTİRME:")
    print("-" * 40)
    feature_names, matrix = vectorize_texts(processed, method='tfidf')
    
    print(f"Feature Names ({len(feature_names)} adet):")
    print(feature_names)
    
    print(f"\nTF-IDF Matrix Shape: {matrix.shape}")
    print("Matrix (her satır bir metin, her sütun bir kelime):")
    print(matrix)

    # 5. Görselleştirme
    print(f"\n6. GÖRSELLEŞTİRME:")
    print("-" * 40)
    print("WordCloud ve Bar Chart grafikleri açılıyor...")
    
    plot_wordcloud(processed, title='TF-IDF Sonrası Word Cloud')
    plot_top_words_bar(feature_names, matrix, top_n=10, title='TF-IDF En Sık Kelimeler')

    # 6. Alternatif: CountVectorizer ile vektörleştirme
    print(f"\n7. COUNT VECTORIZER İLE KARŞILAŞTIRMA:")
    print("-" * 40)
    feature_names2, matrix2 = vectorize_texts(processed, method='count')
    
    print(f"CountVectorizer Feature Names ({len(feature_names2)} adet):")
    print(feature_names2)
    
    print(f"\nCountVectorizer Matrix Shape: {matrix2.shape}")
    print("Matrix:")
    print(matrix2)
    
    plot_top_words_bar(feature_names2, matrix2, top_n=10, title='CountVectorizer En Sık Kelimeler')

    print(f"\n" + "=" * 60)
    print("PIPELINE TAMAMLANDI!")
    print("=" * 60)
