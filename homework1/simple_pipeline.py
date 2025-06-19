# -*- coding: utf-8 -*-
"""
Basit NLP Pipeline 
"""

import re
import nltk
from collections import Counter

# NLTK verilerini indir
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    print("NLTK verileri indirilemedi, devam ediliyor...")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def simple_preprocess_texts(texts, language='english'):
    """
    Basit ön işleme pipeline'ı
    """
    processed_texts = []
    stats = []
    
    lemmatizer = WordNetLemmatizer()
    
    for text in texts:
        orig_len = len(text.split())
        
        # 1. Lowercase
        text = text.lower()
        
        # 2. Noktalama ve sayıları temizle
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', ' ', text)
        
        # 3. Tokenization
        tokens = word_tokenize(text)
        
        # 4. Stopword temizliği
        stopword_count = 0
        try:
            sw = set(stopwords.words(language))
        except:
            sw = set()
        
        filtered_tokens = []
        for token in tokens:
            if token not in sw:
                filtered_tokens.append(token)
            else:
                stopword_count += 1
        
        # 5. Lemmatization
        lemmatized_tokens = []
        for token in filtered_tokens:
            lemmatized = lemmatizer.lemmatize(token)
            lemmatized_tokens.append(lemmatized)
        
        processed_texts.append(' '.join(lemmatized_tokens))
        
        stats.append({
            'original_word_count': orig_len,
            'stopwords_removed': stopword_count,
            'final_word_count': len(lemmatized_tokens)
        })
    
    return processed_texts, stats

def simple_vectorize(texts, method='tfidf'):
    
    #Basit vektörleştirme (TF-IDF benzeri)
    
    # Tüm kelimeleri topla
    all_words = []
    for text in texts:
        all_words.extend(text.split())
    
    # Kelime frekanslarını hesapla
    word_freq = Counter(all_words)
    
    # En sık kelimeleri al (feature names)
    feature_names = [word for word, freq in word_freq.most_common(20)]
    
    # Matrix oluştur
    matrix = []
    for text in texts:
        text_words = text.split()
        row = []
        for feature in feature_names:
            count = text_words.count(feature)
            row.append(count)
        matrix.append(row)
    
    return feature_names, matrix

def print_results(original_texts, processed_texts, stats, feature_names, matrix):
    #Sonuçları güzel bir şekilde yazdır
    
    print("=" * 60)
    print("NLP ÖN İŞLEME VE VEKTÖRLEŞTİRME PIPELINE")
    print("=" * 60)
    
    print(f"\n1. ORİJİNAL METİNLER ({len(original_texts)} adet):")
    print("-" * 40)
    for i, text in enumerate(original_texts, 1):
        print(f"{i}. {text}")
    
    print(f"\n2. ÖN İŞLEME ADIMLARI:")
    print("-" * 40)
    print("✓ Tokenization")
    print("✓ Lowercasing")
    print("✓ Stopword temizliği")
    print("✓ Lemmatization")
    
    print(f"\n3. İŞLENMİŞ METİNLER:")
    print("-" * 40)
    for i, text in enumerate(processed_texts, 1):
        print(f"{i}. {text}")
    
    print(f"\n4. METİN İSTATİSTİKLERİ:")
    print("-" * 40)
    for i, stat in enumerate(stats, 1):
        print(f"{i}. Metin: Toplam {stat['original_word_count']} kelime, {stat['stopwords_removed']} stopword çıkarıldı, {stat['final_word_count']} kelime kaldı.")
    
    print(f"\n5. VEKTÖRLEŞTİRME SONUÇLARI:")
    print("-" * 40)
    print(f"Feature Names ({len(feature_names)} adet):")
    print(feature_names)
    
    print(f"\nMatrix Shape: ({len(matrix)}, {len(feature_names)})")
    print("Matrix (her satır bir metin, her sütun bir kelime):")
    for i, row in enumerate(matrix):
        print(f"Metin {i+1}: {row}")
    
    print(f"\n" + "=" * 60)
    print("PIPELINE TAMAMLANDI!")
    print("=" * 60)

if __name__ == "__main__":
    # Test metinleri
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
    
    # Ön işleme
    processed_texts, stats = simple_preprocess_texts(corpus, language='english')
    
    # Vektörleştirme
    feature_names, matrix = simple_vectorize(processed_texts, method='tfidf')
    
    # Sonuçları yazdır
    print_results(corpus, processed_texts, stats, feature_names, matrix) 