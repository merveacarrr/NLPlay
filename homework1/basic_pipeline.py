# -*- coding: utf-8 -*-
"""
Temel NLP Pipeline - NLTK olmadan çalışır
"""

import re
from collections import Counter

def basic_tokenize(text):
    #Basit tokenization
    return text.split()

def basic_lowercase(text):
    #Küçük harfe çevir
    return text.lower()

def basic_clean(text):
    #Noktalama ve sayıları temizle
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', ' ', text)
    return text

def basic_remove_stopwords(tokens, language='english'):
    #Basit stopword temizliği"
    if language == 'english':
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
            'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
            'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more',
            'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call',
            'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get',
            'come', 'made', 'may', 'part', 'i', 'am', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'
        }
    else:
        stopwords = set()
    
    return [token for token in tokens if token not in stopwords]

def basic_lemmatize(tokens):
    #Basit lemmatization
    # Basit kurallar
    lemmatized = []
    for token in tokens:
        # -ing -> -e (running -> run)
        if token.endswith('ing'):
            token = token[:-3]
        # -ed -> -e (learned -> learn)
        elif token.endswith('ed'):
            token = token[:-2]
        # -s -> - (books -> book)
        elif token.endswith('s') and len(token) > 3:
            token = token[:-1]
        lemmatized.append(token)
    return lemmatized

def basic_preprocess_texts(texts, language='english'):
    
    #Temel ön işleme pipeline'ı
    
    processed_texts = []
    stats = []
    
    for text in texts:
        orig_len = len(text.split())
        
        # 1. Lowercase
        text = basic_lowercase(text)
        
        # 2. Temizlik
        text = basic_clean(text)
        
        # 3. Tokenization
        tokens = basic_tokenize(text)
        
        # 4. Stopword temizliği
        stopword_count = len(tokens)
        tokens = basic_remove_stopwords(tokens, language)
        stopword_count -= len(tokens)
        
        # 5. Lemmatization
        tokens = basic_lemmatize(tokens)
        
        processed_texts.append(' '.join(tokens))
        
        stats.append({
            'original_word_count': orig_len,
            'stopwords_removed': stopword_count,
            'final_word_count': len(tokens)
        })
    
    return processed_texts, stats

def basic_vectorize(texts, max_features=20):
    
    #Temel vektörleştirme
    
    # Tüm kelimeleri topla
    all_words = []
    for text in texts:
        all_words.extend(text.split())
    
    # Kelime frekanslarını hesapla
    word_freq = Counter(all_words)
    
    # En sık kelimeleri al (feature names)
    feature_names = [word for word, freq in word_freq.most_common(max_features)]
    
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
    
    print("=" * 60)
    print("NLP ÖN İŞLEME VE VEKTÖRLEŞTİRME PIPELINE")
    print("=" * 60)
    
    print(f"\n1. ORİJİNAL METİNLER ({len(original_texts)} adet):")
    print("-" * 40)
    for i, text in enumerate(original_texts, 1):
        print(f"{i}. {text}")
    
    print(f"\n2. ÖN İŞLEME ADIMLARI:")
    print("-" * 40)
    print("✓ Tokenization (basit)")
    print("✓ Lowercasing")
    print("✓ Stopword temizliği (basit)")
    print("✓ Lemmatization (basit)")
    
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
    processed_texts, stats = basic_preprocess_texts(corpus, language='english')
    
    # Vektörleştirme
    feature_names, matrix = basic_vectorize(processed_texts, max_features=20)
    
    # Sonuçları yazdır
    print_results(corpus, processed_texts, stats, feature_names, matrix) 