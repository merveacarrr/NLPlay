# -*- coding: utf-8 -*-
"""
NLP Pipeline Test Script
Basit test için kullanılır
"""

from main import preprocess_texts, vectorize_texts, read_texts_from_file

def test_pipeline():
    print("=" * 50)
    print("NLP PIPELINE TEST")
    print("=" * 50)
    
    # Test metinleri
    test_texts = [
        "Natural Language Processing is amazing!",
        "AI and machine learning are transforming the world.",
        "Deep learning models can understand text very well."
    ]
    
    print(f"\n1. Test Metinleri ({len(test_texts)} adet):")
    for i, text in enumerate(test_texts, 1):
        print(f"   {i}. {text}")
    
    # Ön işleme
    print(f"\n2. Ön İşleme Başlatılıyor...")
    result = preprocess_texts(
        test_texts,
        language='english',
        do_tokenize=True,
        do_lowercase=True,
        remove_stopwords=True,
        do_lemmatization=True,
        use_pos_tagging=True
    )
    
    processed = result['processed_texts']
    stats = result['stats']
    
    print(f"\n3. İşlenmiş Metinler:")
    for i, text in enumerate(processed, 1):
        print(f"   {i}. {text}")
    
    print(f"\n4. İstatistikler:")
    for i, stat in enumerate(stats, 1):
        print(f"   Metin {i}: {stat['original_word_count']} → {stat['final_word_count']} kelime")
    
    # Vektörleştirme
    print(f"\n5. TF-IDF Vektörleştirme:")
    feature_names, matrix = vectorize_texts(processed, method='tfidf', max_features=20)
    
    print(f"   Feature Names ({len(feature_names)} adet):")
    print(f"   {feature_names}")
    
    print(f"\n   Matrix Shape: {matrix.shape}")
    print(f"   Matrix:")
    for i, row in enumerate(matrix):
        print(f"   Metin {i+1}: {row}")
    
    print(f"\n" + "=" * 50)
    print("TEST TAMAMLANDI!")
    print("=" * 50)

if __name__ == "__main__":
    test_pipeline() 