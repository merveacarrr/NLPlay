# -*- coding: utf-8 -*-
"""
NLTK Setup Script
Gerekli NLTK verilerini indirir
"""

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_data():
    #NLTK verilerini indir
    print("NLTK verileri indiriliyor...")
    
    # Gerekli NLTK verileri
    nltk_data = [
        'punkt',
        'stopwords', 
        'wordnet',
        'omw-1.4',
        'averaged_perceptron_tagger'
    ]
    
    for data in nltk_data:
        try:
            print(f"İndiriliyor: {data}")
            nltk.download(data, quiet=False)
            print(f"✓ {data} başarıyla indirildi")
        except Exception as e:
            print(f"✗ {data} indirilemedi: {e}")
    
    print("\nNLTK kurulumu tamamlandı!")

if __name__ == "__main__":
    download_nltk_data() 