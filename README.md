# Doğal Dil İşleme Pipeline Projesi

Bu proje, metin verilerinin işlenmesi ve analizi için geliştirilmiş kapsamlı bir NLP (Doğal Dil İşleme) pipeline'ıdır. Özellikle akademik çalışmalar ve araştırma projeleri için tasarlanmıştır.
  ![2025-06-19_13-23-22](https://github.com/user-attachments/assets/4eca7a1b-0268-4975-a781-27a62c61711a)

## Proje Hakkında

Metin verilerinin makine öğrenmesi algoritmaları tarafından işlenebilmesi için geçirdiği temel dönüşüm adımlarını otomatikleştiren bu pipeline, araştırmacıların ve veri bilimcilerin işlerini kolaylaştırmayı amaçlamaktadır.

### Temel Özellikler

**Metin Ön İşleme:**
- Kelime bazlı tokenization
- Büyük/küçük harf standardizasyonu
- Gereksiz kelimelerin (stopwords) temizlenmesi
- Kelimelerin kök haline getirilmesi (lemmatization)

**Vektörleştirme:**
- TF-IDF (Term Frequency-Inverse Document Frequency) yöntemi
- Count Vectorizer alternatifi
- Özelleştirilebilir feature sayısı

**Görselleştirme ve Analiz:**
- İşlem istatistikleri
- Kelime frekans analizi
- Web tabanlı kullanıcı arayüzü

## Kurulum ve Gereksinimler

### Sistem Gereksinimleri
- Python 3.7 veya üzeri
- İnternet bağlantısı (ilk kurulum için)

### Kurulum Adımları

1. **Proje klasörüne geçiş:**
```bash
cd homework1
```

2. **Gerekli Python paketlerinin kurulumu:**
```bash
pip install -r requirements.txt
```

3. **NLTK verilerinin indirilmesi (opsiyonel):**
```bash
python setup_nltk.py
```

## Kullanım Kılavuzu

### Komut Satırı Arayüzü

En basit kullanım yöntemi, hazırlanmış test verileri ile pipeline'ı çalıştırmaktır:

```bash
python basic_pipeline.py
```

Bu komut, örnek metinler üzerinde tüm işlem adımlarını gerçekleştirir ve sonuçları terminal ekranında gösterir.

### Web Tabanlı Arayüz

Daha interaktif bir deneyim için web arayüzünü kullanabilirsiniz:

```bash
python simple_web_app.py
```

Uygulama başlatıldıktan sonra tarayıcınızda `http://localhost:5000` adresini açarak arayüze erişebilirsiniz.

#### Web Arayüzü Özellikleri

- **Metin Girişi:** Her satıra bir metin gelecek şekilde verilerinizi girebilirsiniz
- **Dil Seçimi:** İngilizce ve Türkçe dilleri desteklenmektedir
- **Gerçek Zamanlı İşleme:** Metinler anında işlenir ve sonuçlar gösterilir
- **Detaylı Raporlama:** İşlem adımları ve istatistikler ayrıntılı olarak sunulur
  
  ![2025-06-19_13-23-40](https://github.com/user-attachments/assets/f82c9fd6-7f30-4f9c-9988-901e905d8a5f)
  ![2025-06-19_13-23-59](https://github.com/user-attachments/assets/b18a6d3e-0cf8-44e6-9331-39e66ab8c73a)
  ![2025-06-19_13-24-27](https://github.com/user-attachments/assets/cf58b304-c96c-4ba5-a4f8-c4f19bfc5432)


### Özel Veri Kullanımı

Kendi metin verilerinizi işlemek için:

1. **Metin dosyası hazırlama:** Her satıra bir metin gelecek şekilde .txt dosyası oluşturun
2. **Pipeline'ı özelleştirme:** `basic_pipeline.py` dosyasındaki `corpus` listesini kendi verilerinizle değiştirin
3. **Parametre ayarlama:** Vektörleştirme için maksimum feature sayısını ihtiyacınıza göre ayarlayın

## Teknik Detaylar

### İşlem Adımları

1. **Tokenization:** Metinler boşluk karakterlerine göre kelimelere ayrılır
2. **Lowercasing:** Tüm karakterler küçük harfe dönüştürülür
3. **Temizlik:** Noktalama işaretleri ve sayılar kaldırılır
4. **Stopword Removal:** Dil için tanımlı gereksiz kelimeler çıkarılır
5. **Lemmatization:** Kelimeler kök formlarına dönüştürülür
6. **Vektörleştirme:** İşlenmiş metinler sayısal vektörlere çevrilir

### Çıktı Formatları

**Feature Names:** Kelime listesi, frekanslarına göre sıralanmış
**Matrix:** Her satır bir metni, her sütun bir kelimeyi temsil eder
**İstatistikler:** Her metin için orijinal kelime sayısı, çıkarılan stopword sayısı ve kalan kelime sayısı

## Dosya Yapısı

```
homework1/
├── basic_pipeline.py          # Ana pipeline fonksiyonları
├── simple_web_app.py          # Web uygulaması
├── main.py                    # Gelişmiş pipeline (NLTK gerekli)
├── web_app.py                 # Gelişmiş web uygulaması
├── templates/
│   ├── index.html             # Gelişmiş web arayüzü
│   └── simple_index.html      # Basit web arayüzü
├── sample_data.txt            # Örnek metin verileri
├── requirements.txt           # Python paket gereksinimleri
└── setup_nltk.py              # NLTK kurulum scripti
```

## Sorun Giderme

### Yaygın Sorunlar ve Çözümleri

**NumPy/NLTK Uyumluluk Sorunları:**
- `basic_pipeline.py` dosyasını kullanın (harici kütüphane gerektirmez)
- `simple_web_app.py` ile web arayüzünü deneyin

**Port Çakışması:**
- 5000 portu kullanımdaysa, `simple_web_app.py` dosyasında port numarasını değiştirin
- Alternatif olarak: `app.run(debug=True, host='0.0.0.0', port=5001)`

**Flask Kurulum Sorunu:**
```bash
pip install flask
```

### Performans Optimizasyonu

- Büyük veri setleri için `max_features` parametresini düşürün
- Bellek kullanımını azaltmak için metinleri küçük parçalara bölün
- Web arayüzü yerine komut satırı versiyonunu tercih edin

## Geliştirme ve Genişletme

### Yeni Özellik Ekleme

Pipeline'ı genişletmek için:

1. **Yeni ön işleme adımları:** `basic_preprocess_texts` fonksiyonuna ekleyin
2. **Farklı vektörleştirme yöntemleri:** `basic_vectorize` fonksiyonunu modifiye edin
3. **Görselleştirme:** Matplotlib veya Plotly ile grafik ekleyin

### Kod Yapısı

Proje modüler bir yapıda tasarlanmıştır:
- Her fonksiyon tek bir sorumluluğa sahiptir
- Parametreler esnek ve özelleştirilebilir
- Hata yönetimi kapsamlıdır

## Lisans ve Katkı

Bu proje eğitim amaçlı geliştirilmiştir. Akademik çalışmalarda kullanım için uygundur.

## 🙌 Katkıda Bulunanlar

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/meryemarpaci">
        <img src="https://github.com/meryemarpaci.png?size=100" width="100px;" alt="meryemarpaci"/>
        <br /><sub><b>meryemarpaci</b></sub>
      </a><br />
      <sub></sub>
    </td>
    <td align="center">
      <a href="https://github.com/merveacarrr">
        <img src="https://github.com/merveacarrr.png?size=100" width="100px;" alt="merveacarrr"/>
        <br /><sub><b>merveacarrr</b></sub>
      </a><br />
      <sub></sub>
    </td>
  </tr>
</table>



## İletişim ve Destek

Teknik sorularınız için:
- GitHub Issues kullanabilirsiniz
- Kod yorumlarını inceleyebilirsiniz
- Dokümantasyonu referans alabilirsiniz

---

**Not:** Bu proje, doğal dil işleme alanında temel kavramları öğrenmek ve uygulamak için tasarlanmıştır. Üretim ortamında kullanmadan önce gerekli testleri yapmanız önerilir. 
