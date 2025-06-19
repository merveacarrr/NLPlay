# -*- coding: utf-8 -*-
"""
Flask Web Arayüzü: NLP Ön İşleme ve Vektörleştirme Pipeline
Streamlit sorununu çözmek için alternatif web arayüzü
"""
from flask import Flask, render_template, request, jsonify, send_file
import os
import io
import base64
import matplotlib
matplotlib.use('Agg')  # GUI olmadan çalışması için
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import pandas as pd
from datetime import datetime

# Pipeline fonksiyonlarını içe aktar
from main import preprocess_texts, vectorize_texts, check_text_quality

app = Flask(__name__)

# Global değişkenler sonuçları saklamak için
current_results = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_texts():
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        language = data.get('language', 'english')
        do_tokenize = data.get('do_tokenize', True)
        do_lowercase = data.get('do_lowercase', True)
        remove_stopwords = data.get('remove_stopwords', True)
        do_lemmatization = data.get('do_lemmatization', True)
        use_pos_tagging = data.get('use_pos_tagging', True)
        vector_method = data.get('vector_method', 'tfidf')
        
        if not texts:
            return jsonify({'error': 'Metin girişi gerekli!'}), 400
        
        # Ön işleme
        result = preprocess_texts(
            texts,
            language=language,
            do_tokenize=do_tokenize,
            do_lowercase=do_lowercase,
            remove_stopwords=remove_stopwords,
            do_lemmatization=do_lemmatization,
            use_pos_tagging=use_pos_tagging
        )
        processed = result['processed_texts']
        stats = result['stats']
        
        # Vektörleştirme
        feature_names, matrix = vectorize_texts(processed, method=vector_method)
        
        # Uyarılar
        warnings = check_text_quality(stats)
        
        # Sonuçları sakla
        global current_results
        current_results = {
            'original_texts': texts,
            'processed_texts': processed,
            'stats': stats,
            'feature_names': feature_names.tolist(),
            'matrix': matrix.tolist(),
            'warnings': warnings,
            'vector_method': vector_method
        }
        
        return jsonify({
            'success': True,
            'message': f'{len(texts)} metin başarıyla işlendi',
            'stats': stats,
            'warnings': warnings,
            'feature_count': len(feature_names),
            'matrix_shape': matrix.shape
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_results')
def get_results():
    if not current_results:
        return jsonify({'error': 'Henüz işlenmiş veri yok!'}), 404
    
    return jsonify(current_results)

@app.route('/download_matrix')
def download_matrix():
    if not current_results:
        return jsonify({'error': 'Henüz işlenmiş veri yok!'}), 404
    
    try:
        # DataFrame oluştur
        df = pd.DataFrame(
            current_results['matrix'],
            columns=current_results['feature_names'],
            index=[f'Metin_{i+1}' for i in range(len(current_results['matrix']))]
        )
        
        # CSV olarak kaydet
        output = io.StringIO()
        df.to_csv(output, index=True, encoding='utf-8')
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'nlp_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_plots')
def generate_plots():
    if not current_results:
        return jsonify({'error': 'Henüz işlenmiş veri yok!'}), 404
    
    try:
        processed = current_results['processed_texts']
        feature_names = current_results['feature_names']
        matrix = np.array(current_results['matrix'])
        
        # WordCloud
        plt.figure(figsize=(10, 5))
        text = ' '.join(processed)
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud')
        
        # Base64'e çevir
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        wordcloud_b64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # Bar Chart
        plt.figure(figsize=(10, 5))
        word_scores = matrix.sum(axis=0)
        top_n = min(10, len(feature_names))
        top_idx = np.argsort(word_scores)[::-1][:top_n]
        plt.bar([feature_names[i] for i in top_idx], word_scores[top_idx])
        plt.title('En Sık Kelimeler')
        plt.xlabel('Kelime')
        plt.ylabel('Ağırlık')
        plt.xticks(rotation=45)
        
        # Base64'e çevir
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        barchart_b64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return jsonify({
            'wordcloud': wordcloud_b64,
            'barchart': barchart_b64
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Templates klasörü oluştur
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Static klasörü oluştur
    if not os.path.exists('static'):
        os.makedirs('static')
    
    print("Flask web uygulaması başlatılıyor...")
    print("Tarayıcınızda http://localhost:5000 adresini açın")
    app.run(debug=True, host='0.0.0.0', port=5000) 