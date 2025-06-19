# -*- coding: utf-8 -*-
"""
Basit Flask Web Uygulaması - NLP Pipeline
"""

from flask import Flask, render_template, request, jsonify
import os
from basic_pipeline import basic_preprocess_texts, basic_vectorize

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('simple_index.html')

@app.route('/process', methods=['POST'])
def process_texts():
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        language = data.get('language', 'english')
        
        if not texts:
            return jsonify({'error': 'Metin girişi gerekli!'}), 400
        
        # Ön işleme
        processed_texts, stats = basic_preprocess_texts(texts, language=language)
        
        # Vektörleştirme
        feature_names, matrix = basic_vectorize(processed_texts, max_features=20)
        
        return jsonify({
            'success': True,
            'message': f'{len(texts)} metin başarıyla işlendi',
            'original_texts': texts,
            'processed_texts': processed_texts,
            'stats': stats,
            'feature_names': feature_names,
            'matrix': matrix
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Templates klasörü oluştur
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    print("Basit Flask web uygulaması başlatılıyor...")
    print("Tarayıcınızda http://localhost:5000 adresini açın")
    app.run(debug=True, host='0.0.0.0', port=5000) 