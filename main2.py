import nltk
nltk.download('punkt')

text ="Natural Language Processing is a branch of artificial intelligence."

from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
print(tokens)

#stop-word removal
#is,the,a,of,and,to,in,for,with,on,at,by,from,up,about,into,over,after,since,before,under,out,again,further,then,once,here,there,when,where,why,how,all,any,other,some,such,no,not,only,own,same,so,than,too,very,s,t,can,will,just,don,should,now
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) #dosyadaki kelimelri oku
filtered_tokens = [word for word in tokens if word not in stop_words]
print(filtered_tokens)

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("running",pos="v"))

#pos tagging - part of speech tagging
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag

pos_tags = pos_tag(filtered_tokens)
print(pos_tags)

#Named Entity Recognition
nltk.download('maxent_ne_chunker')
nltk.download("words")

#https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
from nltk import ne_chunk
tree = ne_chunk(pos_tags)
print(tree)

#lowercasing
text = "Natural Language Processing is a branch of artificial intelligence."
text = text.lower()
print(text)

import re
text = re.sub(r'[^\w\s]','',text)
print(text)

text = re.sub(r'\d+',' ',text)#sayılar ve özel karakterler
print(text)

#vectorize etmek
#bag of word
corpus = ["Natural Language Processing is a branch of artificial intelligence.",
          "I love natural language processing.",
          "Language processing is fun.",
          "Langıage models are used in natural language processing."]

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
print(X.toarray())

#tf-ıdf 
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer2 = TfidfVectorizer()
X2 = vectorizer2.fit_transform(corpus)

print(vectorizer2.get_feature_names_out())
print(X2.toarray())

import re
from collections import Counter

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
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    stats = []
    processed_texts = []
    if language == 'turkish':
        try:
            import spacy
            nlp = spacy.blank('tr')
        except ImportError:
            nlp = None
    else:
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
            elif language == 'turkish' and nlp is not None:
                doc = nlp(' '.join(tokens))
                tokens = [token.lemma_ for token in doc]
        processed_texts.append(' '.join(tokens))
        stats.append({
            'original_word_count': orig_len,
            'stopwords_removed': stopword_count,
            'final_word_count': len(tokens)
        })
    return {'processed_texts': processed_texts, 'stats': stats}

# Örnek kullanım:
# result = preprocess_texts([
#     "Artificial Intelligence is the future.",
#     "AI is changing the world."
# ], language='english', do_tokenize=True, do_lowercase=True, remove_stopwords=True, do_lemmatization=True, use_pos_tagging=True)
# print(result)

#fonksiyon kodlama pipline tokenization/lowercasting stopwords temizliği lemmatization tf-ıdf vektörleştirme feature isimlerini ve aarrayi ekrana yazdır