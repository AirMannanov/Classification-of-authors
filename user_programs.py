import numpy as np
import pandas as pd
from collections import Counter

from preprocessing_texts import preprocessing
from parts_of_speech import get_list_parts_of_speech
from bow_of_words import get_dataset, text_to_bow

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec


def predict_author_by_parts_of_speech(file_name, model=None, to_lemmas=False, delete_unknown_words=False):
    with open(file_name, 'r', encoding='utf-8') as inf:
        text = inf.read()
    tokens = preprocessing(text, to_lemmas, delete_unknown_words)
    
    x_pred = get_list_parts_of_speech(tokens)
    x_pred = [i / len(tokens) for i in x_pred]

    labelencoder = LabelEncoder()
    
    dataset = pd.read_csv('parts_of_speech.csv')
    dataset = dataset.sample(frac=1)

    X_part_of_speech = np.array(dataset.iloc[:, :-1])
    y = labelencoder.fit_transform(dataset.iloc[:, -1]) 

    if model == None:
        model = RandomForestClassifier(n_estimators=100)
    
    model.fit(X_part_of_speech, y)
    y_pred = model.predict([x_pred])

    return labelencoder.inverse_transform(y_pred)[0]


def predict_author_by_bow_of_words(file_name, model=None, to_lemmas=False, delete_unknown_words=False):
    with open(file_name, 'r', encoding='utf-8') as inf:
        text = inf.read()
    tokens = preprocessing(text, to_lemmas, delete_unknown_words)

    texts, authors = get_dataset('new_authors') 
    counts = Counter(' '.join(texts).split())
    
    bow_vocabulary = [key for key, val in counts.most_common(100000)]
    X_bow = np.array([text_to_bow(text, bow_vocabulary) for text in texts])
    x_pred = text_to_bow(' '.join(tokens), bow_vocabulary)

    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(authors)

    if model == None:
        model = RandomForestClassifier(n_estimators=100)
        
    model.fit(X_bow, y)
    y_pred = model.predict([x_pred])

    return labelencoder.inverse_transform(y_pred)[0]
    
    
def predict_author_by_tfidf(file_name, model=None, to_lemmas=False, delete_unknown_words=False):
    with open(file_name, 'r', encoding='utf-8') as inf:
        text = inf.read()
    tokens = preprocessing(text, to_lemmas, delete_unknown_words)
    
    labelencoder = LabelEncoder()
    vectorizer = TfidfVectorizer()

    texts, authors = get_dataset('new_authors') 

    X_tfidf = vectorizer.fit_transform(texts)
    x_pred = vectorizer.transform([' '.join(tokens)])
    y = labelencoder.fit_transform(authors)

    if model == None:
        model = RandomForestClassifier(n_estimators=200)
    
    model.fit(X_tfidf, y)
    y_pred = model.predict(x_pred)
    
    return  labelencoder.inverse_transform(y_pred)[0]


def predict_author_by_word2vec(file_name, model=None, to_lemmas=False, delete_unknown_words=False): 
    with open(file_name, 'r', encoding='utf-8') as inf:
        text = inf.read()
    tokens_pred = preprocessing(text, to_lemmas, delete_unknown_words)
    
    texts, authors = get_dataset('new_authors')
    labelencoder = LabelEncoder()    
    y = labelencoder.fit_transform(authors)
    tokenized_texts = [text.split() for text in texts]
    w2c = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)
     
    text_embeddings = []
    for tokens in tokenized_texts:
        valid_tokens = [token for token in tokens if token in w2c.wv]
        if valid_tokens:
            text_embedding = w2c.wv[valid_tokens].mean(axis=0)
            text_embeddings.append(text_embedding)
        else:
            text_embeddings.append([0.0] * w2c.vector_size)
    
    valid_tokens = [token for token in tokens_pred if token in w2c.wv]
    x_pred = w2c.wv[valid_tokens].mean(axis=0)

    if model == None:
        model = RandomForestClassifier(n_estimators=200)
    
    model.fit(text_embeddings, y)
    y_pred = model.predict([x_pred])

    return  labelencoder.inverse_transform(y_pred)[0]