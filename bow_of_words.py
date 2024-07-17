import os
import numpy as np
from collections import Counter


def get_dataset(indir):
    os.chdir(indir)

    texts = []
    authors = []
    for author in os.listdir():
        os.chdir(author)
        
        for file_name in os.listdir():
            with open(file_name, 'r', encoding='utf-8') as inf:
                text = inf.read()
            if '--' in text:
                print(file_name)
            texts.append(text)
            authors.append(author)
        
        os.chdir('..')
    os.chdir('..')

    return texts, authors


def text_to_bow(text, bow_vocabulary):
    vector = np.zeros(len(bow_vocabulary))
    counter = Counter(text.split())
    for i, token in enumerate(bow_vocabulary):
        if token in counter:
            vector[i] = counter[token]

    return np.array(vector, dtype=float)