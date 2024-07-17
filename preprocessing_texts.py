import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation


def preprocessing(text, to_lemmas=False, delete_unknown_words=False):

    text = text.lower()
    tokens = word_tokenize(text)

    set_stopwords = stopwords.words('russian')
    set_stopwords.extend(list(punctuation + '—–«»”“'))
    set_stopwords.extend(['..', '--', '...', "''", '``'])
    
    new_tokens = []   
    if to_lemmas or delete_unknown_words:
        import pymorphy3
        morph = pymorphy3.MorphAnalyzer()
            
    for token in tokens:
        if (token not in set_stopwords) and (not token.isdigit()):
            if to_lemmas and delete_unknown_words:
                token = morph.parse(token)[0]
                if token.is_known:
                    new_tokens.append(token.normal_form)
            elif to_lemmas:
                token = morph.parse(token)[0]
                new_tokens.append(token.normal_form)
            elif delete_unknown_words:
                token = morph.parse(token)[0]
                if token.is_known:
                    new_tokens.append(token)
            else:
                new_tokens.append(token)

    return new_tokens


def change_text(tokens, file_name, author, dirin, dirout):
    os.chdir('..')
    os.chdir('..')
    os.chdir(dirout)
    if not os.path.exists(author):
        os.mkdir(author)
    os.chdir(author)

    if len(tokens) > 2e4:
        for i in range(max_i := (len(tokens) // 10000)):
            with open('new_' + file_name[:-4] + f'_{i}' + '.txt', 'w', encoding='utf-8') as otf:
                if i + 1 < max_i:
                    print(' '.join(tokens[10000*i: 10000*(i + 1)]), file=otf)
                else:
                    print(' '.join(tokens[10000*i:]), file=otf)
    else:
        with open('new_' + file_name, 'w', encoding='utf-8') as otf:
            print(' '.join(tokens), file=otf)

    os.chdir('..')
    os.chdir('..')
    os.chdir(dirin)
    os.chdir(author)


def change_texts(dirin, dirout, to_lemmas=False, delete_unknown_words=False):
    if os.path.exists(dirout):
        import shutil
        shutil.rmtree(dirout)
    os.mkdir(dirout)

    os.chdir(dirin)
    for author in os.listdir():
        os.chdir(author)
        
        texts = os.listdir()
        for file_name in texts:
            with open(file_name, 'r', encoding='utf-8') as inf:
                text = inf.read()
            tokens = preprocessing(text, to_lemmas, delete_unknown_words)
            change_text(tokens, file_name, author, dirin, dirout)
                
        os.chdir('..')
    os.chdir('..')
