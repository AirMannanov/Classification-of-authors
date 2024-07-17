import os
import pandas as pd
from nltk.tokenize import word_tokenize
import pymorphy3

def get_list_parts_of_speech(tokens):
    dict_part_of_speech = {
        'NOUN': 0, # имя существительное
        'ADJF': 1, # имя прилагательное (полное)
        'ADJS': 2, # имя прилагательное (краткое)
        'COMP': 3, # компаратив
        'VERB': 4, # глагол (личная форма)	
        'INFN': 5, # глагол (инфинитив)
        'PRTF': 6, # причастие (полное)
        'PRTS': 7, # причастие (краткое)	
        'GRND': 8, # деепричастие
        'NUMR': 9, # числительное	
        'ADVB': 10, # наречие
        'NPRO': 11, # местоимение-существительное
        'PRED': 12, # предикатив
        'PREP': 13, # предлог
        'CONJ': 14, # союз
        'PRCL': 15, # частица	
        'INTJ': 16, # междометие
    }

    count_parts_of_speech = [0 for i in range(len(dict_part_of_speech))]
    morph = pymorphy3.MorphAnalyzer()
    for token in tokens:
        if (word := morph.parse(token)[0]).is_known:
            count_parts_of_speech[dict_part_of_speech[word.tag.POS]] += 1

    return count_parts_of_speech


def create_csv_parts_of_speech(dirin, file_name):
    os.chdir(dirin)
    matrix = []

    for author in os.listdir():
        os.chdir(author)
        for name_file in os.listdir():
            with open(name_file, 'r', encoding='utf-8') as inf:
                tokens = word_tokenize(inf.read())
            count_parts_of_speech = get_list_parts_of_speech(tokens)
            count_parts_of_speech = [i / sum(count_parts_of_speech) for i in count_parts_of_speech]
            count_parts_of_speech.append(author)
            matrix.append(count_parts_of_speech)
        os.chdir('..')
    os.chdir('..')

    df = pd.DataFrame(matrix, columns=['NOUN', 'ADJF', 'ADJS', 
                                       'COMP', 'VERB', 'INFN', 
                                       'PRTF', 'PRTS', 'GRND', 
                                       'NUMR', 'ADVB', 'NPRO', 
                                       'PRED', 'PREP', 'CONJ', 
                                       'PRCL', 'INTJ', 'author'])
    
    df.to_csv(file_name, index=False)
