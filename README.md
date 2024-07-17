# project-sudoku

## Запус программы

1. Загрузить classification.ipynb в google colab при этом добавить оба zip-файла texts.zip и programs.zip
2. Локальный запуск classification.ipynb возможен при условии установки модулей numpy, pandas, matplotlib, nltk, pymorphy3  и sklearn.

## Структура проекта

Важными частями проекта являются:

- директория `author/`, содержащая тексты, разбитые по авторам;
- вспомогательная директория `new_authors/`, содержащая предобработанные текста, разбитые по авторам;
- основной файл `classification.ipynb`;
- вспомогательные файлы `parts_of_speech.csv`, `parts_of_speech.py`, `bow_of_words.py`, `preprocessing_texts.py`, `user_programs.py`;
- файлы для проверки работы классификатора `testing_1.txt`, `testing_2.txt`, `testing_3.txt`;
- zip-файлы для загрузки в google colab `texts.zip`, `programs.zip`;
- файл `README.md` с базовым описанием проекта и указаниями по сборке и запуску.
