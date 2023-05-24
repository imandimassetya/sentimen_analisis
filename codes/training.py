import pandas as pd
from io import StringIO
import re
import nltk
import csv
import string
string.punctuation
from nltk import word_tokenize 
from nltk.probability import FreqDist
nltk.download('punkt')
nltk.download('brown')
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from keras.preprocessing.text import one_hot, Tokenizer
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, GlobalMaxPool1D, Embedding, Conv1D, LSTM, SpatialDropout1D
from keras import layers
from keras.utils import pad_sequences

def preprocessesing(text):
    def remove_tweet_special(text):
        # remove tab, new line, ans back slice
        text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
        # remove non ASCII (emoticon, chinese word, .etc)
        text = text.encode('ascii', 'replace').decode('ascii')
        # remove mention, link, hashtag
        text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
        # remove incomplete URL
        return text.replace("http://", " ").replace("https://", " ")

    #remove number
    def remove_number(text):
        return  re.sub(r"\d+", "", text)

    #defining the function to remove punctuation
    def remove_punctuation(text):
        punctuationfree="".join([i for i in text if i not in string.punctuation])
        return punctuationfree

    #remove whitespace leading & trailing
    def remove_whitespace_LT(text):
        return text.strip()

    #remove multiple whitespace into single whitespace
    def remove_whitespace_multiple(text):
        return re.sub('\s+',' ',text)

    # remove single char
    def remove_singl_char(text):
        return re.sub(r"\b[a-zA-Z]\b", "", text)

    # NLTK word tokenize 
    def word_tokenize_wrapper(text, language="indonesia"):
        return word_tokenize(text)

    # NLTK calc frequency distribution
    def freqDist_wrapper(text):
        return FreqDist(text)

    # # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def stemming(text):
        text = [stemmer.stem(word) for word in text]
        return text

    # Join token
    def join_token(text):
        text = ' '.join(text)
        return text

    # ----------------------- get stopword from Sastrawi ------------------------------------
    # get stopword indonesia
    stop_factory = StopWordRemoverFactory()

    # ---------------------------- manualy add stopword  ------------------------------------
    more_stopword = ["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                    'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                    'gak', 'ga', 'gk', 'krn', 'nya', 'nih', 'sih', 
                    'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                    'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                    'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                    '&amp', 'yah', 'user']
    # ---------------------------------------------------------------------------------------
    # append additional stopword

    list_stopwords = stop_factory.get_stop_words()+more_stopword
    stopword = stop_factory.create_stop_word_remover()

    # convert list to dictionary
    list_stopwords = set(list_stopwords)

    #remove stopword pada list token
    def stopwords_removal(words):
        return [word for word in words if word not in list_stopwords]

    # ---------------------------------------------------------------------------------------

    kamus = pd.read_csv(r'D:\WFH\BOOTCAMP\Challenge platinum\data\new_kamusalay.csv', encoding = 'latin-1', sep=',', header=None)
    if kamus is not None:  
        # Can be used wherever a "file-like" object is accepted:
        ids_before = kamus
        ids_before = ids_before.set_index(0).to_dict()
        ids = dict(ele for sub in ids_before.values() for ele in sub.items())
    
    text = text.apply(remove_punctuation)
    text = text.apply(remove_tweet_special)
    text = text.apply(remove_number)
    text = text.apply(remove_whitespace_LT)
    text = text.apply(remove_whitespace_multiple)
    text = text.apply(remove_singl_char)
    text = text.apply(word_tokenize_wrapper)

    # Normalized with dictionary
    def normalized_term(text):
        return [ids[term] if term in ids else term for term in text]

    text = text.apply(normalized_term)
    text = text.apply(stopwords_removal)
    text = text.apply(stemming)
    
    return text



def training(df):
    df = pd.DataFrame(df)
    df = df.rename(columns={0 : 'text', 1 : 'label'})
    df = df.groupby(['label']).apply(lambda x: x.sample(n=500, random_state=42))
    df = df.drop(index='neutral')
    df['label'] = df['label'].apply(lambda score: 0 if score=='positive' else 1 if score=='negative' else 2)
    text = preprocessesing(df['text'])
    vec = TfidfVectorizer().fit(text)
    vec_transform = vec.transform(text)
    X = text
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # def flatten_list(nested_list):
    #     flattened_list = []
    #     for item in nested_list:
    #         if isinstance(item, list):
    #             flattened_list.extend(flatten_list(item))
    #         else:
    #             flattened_list.append(item)
    #     return flattened_list
    # token = flatten_list(token)
    word_tokenizer = Tokenizer()
    word_tokenizer.fit_on_texts(X_train)
    vocab_length = 5000
    X_train = word_tokenizer.texts_to_sequences(X_train)
    X_test = word_tokenizer.texts_to_sequences(X_test)
    maxlen = 100
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    return(X_train, X_test, y_train, y_test, vocab_length, maxlen)