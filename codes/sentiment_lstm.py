from tensorflow import keras
from keras.preprocessing.text import one_hot, Tokenizer
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, GlobalMaxPool1D, Embedding, Conv1D, LSTM, SpatialDropout1D
from keras import layers
from keras.backend import clear_session
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.utils import pad_sequences

def analyze_sentiment_lstm(vocab_length, maxlen, X_train, y_train, X_test, y_test):
    embedding_dim = 50

    lstm_model = Sequential()
    lstm_model.add(layers.Embedding(input_dim=vocab_length, 
                            output_dim=embedding_dim, 
                            input_length=maxlen))
    lstm_model.add(LSTM(50, return_sequences = True))
    lstm_model.add(Dropout(0.1))
    lstm_model.add(layers.GlobalMaxPool1D())
    lstm_model.add(Dropout(0.1))
    lstm_model.add(Dense(1, activation='sigmoid'))
    lstm_model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['acc'])
    lstm_model.summary()

    history = lstm_model.fit(X_train, y_train,
                        epochs=10,
                        verbose=False,
                        validation_data=(X_test, y_test),
                        batch_size=8)
    clear_session()
    
    loss, accuracy = lstm_model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = lstm_model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))   
    return lstm_model, history

def prediction_lstm(text, lstm_model):
    # vec = TfidfVectorizer().fit(text)
    # vec_transform = vec.transform(text)

    X_predict = text

    word_tokenizer = Tokenizer()
    word_tokenizer.fit_on_texts(X_predict)

    # Adding 1 to store dimensions for words for which no pretrained word embeddings exist

    X_predict = word_tokenizer.texts_to_sequences(X_predict)
    X_predict = pad_sequences(X_predict, padding='post', maxlen=100)
    predict = lstm_model.predict(X_predict, verbose=0)
    
    result = np.round(predict, decimals=0).astype(int)
    return result

def prediction_text_lstm(text, lstm_model):
    X_predict = text

    word_tokenizer = Tokenizer()
    word_tokenizer.fit_on_texts(X_predict)

    # Adding 1 to store dimensions for words for which no pretrained word embeddings exist

    X_predict = word_tokenizer.texts_to_sequences(X_predict)
    X_predict = pad_sequences(X_predict, padding='post', maxlen=100)
    predict = lstm_model.predict(X_predict, verbose=0)

    result = np.round(predict, decimals=0).astype(int)
    return result