# import streamlit as st

# st.set_page_config(
#     page_title="Hello",
#     page_icon="👋",
# )

# st.write("# Selamat datang di halaman utama teks processing dan teks cleaning! 👋")

# st.sidebar.success("Pilih metode input diatas")

# st.markdown(
#     """
#     Pada halaman ini kamu dapat melakukan teks processing dan
#     teks cleaning dengan pilihan metode input.
#     **👈 Pilih metode input di samping** untuk melakukan teks processing dan teks cleaning.
#     ### Pilihan metode input :
#     - **Input From Text Area :** input teks kamu pada teks box yang tersedia
#     - **Input From Uploaded File :** upload file teks dengan _extension csv_
    
#     Jangan lupa untuk upload file _new_kamusalay.csv_ sebagai dictionary kamu ya !
# """
# )


import streamlit as st
from codes.sentiment_nn import analyze_sentiment_nn
from codes.sentiment_lstm import analyze_sentiment_lstm
from codes.sentiment_nn import prediction_nn
from codes.sentiment_lstm import prediction_lstm
from codes.sentiment_nn import prediction_text_nn
from codes.sentiment_lstm import prediction_text_lstm
from codes.training import preprocessesing
from tensorflow import keras
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from keras.preprocessing.text import one_hot, Tokenizer

# Fungsi utama untuk menjalankan aplikasi Streamlit
def main():
    st.title("Analisis Sentimen dengan Neural Network dan LSTM")
    
    # Memilih model untuk analisis sentimen
    model = st.selectbox("Pilih Model", ["Neural Network", "LSTM"])
    
    # Mengambil input teks atau upload file
    option = st.selectbox("Pilih Opsi", ["Input Teks", "Upload File"])
    
    if option == "Input Teks":
        text = st.text_area("Masukkan teks")
        text = pd.Series(text)
        text = preprocessesing(text)
        # Menganalisis sentimen berdasarkan model yang dipilih
        if st.button("Model"):
            if model == "Neural Network":
                model = keras.models.load_model("model_nn.h5")
                result = prediction_text_nn(text, model)
                result = np.round(result, decimals=0).astype(int)
            else:
                lstm_model = keras.models.load_model("model_lstm.h5")
                result = prediction_text_lstm(text, lstm_model)
                result = np.round(result, decimals=0).astype(int)
            st.write("Hasil Analisis Sentimen:")
            if result == 0:
                result = "positif"
            else:
                result = "negatif"
            st.write("Sentimennya adalah :", result)
    
    else:
        file = st.file_uploader("Upload File")
        
        # Menganalisis sentimen berdasarkan model yang dipilih
        if st.button("Model"):
            if file is not None:
                df_predict = pd.read_csv(file)
                df_predict.drop(columns='No', inplace=True)
                df_predict['content'] = df_predict['content'].astype(str)
                df_predict['label_true'] = df_predict['Label'].apply(lambda score: 0 if score=='Positif' else 1 if score=='Negatif' else 2)
                text = preprocessesing(df_predict['content'])
                if model == "Neural Network":
                    model = keras.models.load_model("model_nn.h5")
                    result = prediction_nn(text, model)
                    df_predict['prediction'] = np.round(result, decimals=0).astype(int)
                    df_predict['prediction'] = df_predict['prediction'].replace({0 : 'Positif', 1: 'Negatif'})
                else:
                    lstm_model = keras.models.load_model("model_lstm.h5")
                    result = prediction_lstm(text, lstm_model)
                    df_predict['prediction'] = np.round(result, decimals=0).astype(int)
                    df_predict['prediction'] = df_predict['prediction'].replace({0 : 'Positif', 1: 'Negatif'})
                
                st.write("Hasil Analisis Sentimen:")
                st.write(df_predict[['content', 'prediction']])
                
                # Create and generate a word cloud image:
                tokens = str([token for sublist in text for token in sublist]).replace(" ", "").replace("'", "")
                wordcloud = WordCloud(background_color='white', colormap='autumn_r').generate(tokens)

                # Display the generated image:
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.show()
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()
                
                count = df_predict['prediction'].value_counts().sort_index()
                labels = count.index
                sizes = count.values
                fig, ax = plt.subplots()
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Mengatur proporsi lingkaran
                plt.title('Pie Chart')

                # Menampilkan pie chart menggunakan Streamlit
                st.pyplot(fig)

if __name__ == "__main__":
    main()
