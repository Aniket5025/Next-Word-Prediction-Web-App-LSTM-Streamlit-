import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tensorflow as tf

tf.keras.backend.clear_session()

@st.cache_resource
def load_lstm_model():
    model = load_model('next_word.h5', compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handler:
        tokenizer = pickle.load(handler)
    return tokenizer

model = load_lstm_model()
tokenizer = load_tokenizer()

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

st.title("Next Word Prediction With LSTM And Early Stopping")
input_text = st.text_input("Enter the sequence of Words", "To be or not to be that is the")
if st.button("Predict Next Word"):
    max_sequences_length = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequences_length)
    st.write(f'Next Word: {next_word}')
