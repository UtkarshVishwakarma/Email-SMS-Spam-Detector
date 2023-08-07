import nltk
nltk.download('stopwords')
nltk.download('punkt')

import streamlit as st
import pickle
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import sklearn

ps = PorterStemmer()



def text_preprocessor(text):
    text_lower = text.lower()
    text_token = word_tokenize(text_lower)
    text_clean = [i for i in text_token if i.isalnum()]
    more_clean = [i for i in text_clean if i not in stopwords.words('english') and i not in string.punctuation]
    final = [ps.stem(i) for i in more_clean]
    return " ".join(final)

model = pickle.load(open('models/model.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

st.title("Email/SMS Spam Detector")

text = st.text_area("Enter The Message")

if st.button("Predict"):
    message = text_preprocessor(text)
    vector = vectorizer.transform([message])
    prediction = model.predict(vector)

    if prediction[0] == 0:
        prediction = 'Not Spam'

    else:
        prediction = 'Spam'

    st.header(prediction)
