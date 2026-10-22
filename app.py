import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model= pickle.load(open('model.pkl', 'rb'))

PS = PorterStemmer()

# Download the required NLTK dataset
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))
punctuation = set(string.punctuation)

def text_transform(text):
    text = text.lower() #lower case
    text = nltk.word_tokenize(text) #tokenize

    y = []
    # alphanumeric or remove special chars
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y.copy()
    y.clear()

    #remove stopwords and punctuation
    for i in text:
        if i not in stopwords and i not in punctuation:
            y.append(i)
    text = y.copy()
    y.clear()

    #stemming
    for i in text:
        y.append(PS.stem(i))
    text = y.copy()
    y.clear()
            
    return " ".join(text)


st.title("Email/SMS Spam Classifier Model")

inp_msg = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess the input
    trans_msg = text_transform(inp_msg)

    # 2. Vectorize
    vector_msg = tfidf.transform([trans_msg])
    # 3. Predict
    result = model.predict(vector_msg)[0]
    inp_msg = " "
    # 4. Display

    if result == 1:
        st.header("🚨 Spam Message Detected! ⚠️")
        
    else:
        st.header("✅ Not Spam — Safe Message 😊")

