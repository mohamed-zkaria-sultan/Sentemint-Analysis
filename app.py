import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

lr = pickle.load(open('lr.pkl', 'rb'))
tf = pickle.load(open('tf.pkl', 'rb'))
dt = pickle.load(open('dt.pkl', 'rb'))
svc = pickle.load(open('svc.pkl', 'rb'))

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

stop_words = stopwords.words('english')
stop_words.remove('not')
stop_words.remove('no')

lemmatizer = WordNetLemmatizer()

analyzer = SentimentIntensityAnalyzer()

def text_preprocessing(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    text = [lemmatizer.lemmatize(word) for word in text]
    text = ' '.join(text)
    return text

def predict_feedback(text):
    processed_text = text_preprocessing(text)
    sentiment_score = analyzer.polarity_scores(processed_text)
    
    if sentiment_score['compound'] > 0:
        return "Positive"
    else:
        return "Negative"

st.title('Product Feedback Prediction')  

user_input = st.text_area("Enter your feedback:", "I don't like this product.")

if st.button('Predict'):
    prediction = predict_feedback(user_input)
    st.write(f"Predicted Feedback: {prediction}")
