import helper
import streamlit as st 
import pickle

model = pickle.load(open("artifacts/lr.pkl",'rb'))

text = st.text_input('enter your review')

text = helper.text_preprocessing(text)

if st.button("predict"):
    model.predict(text)
