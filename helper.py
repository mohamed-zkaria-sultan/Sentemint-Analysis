from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
import re
import pickle

tf = pickle.load(open("Art/tf.pkl",'rb'))


nltk.download('stopwords')
nltk.download('punkt_tab')

stop_words = stopwords.words('english')
stemmer = PorterStemmer()

def text_preprocessing(text):
  ## lower case
  text = text.lower()
  ## special charcter
  text = re.sub('[^a-zA-z]', ' ', text)
  ## Tokinzation
  text = word_tokenize(text)
  ## stopwords
  text = [word for word in text if word not in stop_words]
  ## lemmetization
  text = [stemmer.stem(word) for word in text]
  text = ' '.join(text)
  text = tf.transpose([text])
  return text