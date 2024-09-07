from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def tokenization():
    
    text_data = pd.read_csv("text.txt", delimiter=";", names=["text", "label"])
    token_list = []
    
    for text in text_data["text"]:
        tokens = word_tokenize(text)
        tokens_lower = [token.lower() for token in tokens]
    
        stop_words = set(stopwords.words('english') + list(punctuation))
    
        filtered_tokens = [token for token in tokens_lower if token not in stop_words]

        lemetizer = WordNetLemmatizer()
        token_lemmatizer = [lemetizer.lemmatize(token) for token in filtered_tokens ]

        stemming = PorterStemmer()
        token_stemming = [stemming.stem(token) for token in token_lemmatizer]

        token_list.append(" ".join(token_stemming))

    vectorize = TfidfVectorizer()
    token_vector = vectorize.fit_transform(token_list)

    model = Word2Vec(sentences=token_list,min_count = 1)
    word = list(model.wv.index_to_key)
        



    


    



    

 




    


    
  
    
    
