from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import pandas as pd




ranking = defaultdict(int)

def tokenization(n):
    
    text_data = pd.read_csv("text.txt", delimiter=";", names=["text", "label"])
    token_list = []
    
    for text in text_data["text"]:
        tokens = word_tokenize(text)
        token_list.extend(tokens)
        
        assert n <= len(token_list), "O valor de n é maior que o número de tokens disponíveis."
    
        tokens_lower = [token.lower() for token in token_list]
    
    stop_words = set(stopwords.words('english') + list(punctuation))
    
    filtered_tokens = [token for token in tokens_lower if token not in stop_words]

    lemetizer = WordNetLemmatizer()
    token_lemmatizer = [lemetizer.lemmatize(token) for token in filtered_tokens ]

    stemming = PorterStemmer()
    token_stemming = [stemming.stem(token) for token in token_lemmatizer]

    vectorize = CountVectorizer()
    token_vector = vectorize.transform(token_stemming)

    print(token_vector.shape)
    print(token_vector.toarray())
    



    

 




    


    
  
    
    
