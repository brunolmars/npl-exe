from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from chatterbot import ChatBot
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd




token_data = pd.read_csv("text.txt", delimiter=";", names=["text"])

def text_transformed(text_data):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english') + list(punctuation))
    token_list = []

    for text in text_data["text"]:
        tokens = word_tokenize(text)
        tokens_lower = [token.lower() for token in tokens]
        filtered_tokens = [token for token in tokens_lower if token not in stop_words]
        token_lemmatized = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        token_list.append(" ".join(token_lemmatized))

    return token_list

def label_words(token_list):
    return [
        token.replace("surprise", "1")
             .replace("happy", "1")
             .replace("sad", "0")
             .replace("angry", "0")
             .replace("love", "1")
             .replace("fear", "0")
             .replace("joy", "1")
             .replace("sadness", "0")
        for token in token_list
    ]


token_process = text_transformed(token_data)
print(token_process)

#Rotulação de palavras
token_enumerated = label_words(token_process)
print(token_enumerated)

#Vetorização
count_vectorizer = CountVectorizer()
token_vectorized = count_vectorizer.fit_transform(token_process)


parameters = {
    'max_features': ('auto', 'sqrt'),
    'n_estimators': [500, 1000, 1500],
    'max_depth': [5, 10, None],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [1, 2, 5, 10],
    'bootstrap': [True, False]
}


search = GridSearchCV(RandomForestClassifier(), parameters, cv=5, scoring='accuracy')
search.fit(token_vectorized, )  


best_rfc = RandomForestClassifier(**search.best_params_)
best_rfc.fit(token_vectorized, token_enumerated)


predictions = best_rfc.predict(token_vectorized)



#  chatbot









    


    



    

 




    


    
  
    
    
