from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.wsd import lesk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import spacy

token_data = pd.read_csv("text.txt", delimiter=";", names=["text"])
token_list = []

def text_trasformeted(text_data):
    global token_list
    for text in text_data["text"]:
            tokens = word_tokenize(text)
            tokens_lower = [token.lower() for token in tokens]
            stop_words = set(stopwords.words('english') + list(punctuation))
            filtered_tokens = [token for token in tokens_lower if token not in stop_words]
            lemetizer = WordNetLemmatizer()
            token_lemmatizer = [lemetizer.lemmatize(token) for token in filtered_tokens ]
            token_list.append(" ".join(token_lemmatizer))

    return token_list


def palavras_enumeradas(token_list):

    token_list = [token.replace("surprise", "1")
                  .replace("happy", "1")
                  .replace("sad", "0")
                  .replace("angry", "0")
                  .replace("love", "1")
                  .replace("fear", "0")
                  .replace("joy", "1")
                  .replace("sadness", "0") for token in token_list]

    return token_list

coutvectorize = CountVectorizer()
token_sent = lesk(token_process[0],"paris")
nlp = spacy.load("en_core_web_sm")
token_sent2 =nlp("".join(token_list[0]))
for token in token_sent2:
    print("{0}".format(token.head.text))

parameters = {'max_features': ('auto','sqrt'),
             'n_estimators': [500, 1000, 1500],
             'max_depth': [5, 10, None],
             'min_samples_split': [5, 10, 15],
             'min_samples_leaf': [1, 2, 5, 10],
             'bootstrap': [True, False]}

search = GridSearchCV(RandomForestClassifier(),parameters,cv=5,scoring='accuracy')

token_process = text_trasformeted(token_list)
print(token_list)

token_enumereted = palavras_enumeradas(token_list)
print(token_enumereted)

token_vectorize = coutvectorize.fit_transform(token_process)

token_vectorize2 = coutvectorize.transform(token_vectorize)

rfc = RandomForestClassifier(max_features=search.best_params_['max_features'],                                  max_depth=search.best_params_['max_depth'],
                                  n_estimators=search.best_params_['n_estimators'],                                      min_samples_split=search.best_params_['min_samples_split'],                                    min_samples_leaf=search.best_params_['min_samples_leaf'],
                                    bootstrap=search.best_params_['bootstrap'])
rfc.fit(token_vectorize2)

prediction = rfc.predict(token_vectorize2)








    


    



    

 




    


    
  
    
    
