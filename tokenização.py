from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.collocations import *
from string import punctuation
from nltk.probability import FreqDist
from heapq import nlargest
from collections import defaultdict

ranking = defaultdict


def tokenization(n):

    text =  """France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches. Paris, its capital, is famed for its fashion houses, classical art museums including the Louvre and monuments like the Eiffel Tower. The country is also renowned for its wines and sophisticated cuisine. Lascaux’s ancient cave drawings, Lyon’s Roman theater and the vast Palace of Versailles attest to its rich history."""
    token = word_tokenize(text)

    assert n <= len(token)
    
    sent = [token.lower() for token in token]
    stop_words = set(stopwords.words('english') + list(punctuation))

    filtoken =[token for token in sent if token not in stop_words]

    

