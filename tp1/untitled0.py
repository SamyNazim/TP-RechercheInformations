import os
import math
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer

collection_path = "collection"     
documents = {}
for filename in os.listdir(collection_path):
    if filename.endswith(".txt"):
        doc_id = filename.split(".")[0]         
        with open(os.path.join(collection_path, filename), "r", encoding="utf-8") as f:
            documents[doc_id] = f.read()


            