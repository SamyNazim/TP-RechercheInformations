import os
import math
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
 
nltk.download('stopwords')
 
# EXERCISE 1 - I. Read the collection

collection_path = "collection"     
documents = {}
for filename in os.listdir(collection_path):
    if filename.endswith(".txt"):
        doc_id = filename.split(".")[0]         
        with open(os.path.join(collection_path, filename), "r", encoding="utf-8") as f:
            documents[doc_id] = f.read()
 
print("Documents read:", sorted(documents))
 
# Tools (tokenizer, stopwords, stemmers)

tokenizer = RegexpTokenizer(
    r'(?:[A-Za-z]\.)+'                  # Abbreviations like A.I.
    r'|\d+(?:[\.\,]\d+)*[A-Za-z%]*'     # Numbers: 2025, 5.0, 1.75B, 120.50M
    r'|\.{3}'                           # Ellipsis ...
    r'|[A-Za-z]+'                       # Simple words
)
 
stop_words = set(stopwords.words('english'))
Porter = PorterStemmer()
Lancaster = LancasterStemmer()
 
# EXERCISE 1 - II. Build the index

Index = {"Split_Porter": {}, "Split_Lancaster": {},
         "Regex_Porter": {}, "Regex_Lancaster": {}}
 
for doc_id in documents:
    text = documents[doc_id]
 
    # 1) Extract terms
    TermsSplit = text.split()
    TermsRegex = tokenizer.tokenize(text)
 
    # 2) Remove stop words
    SplitNoStop = [t for t in TermsSplit if t.lower() not in stop_words]
    RegexNoStop = [t for t in TermsRegex if t.lower() not in stop_words]
 
    # 3) Normalization
    Index["Split_Porter"][doc_id]    = [Porter.stem(t) for t in SplitNoStop]
    Index["Split_Lancaster"][doc_id] = [Lancaster.stem(t) for t in SplitNoStop]
    Index["Regex_Porter"][doc_id]    = [Porter.stem(t) for t in RegexNoStop]
    Index["Regex_Lancaster"][doc_id] = [Lancaster.stem(t) for t in RegexNoStop]
 
os.makedirs("output", exist_ok=True)
N = len(documents)      
 
for version in Index:
 
    # 4) Descriptor file: <Document number> <Term>
    # 5) Inverted index: <Term> <Document number>

    DocTerm = []        # list of (doc, term)
    for doc_id in sorted(Index[version]):
        for term in sorted(set(Index[version][doc_id])):
            DocTerm.append((doc_id, term))
 
    Inverted = sorted(DocTerm, key=lambda x: (x[1], x[0]))   # sort by term
 
    with open("output/DocTerm_" + version + ".txt", "w", encoding="utf-8") as f:
        for doc_id, term in DocTerm:
            f.write(doc_id + " " + term + "\n")
 
    with open("output/InvertedIndex_" + version + ".txt", "w", encoding="utf-8") as f:
        for doc_id, term in Inverted:
            f.write(term + " " + doc_id + "\n")
 
    # EXERCISE 2 - frequency + TF-IDF weight

    # frequency dictionary for each document
    Freq = {}           # Freq[doc_id][term] = frequency
    for doc_id in Index[version]:
        Freq[doc_id] = {}
        for term in Index[version][doc_id]:
            if term in Freq[doc_id]:
                Freq[doc_id][term] += 1
            else:
                Freq[doc_id][term] = 1
 
    # n_t = number of documents containing the term
    nt = {}
    for doc_id in Freq:
        for term in Freq[doc_id]:
            if term in nt:
                nt[term] += 1
            else:
                nt[term] = 1
 
    # weight(t,d) = freq(t,d) / max freq in d * log10(1 + N / n_t)
    Weight = {}         # Weight[(doc_id, term)] = weight
    for doc_id in Freq:
        max_freq = max(Freq[doc_id].values())
        for term in Freq[doc_id]:
            Weight[(doc_id, term)] = (Freq[doc_id][term] / max_freq) * math.log10(1 + N / nt[term])
 
    # Descriptor file: <Document number> <Term> <Frequency> <Weight>
    with open("output/DocTerm_" + version + "_weighted.txt", "w", encoding="utf-8") as f:
        for doc_id, term in DocTerm:
            f.write(doc_id + " " + term + " " + str(Freq[doc_id][term]) + " "
                    + str(round(Weight[(doc_id, term)], 4)) + "\n")
 
    # Inverted index: <Term> <Document number> <Frequency> <Weight>
    with open("output/InvertedIndex_" + version + "_weighted.txt", "w", encoding="utf-8") as f:
        for doc_id, term in Inverted:
            f.write(term + " " + doc_id + " " + str(Freq[doc_id][term]) + " "
                    + str(round(Weight[(doc_id, term)], 4)) + "\n")
 
print("Done. Files are in the 'output' folder.")
