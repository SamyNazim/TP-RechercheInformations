print("Extraction using split():")

Text = """In 2025, Dr. A.I. Research-Lab released GPT-5.0 with
1.75B parameters - costing $120.50M! It outperforms BERT, ROBERTa, etc... and supports 100+ Languages.
"""

Terms = Text.split()
print("Extracted terms:\n", Terms)

print("-----------------------------------------------")

print("Extraction using NLTK (Regular Expressions)")

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(
r'(?:[A-Za-z].)+'                    # Abbreviations like A.I.
r'|[A-Za-z]+(?:-[A-Za-z]+)+'          # Hyphenated words
r'|[A-Za-z]+-\d+(?:.\d+)?'           # Words followed by numbers, e.g. GPT-5.0
r'|\d+(?:.\d+)?[A-Za-z]+'            # Numbers followed by letters, e.g. 1.75B
r'|[A-Za-z]+@\d+(?:.\d+)?'            # Words with @ and numbers
r'|\d+(?:[.,]\d+)*%?'                # Numbers, decimals, percentages
r'|[A-Za-z]+'                         # Simple alphabetic words
r'|[^\w\s]'                           # Other punctuation
)

Terms = tokenizer.tokenize(Text)
print("Extracted tokens:\n", Terms)

print("-----------------------------------------------")

print("Stopwords Removal")

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
TermsNoStop = [t for t in Terms if t.lower() not in stop_words]

print("Terms without stop words:\n", TermsNoStop)

print("------------------------------------------------")

print("Porter Stemmer")

from nltk.stem import PorterStemmer

Porter = PorterStemmer()
TermsPorter = [Porter.stem(t) for t in TermsNoStop]
print(TermsPorter)

print("---------------------------------------------")

print("Lancaster Stemmer")

from nltk.stem import LancasterStemmer

Lancaster = LancasterStemmer()
TermsLancaster = [Lancaster.stem(t) for t in TermsNoStop]
print(TermsLancaster)


print("------------------------------------------------")

from nltk.stem import LancasterStemmer

Lancaster = LancasterStemmer()
TermsLancaster = [Lancaster.stem(t) for t in TermsNoStop]
print(TermsLancaster)

print("------------------------------------------------")





