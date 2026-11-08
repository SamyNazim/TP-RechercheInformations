
import os
import nltk
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

# ===== File paths =====

Output = r"C:\Users\Samy\Desktop\TP-RI\tp1\output"

InvertedIndexFile = os.path.join(Output, "InvertedIndex_Regex_Porter.txt")
WeightedIndexFile = os.path.join(Output, "InvertedIndex_Regex_Porter_weighted.txt")

# ===== Preprocessing =====

Stemmer = PorterStemmer()

try:
    StopWords = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    StopWords = set(stopwords.words("english"))

Pattern = r"(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+(?:[\.\,\-]\d+)*%?|/[A-Za-z]+|[A-Za-z]+"

Tokenizer = RegexpTokenizer(Pattern)


def PreprocessQuery(Query):

    Tokens = Tokenizer.tokenize(Query.lower())

    Tokens = [
        Token for Token in Tokens
        if Token not in StopWords
    ]

    Tokens = [
        Stemmer.stem(Token)
        for Token in Tokens
    ]

    return Tokens


# ===== Load classic inverted index =====

InvertedIndex = {}

with open(InvertedIndexFile, "r", encoding="utf-8") as File:

    for Line in File:

        Parts = Line.split()

        Term = Parts[0]
        Document = Parts[1]

        if Term not in InvertedIndex:
            InvertedIndex[Term] = set()

        InvertedIndex[Term].add(Document)


# ===== Load weighted inverted index =====

WeightedIndex = {}

with open(WeightedIndexFile, "r", encoding="utf-8") as File:

    for Line in File:

        Parts = Line.split()

        Term = Parts[0]
        Document = Parts[1]
        Weight = float(Parts[-1])

        if Term not in WeightedIndex:
            WeightedIndex[Term] = {}

        WeightedIndex[Term][Document] = Weight


# ===== Normalize weights to [0,1] if needed =====

MaxWeight = max(
    Weight
    for Documents in WeightedIndex.values()
    for Weight in Documents.values()
)

if MaxWeight > 1:

    for Term in WeightedIndex:

        for Document in WeightedIndex[Term]:

            WeightedIndex[Term][Document] /= MaxWeight


# ===== Build document vectors =====

Documents = set()

for Term in WeightedIndex:

    for Document in WeightedIndex[Term]:

        Documents.add(Document)


DocumentVectors = {}

for Document in Documents:

    DocumentVectors[Document] = {}

    for Term in WeightedIndex:

        if Document in WeightedIndex[Term]:

            DocumentVectors[Document][Term] = WeightedIndex[Term][Document]


# ===== Similarity measures =====

def InnerProduct(DocumentVector, QueryVector):

    Score = 0

    for Term in QueryVector:

        if Term in DocumentVector:

            Score += DocumentVector[Term] * QueryVector[Term]

    return Score


def CosineSimilarity(DocumentVector, QueryVector):

    Product = InnerProduct(DocumentVector, QueryVector)

    DocumentNorm = sum(
        Weight ** 2
        for Weight in DocumentVector.values()
    )

    QueryNorm = sum(
        Weight ** 2
        for Weight in QueryVector.values()
    )

    Denominator = np.sqrt(DocumentNorm * QueryNorm)

    if Denominator == 0:
        return 0

    return Product / Denominator


def JaccardSimilarity(DocumentVector, QueryVector):

    Product = InnerProduct(DocumentVector, QueryVector)

    DocumentNorm = sum(
        Weight ** 2
        for Weight in DocumentVector.values()
    )

    QueryNorm = sum(
        Weight ** 2
        for Weight in QueryVector.values()
    )

    Denominator = DocumentNorm + QueryNorm - Product

    if Denominator == 0:
        return 0

    return Product / Denominator


# ===== Test queries =====

Queries = {

    "q1": "large language models for information retrieval and ranking",

    "q2": "LLM for information retrieval and Ranking",

    "q3": "query Reformulation in information retrieval",

    "q4": "ranking Documents",

    "q5": "Optimizing recommendation systems with LLMs by leveraging item metadata"

}


# ===== Execute Vector Space Model =====

SimilarityMeasures = {

    "Inner Product": InnerProduct,

    "Cosine Similarity": CosineSimilarity,

    "Jaccard Similarity": JaccardSimilarity

}


for QueryName, Query in Queries.items():

    print("\n" + "=" * 70)
    print(QueryName, ":", Query)
    print("=" * 70)

    # Preprocess query

    QueryTerms = PreprocessQuery(Query)

    print("\nPreprocessed query:", QueryTerms)

    # Binary query vector

    QueryVector = {

        Term: 1

        for Term in set(QueryTerms)

    }

    print("\nQuery vector:", QueryVector)

    # Calculate similarity and rank documents

    for MeasureName, MeasureFunction in SimilarityMeasures.items():

        Results = []

        for Document in DocumentVectors:

            DocumentVector = DocumentVectors[Document]

            Score = MeasureFunction(
                DocumentVector,
                QueryVector
            )

            Results.append((Document, Score))

        # Sort documents by descending similarity

        Results.sort(
            key=lambda Item: (-Item[1], Item[0])
        )

        # Display results

        print("\n---", MeasureName, "---")

        for Rank, (Document, Score) in enumerate(Results, start=1):

            print(
                f"{Rank}. {Document} : {Score:.6f}"
            )
