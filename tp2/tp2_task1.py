
import os
from nltk.stem import PorterStemmer

Output = r"C:\Users\Samy\Desktop\TP-RI\tp1\output"
InvertedIndexFile = os.path.join(Output, "InvertedIndex_Regex_Porter.txt")

InvertedIndex = {}

with open(InvertedIndexFile, "r", encoding="utf-8") as File:
    for Line in File:
        Term, Document = Line.split()

        if Term not in InvertedIndex:
            InvertedIndex[Term] = set()

        InvertedIndex[Term].add(Document)

Stemmer = PorterStemmer()

QueryDocs = InvertedIndex.get(Stemmer.stem("query"), set())
ReformulationDocs = InvertedIndex.get(Stemmer.stem("reformulation"), set())
LanguageDocs = InvertedIndex.get(Stemmer.stem("Language"), set())
ModelDocs = InvertedIndex.get(Stemmer.stem("model"), set())

Part1 = QueryDocs & ReformulationDocs
Part2 = LanguageDocs & ModelDocs

Results = Part1 | Part2

print("\n===== CLASSIC BOOLEAN MODEL =====")
print("Retrieved documents:", sorted(Results))

