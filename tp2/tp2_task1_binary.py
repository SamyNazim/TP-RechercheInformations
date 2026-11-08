import os
from nltk.stem import PorterStemmer

Output = r"C:\Users\Samy\Desktop\TP-RI\tp1\output"
InvertedIndexFile = os.path.join(Output, "InvertedIndex_Regex_Porter.txt")
WeightedIndexFile = os.path.join(Output, "InvertedIndex_Regex_Porter_weighted.txt")

P = 2

Stemmer = PorterStemmer()

# ===== Classic boolean index =====
InvertedIndex = {}

with open(InvertedIndexFile, "r", encoding="utf-8") as File:
    for Line in File:
        Term, Document = Line.split()

        if Term not in InvertedIndex:
            InvertedIndex[Term] = set()

        InvertedIndex[Term].add(Document)

# ===== Weighted index: Term Document Weight =====
WeightedIndex = {}

with open(WeightedIndexFile, "r", encoding="utf-8") as File:
    for Line in File:
        Parts = Line.split()

        Term = Parts[0]
        Document = Parts[1]
        Weight = Parts[-1]

        if Term not in WeightedIndex:
            WeightedIndex[Term] = {}

        WeightedIndex[Term][Document] = float(Weight)

# Normalize weights to [0,1] if needed
MaxWeight = max(w for Docs in WeightedIndex.values() for w in Docs.values())

if MaxWeight > 1:
    for Term in WeightedIndex:
        for Document in WeightedIndex[Term]:
            WeightedIndex[Term][Document] /= MaxWeight

# ===== Classic boolean model =====
QueryDocs = InvertedIndex.get(Stemmer.stem("query"), set())
ReformulationDocs = InvertedIndex.get(Stemmer.stem("reformulation"), set())
LanguageDocs = InvertedIndex.get(Stemmer.stem("Language"), set())
ModelDocs = InvertedIndex.get(Stemmer.stem("model"), set())

Part1 = QueryDocs & ReformulationDocs
Part2 = LanguageDocs & ModelDocs

Results = Part1 | Part2

print("\n===== CLASSIC BOOLEAN MODEL =====")
print("Retrieved documents:", sorted(Results))

# ===== Fuzzy and extended models =====
AllDocs = {Document for Docs in WeightedIndex.values() for Document in Docs}

FuzzyScores = {}
ExtendedScores = {}

for Document in AllDocs:
    Query = WeightedIndex.get(Stemmer.stem("query"), {}).get(Document, 0)
    Reformulation = WeightedIndex.get(Stemmer.stem("reformulation"), {}).get(Document, 0)
    Language = WeightedIndex.get(Stemmer.stem("Language"), {}).get(Document, 0)
    Model = WeightedIndex.get(Stemmer.stem("model"), {}).get(Document, 0)

    # Fuzzy: AND = min, OR = max
    FuzzyScores[Document] = max(min(Query, Reformulation), min(Language, Model))

    # Extended p-norm: (query AND reformulation) OR (Language AND model)
    And1 = 1 - (((1 - Query) ** P + (1 - Reformulation) ** P) / 2) ** (1 / P)
    And2 = 1 - (((1 - Language) ** P + (1 - Model) ** P) / 2) ** (1 / P)
    ExtendedScores[Document] = ((And1 ** P + And2 ** P) / 2) ** (1 / P)

for Title, Scores in [("FUZZY BOOLEAN MODEL", FuzzyScores),
                      (f"EXTENDED BOOLEAN MODEL (p={P})", ExtendedScores)]:
    print(f"\n===== {Title} =====")
    for Document, Score in sorted(Scores.items(), key=lambda x: x[1], reverse=True):
        if Score > 0:
            print(f"{Document:<30}{Score:.4f}")