import pandas as pd

# Load dataset
df = pd.read_csv("C:/Users/methu/Downloads/fake reviews dataset.csv")
df = df[['category', 'label', 'text_']]

# Explore unique categories and labels
groupDf = df.groupby(df['category'])
setOfCat = set(df['category'].values.tolist())
setOflab = set(df['label'].values.tolist())
print(setOfCat)
print(setOflab)

# Filter and preprocess text
trueDf = df[df['label'] == 'OR']
cat1TrueDfListOfText = trueDf[trueDf['category'] == 'Tools_and_Home_Improvement_5']['text_'].values.tolist()

fDf = df[df['label'] == 'CG']
cat1FDfListOfText = fDf[fDf['category'] == 'Tools_and_Home_Improvement_5']['text_'].values.tolist()

punc = '''!()-[]{};:'",<>./?@#$%^&*_~'''

preprocesscat1TrueDfListOfText = []
for sentence in cat1TrueDfListOfText:
    sent = ""
    for letter in sentence:
        if letter not in punc:
            sent += letter.lower()
    preprocesscat1TrueDfListOfText.append(sent)
print(preprocesscat1TrueDfListOfText)

preprocesscat1FDfListOfText = []
for sentence in cat1FDfListOfText:
    sent = ""
    for letter in sentence:
        if letter not in punc:
            sent += letter.lower()
    preprocesscat1FDfListOfText.append(sent)
print(preprocesscat1FDfListOfText)

# Text processing - remove stopwords
import nltk
from nltk.corpus import stopwords

stp = stopwords.words('english')

temp = []
for sentence in preprocesscat1TrueDfListOfText:
    subTemp = []
    for word in sentence.split(" "):
        if word not in stp:
            subTemp.append(word)
    temp.append(subTemp)
preprocesscat1TrueDfListOfText = []
for lis in temp:
    preprocesscat1TrueDfListOfText.append(" ".join(lis))
print(preprocesscat1TrueDfListOfText)

temp = []
for sentence in preprocesscat1FDfListOfText:
    subTemp = []
    for word in sentence.split(" "):
        if word not in stp:
            subTemp.append(word)
    temp.append(subTemp)
preprocesscat1FDfListOfText = []
for lis in temp:
    preprocesscat1FDfListOfText.append(" ".join(lis))
print(preprocesscat1FDfListOfText)

# Sentiment analysis and association rule mining
import matplotlib.pyplot as plt
import numpy as np
from apyori import apriori
from nltk.sentiment import SentimentIntensityAnalyzer

SA = SentimentIntensityAnalyzer()

posNnegInCat1True = []
total = 0
for sentence in preprocesscat1TrueDfListOfText:
    total += 1
    neg = []
    pos = []
    for word in sentence.split(" "):
        if SA.polarity_scores(word)["pos"] == 1:
            pos.append(word)
        elif SA.polarity_scores(word)["neg"] == 1:
            neg.append(word)

    if len(neg) != 0 and len(pos) != 0 and len(pos) > len(neg):
        posNnegInCat1True.append([pos, neg])

print(len(posNnegInCat1True), posNnegInCat1True)

posNnegInCat1F = []
for sentence in preprocesscat1FDfListOfText:
    neg = []
    pos = []
    for word in sentence.split(" "):
        if SA.polarity_scores(word)["pos"] == 1:
            pos.append(word)
        elif SA.polarity_scores(word)["neg"] == 1:
            neg.append(word)
    if len(neg) != 0 and len(pos) != 0 and len(pos) > len(neg):
        posNnegInCat1F.append([pos, neg])

print(len(posNnegInCat1F), posNnegInCat1F)

# N-gram analysis
joinedCat1True = (" ".join(preprocesscat1TrueDfListOfText)).split(" ")
joinedCat1True = [i for i in joinedCat1True if i != '']
print(joinedCat1True)

n3GramT = nltk.collocations.TrigramCollocationFinder.from_words(joinedCat1True)
n2GramT = nltk.collocations.BigramCollocationFinder.from_words(joinedCat1True)

joinedCat1F = (" ".join(preprocesscat1FDfListOfText)).split(" ")
joinedCat1F = [i for i in joinedCat1F if i != '']
print(joinedCat1F)

n3GramF = nltk.collocations.TrigramCollocationFinder.from_words(joinedCat1F)
n2GramF = nltk.collocations.BigramCollocationFinder.from_words(joinedCat1F)

print(n3GramT.ngram_fd.most_common(9))
print(n2GramT.ngram_fd.most_common(9))
print(n3GramF.ngram_fd.most_common(9))
print(n2GramF.ngram_fd.most_common(9))

# N-gram Words
n3GramWords = []
for i in n3GramT.ngram_fd.most_common(9):
    for j in i[0]:
        n3GramWords.append(j)
print(n3GramWords)

# Part-of-speech tagging
posTagN3 = nltk.pos_tag(n3GramWords)
print(posTagN3)

n2GramWords = []
for i in n2GramT.ngram_fd.most_common(9):
    for j in i[0]:
        n2GramWords.append(j)
print(n2GramWords)

posTagN2 = nltk.pos_tag(n2GramWords)
print(posTagN2)

from sematch.semantic.similarity import WordNetSimilarity
wns = WordNetSimilarity()

import mlxtend
from mlxtend.frequent_patterns import apriori, association_rules

data = df
data.columns

posNnegInCat1True = []
posNum = 0
negNum = 0
total = 0
posNnegNum = 0
for sentence in preprocesscat1TrueDfListOfText+preprocesscat1FDfListOfText:
    total += 1
    neg = []
    pos = []
    for word in sentence.split(" "):
        if SA.polarity_scores(word)["pos"] == 1:
            pos.append(word)
        elif SA.polarity_scores(word)["neg"] == 1:
            neg.append(word)
    if len(neg) != 0:
        negNum += 1
    if len(pos) != 0:
        posNum += 1
    if len(neg) != 0 and len(pos) != 0:
        posNnegInCat1True.append([pos, neg])
        posNnegNum += 1
t = posNnegNum*2/3
print("Support :", posNnegNum/total)

posNnegInCat1True = []
posNum = 0
negNum = 0
total = 0
posNnegNum = 0
SUPP = t
for sentence in preprocesscat1TrueDfListOfText:
    total += 1
    neg = []
    pos = []
    for word in sentence.split(" "):
        if SA.polarity_scores(word)["pos"] == 1:
            pos.append(word)
        elif SA.polarity_scores(word)["neg"] == 1:
            neg.append(word)
    if len(neg) != 0:
        negNum += 1
    if len(pos) != 0:
        posNum += 1
    if len(neg) != 0 and len(pos) != 0:
        posNnegInCat1True.append([pos, neg])
        posNnegNum += 1
print("Confidence :", posNnegNum/SUPP)
