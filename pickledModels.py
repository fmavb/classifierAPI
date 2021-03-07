import os
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import mysql.connector
from sklearn.base import BaseEstimator, TransformerMixin
import spacy
import xx_ent_wiki_sm
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import joblib


class ItemSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, key):
        self.key = key
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, data_dict):
        return data_dict[self.key]

#@Tokenize
def spacy_tokenize(string):
    tokens = list()
    doc = nlp(string)
    for token in doc:
        tokens.append(token)
    return tokens

#@Normalize
def normalize(tokens):
    normalized_tokens = list()
    for token in tokens:
        normalized = token.text.lower().strip()
        if ((token.is_alpha or token.is_digit)):
            normalized_tokens.append(normalized)
    return normalized_tokens

#@Tokenize and normalize
def tokenize_normalize(string):
    return normalize(spacy_tokenize(string))


nlp = xx_ent_wiki_sm.load(disable=['ner'])
"""
db = mysql.connector.connect(host="localhost",
user="root", password="yourpassword", database="ontology")

cursor = db.cursor()

cursor.execute("set @@sql_mode='NO_ENGINE_SUBSTITUTION';")
db.commit()

sql = "SELECT word, sensitivity FROM vocabulary WHERE sensitivity IS NOT NULL GROUP BY word;"
cursor.execute(sql)
result = cursor.fetchall()

sql2 = 'SELECT word, sensitivity FROM vocabularyopencyc WHERE sensitivity IS NOT NULL GROUP BY word'
cursor.execute(sql2)
result += cursor.fetchall()"""

words = pd.read_csv("ontology.csv", header=0)
words.columns = ["words", "sensitivity"]

opencyc = pd.read_csv("opencyc.csv", header=0)
opencyc.columns = ["words", "sensitivity"]

words = words.append(opencyc)
print(words)
words = words.sample(frac=1, random_state=2)

training = 0.6
validation = 0.75
testing = 0.8

training_data = words.iloc[:int(len(words)*training),:]
validation_data = words.iloc[int(len(words)*training):int(len(words)*validation),:]
testing_data = words.iloc[int(len(words)*validation):,:]

train_labels = training_data["sensitivity"]
validation_labels = validation_data["sensitivity"]
test_labels = testing_data["sensitivity"]



pipeSVM = Pipeline([("word", Pipeline([('selector', ItemSelector(key='words')), 
                                         ('tfidf', CountVectorizer(tokenizer=tokenize_normalize,binary=True))])), 
                      ("SVM", SVC(kernel="linear", break_ties=True, max_iter=-1, C=100, tol=0.1))])

pipeSVM.fit(training_data, train_labels)

pipeTree = Pipeline([("word", Pipeline([('selector', ItemSelector(key='words')), 
                                         ('tfidf', CountVectorizer(tokenizer=tokenize_normalize,binary=True))])), 
                      ("tree", DecisionTreeClassifier())])



pipeTree.fit(training_data, train_labels)

pipeLogis = Pipeline([("word", Pipeline([('selector', ItemSelector(key='words')), 
                                         ('tfidf', CountVectorizer(tokenizer=tokenize_normalize,binary=True))])), 
                      ("lr", LogisticRegression())])


pipeLogis.fit(training_data, train_labels)

pickl = {
    "SVM": pipeSVM,
    "tree": pipeTree,
    "Logistic": pipeLogis,
}
joblib.dump(pickl, open('models.p', "wb"))
