from sklearn.base import BaseEstimator, TransformerMixin
import spacy
import xx_ent_wiki_sm
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier

class ItemSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, key):
        self.key = key
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, data_dict):
        return data_dict[self.key]

def spacy_tokenize(string):
    nlp = xx_ent_wiki_sm.load(disable=['ner'])
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

pipeSVM = Pipeline([("word", Pipeline([('selector', ItemSelector(key='words')), 
                                         ('tfidf', CountVectorizer(tokenizer=tokenize_normalize,binary=True))])), 
                      ("SVM", SVC(kernel="linear", break_ties=True, max_iter=-1, C=100, tol=0.1))])

pipeTree = Pipeline([("word", Pipeline([('selector', ItemSelector(key='words')), 
                                         ('tfidf', CountVectorizer(tokenizer=tokenize_normalize,binary=True))])), 
                      ("tree", DecisionTreeClassifier())])