from django.apps import AppConfig
import joblib
import os
from api.settings import BASE_DIR
from pickledModels import ItemSelector, tokenize_normalize, normalize, spacy_tokenize

class OntologyConfig(AppConfig):
    name = 'ontology'
    pipeSVM = None
    verbose_name = "Ontology"
    machineLearning = {
        "SVM": None,
        "Tree": None,
        "Logistic": None
    }

    def ready(self):
        print("Loading Django")
        pickledModels = open(os.path.join(BASE_DIR, "models.p"), "rb")
        data = joblib.load(pickledModels)
        self.machineLearning["SVM"] = data["SVM"]
        self.machineLearning["Tree"] = data["tree"]
        self.machineLearning["Logistic"] = data["Logistic"]
        pickledModels.close()
