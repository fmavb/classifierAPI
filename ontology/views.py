from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest
from rest_framework.parsers import JSONParser
from ontology.models import Vocabulary, VocabularyOpenCyc
from ontology.serializers import VocabularySerializer
import requests, json
import pandas as pd
from ontology.apps import OntologyConfig
from django.views.decorators.csrf import csrf_exempt
from statistics import mode

@csrf_exempt
def search_vocabulary(request):
    if request.method != 'POST':
        return HttpResponseBadRequest("Invalid request method " + request.method)
    else:
        body = json.loads(request.body)
        keywords = body.get("keywords")
        if keywords:
            try:
                pipeSVM = OntologyConfig.machineLearning["SVM"]
                pipeTree = OntologyConfig.machineLearning["Tree"]
                pipeLogis = OntologyConfig.machineLearning["Logistic"]
                gaussianVec = OntologyConfig.machineLearning["GaussianVec"]
                gaussian = OntologyConfig.machineLearning["Gaussian"]

                toDF = []
                for keyword in keywords:
                    toDF.append([keyword, 0])
                classify = pd.DataFrame(toDF, columns=("words", "sensitivity"))
                
                predictionSVM = pipeSVM.predict(classify)
                predictionTree = pipeTree.predict(classify)
                predictionLogis = pipeLogis.predict(classify)

                vectors = gaussianVec.transform(classify["words"])
                predictionGauss = gaussian.predict(vectors.toarray())
                
                prediction = []
                
                for i in range(len(predictionSVM)):
                    svm = predictionSVM[i]
                    tree = predictionTree[i]
                    logis = predictionLogis[i]
                    gaus = predictionGauss[i]
                    vote = [svm, svm, svm, tree, tree, tree, logis, logis, gaus]
                    
                    prediction.append(mode(vote))
                    
                response = []
                for i in range(len(prediction)):
                    response.append({"word": classify["words"][i], "prediction": prediction[i].item()}) 
                return JsonResponse(response, safe=False)
            except Vocabulary.DoesNotExist:
                lov = requests.get("https://lov.linkeddata.es/dataset/lov/api/v2/term/search?q="+keyword)
                print(lov.json())
        else:
            return HttpResponseBadRequest("Invalid argument")
        return JsonResponse({'response':keyword})


