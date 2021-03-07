from django.urls import path
from ontology import views

urlpatterns = [
    path('search/', views.search_vocabulary)
]
