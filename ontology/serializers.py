from rest_framework import serializers
from ontology.models import Vocabulary

class VocabularySerializer(serializers.ModelSerializer):
    class Meta:
        model = Vocabulary
        fields = ['word', 'theme', 'sensitivity']