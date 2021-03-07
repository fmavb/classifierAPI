from django.db import models

# Create your models here.
class Sensitivity(models.Model):
    sensitivityID = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=50)

class Themes(models.Model):
    themeID = models.IntegerField(primary_key=True)
    theme = models.CharField(max_length=50, blank=True)
    sensitivity = models.IntegerField(null=True)
    created = models.DateTimeField()

class Vocabulary(models.Model):
    word = models.CharField(max_length=50)
    language = models.CharField(max_length=50)
    themeID = models.IntegerField(null=True)
    sensitivity = models.IntegerField(null=True)
    created = models.DateTimeField()
    fieldID = models.AutoField(primary_key=True)

class VocabularyOpenCyc(models.Model):
    word = models.CharField(max_length=50)
    theme = models.CharField(max_length=50)
    sensitivity = models.IntegerField(null=True)
    created = models.DateTimeField()
    fieldID = models.AutoField(primary_key=True)
