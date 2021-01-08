from django.core.validators import MinValueValidator, MaxValueValidator
from django.db import models

# Create your models here.
class Calculations(models.Model):
    p_class = models.CharField(max_length=32)
    sex = models.IntegerField()
    age = models.IntegerField(validators=[MinValueValidator(0)])
    subling = models.IntegerField(validators=[MinValueValidator(0), MaxValueValidator(25)])
    parents = models.IntegerField(validators=[MinValueValidator(0), MaxValueValidator(2)])
    fare = models.IntegerField(validators=[MinValueValidator(0), MaxValueValidator(512)])
    answers = models.BooleanField(default=False)

class Counts(models.Model):
    enter = models.IntegerField(validators=[MinValueValidator(0)])
