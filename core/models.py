from django.core.validators import MinValueValidator, MaxValueValidator
from django.db import models

P_CLASS_CHOICES = [
    ('1', '1'),
    ('2', '2'),
    ('3', '3'),
]
SEX_CHOICES = [
    ('0', 'Man'),
    ('1', 'Woman'),
]
# Create your models here.
class Calculations(models.Model):
    p_class = models.CharField(max_length=32, choices=P_CLASS_CHOICES)
    sex = models.CharField(max_length=3, choices=SEX_CHOICES)
    age = models.IntegerField(validators=[MinValueValidator(0)])
    subling = models.IntegerField(validators=[MinValueValidator(0), MaxValueValidator(25)])
    parents = models.IntegerField(validators=[MinValueValidator(0), MaxValueValidator(2)])
    fare = models.IntegerField(validators=[MinValueValidator(0), MaxValueValidator(512)])
    answers = models.BooleanField(default=False)

class Counts(models.Model):
    enter = models.IntegerField(validators=[MinValueValidator(0)])
