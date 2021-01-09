from django.forms import Form, ModelForm

from core.models import Calculations


class User(ModelForm):
    class Meta:
        model = Calculations
        fields = ['p_class', 'age', 'subling', 'parents', 'fare']
        # fields = '__all__'
