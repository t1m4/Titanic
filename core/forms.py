from django import  forms

from core.models import Calculations


class User(forms.ModelForm):
    age = forms.IntegerField(min_value=0, max_value=150)
    siblings = forms.IntegerField(min_value=0, max_value=25, label="Sibling/Spouses aboard")
    parents = forms.IntegerField(min_value=0,max_value=2, label="Parents/Children aboard")
    fare = forms.IntegerField(min_value=0, max_value=512)
    class Meta:
        model = Calculations
        fields = ['p_class', 'sex']
