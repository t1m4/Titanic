import json

from django.contrib.auth.decorators import login_required
from django.db.models import F
from django.http import HttpResponse
from django.shortcuts import render, redirect

# Create your views here.
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.views import View
import numpy as np

from core.forms import User
from core.models import Calculations, Counts
from project import start, datas_train, targets_train, weight, learning_rate, loops, sigma, write_file, read_file, \
    my_vectors, normalize_dataset, get_prediction


@method_decorator(login_required, name='dispatch')
class StartView(View):
    def get(self, request, *args, **kwargs):
        try:
            res = start(datas_train, targets_train, weight, learning_rate, loops, sigma=sigma)
            write_file(res)
            return HttpResponse('ok', status=200)
        except Exception as e:
            return HttpResponse(e, status=200)

def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[-1].strip()
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip
class FormView(View):
    template_name = 'core/index.html'
    success_url = 'core-form'
    form_class = User
    context = {}
    def get(self, request, *args, **kwargs):
        form = self.form_class()
        self.context = {'form': form}
        return render(request, self.template_name, context=self.context)
    def post(self, request, *args, **kwargs):
        form = self.form_class(request.POST)
        if form.is_valid():
            result = read_file()
            user_vectors = np.array([int(v) for k, v in form.cleaned_data.items()])
            my_vector = normalize_dataset(user_vectors)
            answer = get_prediction(my_vector, result['best_weight'])
            Calculations.objects.create(
                p_class=form.cleaned_data['p_class'],
                sex=form.cleaned_data['sex'],
                age=form.cleaned_data['age'],
                subling=form.cleaned_data['siblings'],
                parents=form.cleaned_data['parents'],
                fare=form.cleaned_data['fare'],
                answers=answer
            ).save()
            Counts.objects.all().update(enter=F('enter') + 1)
            return redirect(reverse(self.success_url))
