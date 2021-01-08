import json

from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
from django.utils.decorators import method_decorator
from django.views import View
import numpy as np
from project import start, datas_train, targets_train, weight, learning_rate, loops, sigma, write_file, read_file


@method_decorator(login_required, name='dispatch')
class StartView(View):
    def get(self, request, *args, **kwargs):
        try:
            res = start(datas_train, targets_train, weight, learning_rate, loops, sigma=sigma)
            write_file(res)
            return HttpResponse('ok', status=200)
        except Exception as e:
            return HttpResponse(e, status=200)