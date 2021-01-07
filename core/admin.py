from django.contrib import admin

# Register your models here.
from core.models import Calculations, Counts

admin.site.register(Calculations)
admin.site.register(Counts)