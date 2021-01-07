# Generated by Django 3.1.3 on 2021-01-07 10:30

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Calculations',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('p_class', models.CharField(max_length=32)),
                ('sex', models.IntegerField()),
                ('age', models.IntegerField(validators=[django.core.validators.MinValueValidator(0)])),
                ('subling', models.IntegerField(validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(25)])),
                ('parents', models.IntegerField(validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(2)])),
                ('fare', models.IntegerField(validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(512)])),
                ('answers', models.BooleanField(default=False)),
            ],
        ),
        migrations.CreateModel(
            name='Counts',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('enter', models.IntegerField(validators=[django.core.validators.MinValueValidator(0)])),
            ],
        ),
    ]
