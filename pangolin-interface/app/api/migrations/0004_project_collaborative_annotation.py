# Generated by Django 2.1.7 on 2019-08-01 18:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0003_support_sql_server'),
    ]

    operations = [
        migrations.AddField(
            model_name='project',
            name='collaborative_annotation',
            field=models.BooleanField(default=False),
        ),
    ]
