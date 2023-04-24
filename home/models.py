from django.db import models

# Create your models here.

class UserUpload(models.Model):
    image = models.ImageField()
    timestamp = models.DateTimeField(auto_now_add=True)