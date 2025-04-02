from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import PDFDocument, Summary

admin.site.register(PDFDocument)
admin.site.register(Summary)
