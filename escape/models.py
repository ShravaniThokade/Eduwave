from django.db import models

# Create your models here.

from django.db import models

class PDFDocument(models.Model):
    title = models.CharField(max_length=255)  # PDF title
    file = models.FileField(upload_to='pdfs/')  # Store the uploaded PDF
    uploaded_at = models.DateTimeField(auto_now_add=True)  # Upload timestamp

    def __str__(self):
        return self.title

class Summary(models.Model):
    pdf = models.OneToOneField(PDFDocument, on_delete=models.CASCADE)  # Link to PDF
    summary_text = models.TextField()  # Store the summary
    generated_at = models.DateTimeField(auto_now_add=True)  # When summary was created

    def __str__(self):
        return f"Summary of {self.pdf.title}"
