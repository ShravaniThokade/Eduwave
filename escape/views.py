import os
import re
import shutil
import tempfile

import whisper
import yt_dlp
import pymupdf

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate
from django.contrib.auth import login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required

from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


from .forms import CreateUserForm, LoginForm

# 🔄 Global BART summarizer for YouTube
bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ----------------------------- AUTH VIEWS -----------------------------

def homepage(request):
    return render(request, 'escape/index.html')

def register(request):
    form = CreateUserForm()
    if request.method == "POST":
        form = CreateUserForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("login")
    return render(request, 'escape/register.html', {'registerform': form})

def login(request):
    form = LoginForm()
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            user = authenticate(
                request,
                username=request.POST.get('username'),
                password=request.POST.get('password')
            )
            if user:
                auth_login(request, user)
                return redirect("dashboard")
    return render(request, 'escape/login.html', {'loginform': form})

def user_logout(request):
    auth_logout(request)
    return redirect("login")

@login_required(login_url="login")
def dashboard(request):
    return render(request, 'escape/dashboard.html')

# ----------------------------- YOUTUBE PROCESSING -----------------------------

def download_audio(video_url, output_dir):
    output_path = os.path.join(output_dir, "audio.%(ext)s")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'quiet': True,
        'noplaylist': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

def summarize_text(text):
    try:
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        summarized = [bart_summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
                      for chunk in chunks]
        return " ".join(summarized)
    except Exception as e:
        return f"Summarization failed: {str(e)}"

def transcribe_youtube_audio(video_url):
    print("🎬 Downloading audio...")
    temp_dir = tempfile.mkdtemp()
    try:
        download_audio(video_url, temp_dir)
        final_audio = os.path.join(temp_dir, "audio.wav")
        if not os.path.exists(final_audio):
            return {"summary": "Audio not found", "insights": []}

        print("🧠 Loading Whisper model...")
        model = whisper.load_model("base")

        print("📝 Transcribing (Hindi → English)...")
        result = model.transcribe(final_audio, language="hi", task="translate")
        full_text = result["text"]

        print("📄 Generating Summary...")
        summary = summarize_text(full_text)

        print("🔍 Extracting Key Insights...")
        insights = extract_key_insights(full_text)

        return {
            "summary": summary,
            "insights": insights
        }

    finally:
        shutil.rmtree(temp_dir)

def extract_key_insights(text, top_n=5):
    """Return top N most informative sentences using TF-IDF scores."""
    try:
        sentences = re.split(r'(?<=[.!?]) +', text.strip())
        if len(sentences) <= top_n:
            return sentences

        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Sum of TF-IDF scores across all terms for each sentence
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).ravel()

        # Get top N indices
        top_indices = sentence_scores.argsort()[-top_n:][::-1]
        key_sentences = [sentences[i] for i in top_indices]
        return key_sentences

    except Exception as e:
        print(f"⚠️ TF-IDF extraction failed: {e}")
        return []


@csrf_exempt
def youtube(request):
    if request.method == 'POST':
        video_url = request.POST.get('youtube_video')
        print(f"📹 Received: {video_url}")
        result = transcribe_youtube_audio(video_url)
        return render(request, 'escape/youtube.html', {
            'summary': result["summary"],
            'insights': result["insights"]
        })
    return render(request, 'escape/youtube.html')


# ----------------------------- PDF SUMMARIZER -----------------------------

@csrf_exempt
def generate_summary(request):
    if request.method == 'POST' and request.FILES.get('pdf_file'):
        temp_pdf_path = None
        try:
            # Save PDF temporarily
            pdf_file = request.FILES['pdf_file']
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                for chunk in pdf_file.chunks():
                    temp_pdf.write(chunk)
                temp_pdf_path = temp_pdf.name

            doc = pymupdf.open(temp_pdf_path)
            summarizer = LsaSummarizer()
            summaries = {}

            question_pattern = re.compile(
                r"^(Q\d+[\.\:\)]?|[0-9]+[\.\:\)]\s+(Explain|What|Describe|Define|How|Why|List|State|Discuss|Write))",
                re.IGNORECASE
            )

            for i, page in enumerate(doc):
                page_text = page.get_text()
                if not page_text.strip():
                    continue

                lines = [line.strip() for line in page_text.split('\n')
                         if not re.match(r'^(page\s*)?\d+$', line.strip(), re.IGNORECASE)]

                colored_questions = [
                    span["text"].strip()
                    for block in page.get_text("dict")["blocks"]
                    for line in block.get("lines", [])
                    for span in line["spans"]
                    if span.get("color", 0) != 0 and len(span["text"].strip()) > 5
                ]

                question_candidates = set(colored_questions) | set(lines)

                question_blocks = []
                current_question = None
                current_block = []

                for line in question_candidates:
                    line = line.strip()
                    if question_pattern.match(line) or line in colored_questions:
                        if current_question:
                            question_blocks.append((current_question, ' '.join(current_block)))
                            current_block = []
                        current_question = line
                    elif current_question:
                        current_block.append(line)

                if current_question and current_block:
                    question_blocks.append((current_question, ' '.join(current_block)))

                if question_blocks:
                    for question, content in question_blocks:
                        if content.strip():
                            num_sentences = min(5 + (len(content.split()) // 100), 10)
                            parser = PlaintextParser.from_string(content, Tokenizer('english'))
                            summary = summarizer(parser.document, num_sentences)
                            summary_text = "\n".join(f"• {sentence}" for sentence in summary)
                            summaries[question] = summary_text
                else:
                    num_sentences = min(5 + (len(page_text.split()) // 100), 15)
                    parser = PlaintextParser.from_string(page_text, Tokenizer('english'))
                    summary = summarizer(parser.document, num_sentences)
                    summaries[f"Page {i+1}"] = "\n".join(f"• {sentence}" for sentence in summary)

            doc.close()
            if not summaries:
                return JsonResponse({'error': "No summary could be generated."}, status=500)

            return JsonResponse({'summaries': summaries}, json_dumps_params={'indent': 4})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        finally:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

    return render(request, 'escape/pdf_summary.html')

# ----------------------------- MISC VIEWS -----------------------------

def study_set(request):
    return render(request, 'escape/study_set.html')

def import_materials(request):
    return render(request, 'escape/import_materials.html')

def transcription(request):
    return render(request, 'escape/transcription.html')

def test(request):
    return render(request, 'escape/test.html')
