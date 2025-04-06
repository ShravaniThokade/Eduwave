import os
import tempfile
import whisper
import yt_dlp
from transformers import pipeline
from django.shortcuts import render, redirect
from .forms import CreateUserForm, LoginForm
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import auth
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import shutil

import pymupdf
import tempfile
import re

# Load the summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")



def download_audio(video_url, output_dir):
    """Download the audio from the YouTube video."""
    output_path = os.path.join(output_dir, "audio.%(ext)s")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'quiet': True,
        'noplaylist': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',  # Use WAV for Whisper
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

def summarize_text(text):
    """Summarize the transcribed text using BART."""
    try:
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        summarized = []
        for chunk in chunks:
            summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
            summarized.append(summary[0]['summary_text'])
        return " ".join(summarized)
    except Exception as e:
        return f"Summarization failed: {str(e)}"

def transcribe_youtube_audio(video_url):
    print("🎬 Downloading audio using yt_dlp...")
    temp_dir = tempfile.mkdtemp()
    
    try:
        download_audio(video_url, temp_dir)

        final_audio = os.path.join(temp_dir, "audio.wav")
        if not os.path.exists(final_audio):
            print("❌ Audio file not found. Check yt_dlp download path.")
            return "Audio file not found after download."

        print("🧠 Loading Whisper model...")
        model = whisper.load_model("base")

        print("📝 Transcribing (Hindi → English)...")
        result = model.transcribe(final_audio, language="hi", task="translate")

        full_text = result["text"]
        print("\n🔊 Transcription Complete!\n")

        print("\n📄 Generating Summary...\n")
        summary = summarize_text(full_text)
        print("✅ Summary Generated.")
        return summary

    finally:
        # Optional: clean up the temp directory
        shutil.rmtree(temp_dir)

@csrf_exempt
def youtube(request):
    if request.method == 'POST':
        youtube_video = request.POST.get('youtube_video')
        print(f"📹 Video URL received: {youtube_video}")
        summary = transcribe_youtube_audio(youtube_video)
        return render(request, 'escape/youtube.html', {'summary': summary})
    else:
        return render(request, 'escape/youtube.html')


def homepage(request):
    return render(request, 'escape/index.html')

def register(request):
    form = CreateUserForm()
    if request.method == "POST":
        form = CreateUserForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("login")
    context = {'registerform': form}
    return render(request, 'escape/register.html', context=context)

def login(request):
    form = LoginForm()
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            username = request.POST.get('username')
            password = request.POST.get('password')
            user = authenticate(request, username=username, password=password)
            if user is not None:
                auth.login(request, user)
                return redirect("dashboard")
    context = {'loginform': form}
    return render(request, 'escape/login.html', context=context)

def user_logout(request):
    auth.logout(request)
    return redirect("login")

@login_required(login_url="login")
def dashboard(request):
    return render(request, 'escape/dashboard.html')

@csrf_exempt
def generate_summary(request):
    if request.method == 'POST' and request.FILES.get('pdf_file'):
        temp_pdf_path = None  
        try:
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
                text = page.get_text()
                if not text.strip():
                    continue

                lines = []
                for line in text.split('\n'):
                    clean_line = line.strip()
                    if re.match(r'^(page\s*)?\d+$', clean_line, re.IGNORECASE):
                        continue
                    lines.append(clean_line)

                cleaned_text = '\n'.join(lines)

                question_blocks = []
                current_question = None
                current_block = []

                colored_questions = []
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    for line in block.get("lines", []):
                        colored_text = ""
                        is_colored = False
                        for span in line["spans"]:
                            if span.get("color", 0) != 0:
                                is_colored = True
                            colored_text += span["text"].strip() + " "
                        colored_text = colored_text.strip()
                        if is_colored and len(colored_text) > 5:
                            colored_questions.append(colored_text)

                question_candidates = set(colored_questions) | set(lines)

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
                        if not content.strip():
                            continue
                        num_sentences = min(5 + (len(content.split()) // 100), 10)
                        parser = PlaintextParser.from_string(content, Tokenizer('english'))
                        summary = summarizer(parser.document, num_sentences)
                        summary_text = "\n".join(f"• {sentence}" for sentence in summary)
                        summaries[question] = summary_text
                else:
                    num_sentences = min(5 + (len(text.split()) // 100), 15)
                    parser = PlaintextParser.from_string(text, Tokenizer('english'))
                    summary = summarizer(parser.document, num_sentences)
                    if not summary:
                        continue
                    summary_text = "\n".join(f"• {sentence}" for sentence in summary)
                    summaries[f"Page {i+1}"] = summary_text

            doc.close()

            if not summaries:
                return JsonResponse({'error': "No summary could be generated. Try a different PDF."}, status=500)

            return JsonResponse({'summaries': summaries}, json_dumps_params={'indent': 4})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

        finally:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.remove(temp_pdf_path)
                except Exception:
                    pass
    else:
        return render(request, 'escape/pdf_summary.html')
def study_set(request):
    return render(request, 'escape/study_set.html')

def import_materials(request):
    return render(request, 'escape/import_materials.html')

def transcription(request):
    return render(request, 'escape/transcription.html')

def test(request):
    return render(request, 'escape/test.html') 

