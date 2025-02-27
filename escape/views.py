from django.shortcuts import render, redirect
from . forms import CreateUserForm, LoginForm
from django.contrib.auth.models import auth
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from transformers import pipeline
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi
from django.views.decorators.csrf import csrf_exempt
import fitz
import sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer


def homepage(request):
    return render(request, 'escape/index.html')

def register(request):
    form = CreateUserForm()
    if request.method == "POST":
        form = CreateUserForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("login")
    context = {'registerform':form}
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
    context = {'loginform':form}
    return render(request, 'escape/login.html', context=context)

def user_logout(request):
    auth.logout(request)
    return redirect("")

@login_required(login_url="login")
def dashboard(request):
    return render(request, 'escape/dashboard.html')

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-medium.en")

def transcribe_audio(stream, new_chunk):
    sr, y = new_chunk
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y
    return stream, transcriber({"sampling_rate": sr, "raw": stream})["text"]


def process_audio(request):
    if request.method == 'POST':
        audio_data = request.POST.get('audio')
        # Process the audio data (e.g., transcribe with Whisper model)
        transcription = transcribe_audio(audio_data)
        return JsonResponse({'transcription': transcription})
    else:
        return JsonResponse({'error': 'Invalid request method'})


def youtube(request):
    if request.method == 'POST':
        youtube_video = request.POST.get('youtube_video')
        video_id = youtube_video.split("=")[1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        result = ""
        for i in transcript:
            result += ' ' + i['text']
        summarizer = pipeline('summarization')
        num_iters = int(len(result)/1000)
        summarized_text = []
        for i in range(0, num_iters + 1):
            start = 0
            start = i * 1000
            end = (i + 1) * 1000
            out = summarizer(result[start:end])
            out = out[0]
            out = out['summary_text']
            summarized_text.append(out)
        return render(request, 'escape/youtube.html', {'summary': summarized_text})
    else:
        return render(request, 'escape/youtube.html')


@csrf_exempt
def generate_summary(request):
    if request.method == 'POST' and request.FILES.get('pdf_file'):
        try:
            pdf_file = request.FILES['pdf_file']

            # Save the uploaded PDF temporarily
            with open('temp.pdf', 'wb') as f:
                for chunk in pdf_file.chunks():
                    f.write(chunk)

            # Open the PDF file and extract text content
            doc = fitz.open('temp.pdf')
            text = ""
            for page in doc:
                text += page.get_text()

            # Use Sumy library for text summarization
            parser = PlaintextParser.from_string(text, Tokenizer('english'))
            summarizer = LsaSummarizer()
            summary = summarizer(parser.document, 10)  # Extract 10 sentences as summary

            # Join extracted sentences into a single summary string
            summary_text = ' '.join(str(sentence) for sentence in summary)

            return JsonResponse({'summary': summary_text})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
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
