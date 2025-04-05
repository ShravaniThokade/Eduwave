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
import pymupdf
import sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import tempfile
import nltk
import os
import re
nltk.download('punkt')
nltk.download('punkt_tab') 


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
    return redirect("login")

@login_required(login_url="login")
def dashboard(request):
    return render(request, 'escape/dashboard.html')

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-medium.en")

# def transcribe_audio(stream, new_chunk):
#     sr, y = new_chunk
#     y = y.astype(np.float32)
#     y /= np.max(np.abs(y))

#     if stream is not None:
#         stream = np.concatenate([stream, y])
#     else:
#         stream = y
#     return stream, transcriber({"sampling_rate": sr, "raw": stream})["text"]


# def process_audio(request):
#     if request.method == 'POST':
#         audio_data = request.POST.get('audio')
#         # Process the audio data (e.g., transcribe with Whisper model)
#         transcription = transcribe_audio(audio_data)
#         return JsonResponse({'transcription': transcription})
#     else:
#         return JsonResponse({'error': 'Invalid request method'})


# def youtube(request):
#     if request.method == 'POST':
#         youtube_video = request.POST.get('youtube_video')
#         video_id = youtube_video.split("=")[1]
#         transcript = YouTubeTranscriptApi.get_transcript(video_id)
#         result = ""
#         for i in transcript:
#             result += ' ' + i['text']
#         summarizer = pipeline('summarization')
#         num_iters = int(len(result)/1000)
#         summarized_text = []
#         for i in range(0, num_iters + 1):
#             start = 0
#             start = i * 1000
#             end = (i + 1) * 1000
#             out = summarizer(result[start:end])
#             out = out[0]
#             out = out['summary_text']
#             summarized_text.append(out)
#         return render(request, 'escape/youtube.html', {'summary': summarized_text})
#     else:
#         return render(request, 'escape/youtube.html')


def youtube(request):
    if request.method == 'POST':
        youtube_video = request.POST.get('youtube_video')
        video_id = youtube_video.split("=")[-1]
        
        # Fetch transcript with timestamps
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # Summarization model
        summarizer = pipeline('summarization')

        # Organizing transcript into 2-minute intervals
        interval = 120  # 2 minutes in seconds
        grouped_transcript = {}
        current_time = 0
        current_text = ""

        for entry in transcript:
            start_time = int(entry['start'])
            
            # If we reach the next 2-minute mark, store the previous segment
            if start_time >= current_time + interval:
                grouped_transcript[f"{current_time//60:02}:00"] = current_text.strip()
                current_time += interval
                current_text = entry['text']
            else:
                current_text += " " + entry['text']

        # Store the last segment
        if current_text:
            grouped_transcript[f"{current_time//60:02}:00"] = current_text.strip()

        # Summarize each 2-minute segment
        final_summary = []
        for timestamp, text in grouped_transcript.items():
            summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
            final_summary.append(f"{timestamp}\n{summary}\n")

        return render(request, 'escape/youtube.html', {'summary': final_summary})

    return render(request, 'escape/youtube.html')





@csrf_exempt
# import tempfile
# import pymupdf  # fitz
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.summarizers.lsa import LsaSummarizer
# from django.http import JsonResponse
# from django.shortcuts import render
# import os

def generate_summary(request):
    if request.method == 'POST' and request.FILES.get('pdf_file'):
        temp_pdf_path = None  
        try:
            pdf_file = request.FILES['pdf_file']
            print(f"📄 Received file: {pdf_file.name}")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                for chunk in pdf_file.chunks():
                    temp_pdf.write(chunk)
                temp_pdf_path = temp_pdf.name
                print(f"📂 Saved temporary file at: {temp_pdf_path}")

            doc = pymupdf.open(temp_pdf_path)
            num_pages = len(doc)
            print(f"📜 PDF has {num_pages} pages.")

            summarizer = LsaSummarizer()
            summaries = {}

            question_pattern = re.compile(
                r"^(Q\d+[\.\:\)]?|[0-9]+[\.\:\)]\s+(Explain|What|Describe|Define|How|Why|List|State|Discuss|Write))",
                re.IGNORECASE
            )

            for i, page in enumerate(doc):
                text = page.get_text()
                print(f"📃 Page {i+1} Text Extracted (First 200 chars): {text[:200]}")
                if not text.strip():
                    print(f"⚠️ Page {i+1} is empty, skipping...")
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

                # ✅ Added: Collect non-black colored text
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

                # ✅ Combine colored questions into detection
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
                    print(f"❓ Found {len(question_blocks)} questions on Page {i+1}")
                    for idx, (question, content) in enumerate(question_blocks):
                        print(f"🔍 Processing Q{idx+1}: {question[:100]}")
                        if not content.strip():
                            print(f"⚠️ No content found for question: {question}")
                            continue

                        num_sentences = min(5 + (len(content.split()) // 100), 10)
                        parser = PlaintextParser.from_string(content, Tokenizer('english'))
                        summary = summarizer(parser.document, num_sentences)
                        summary_text = "\n".join(f"• {sentence}" for sentence in summary)
                        summaries[question] = summary_text
                        print(f"✅ Summary for '{question}': {summary_text[:200]}...")
                else:
                    print(f"ℹ️ No questions detected on Page {i+1}. Using fallback summarization.")
                    num_sentences = min(5 + (len(text.split()) // 100), 15)
                    parser = PlaintextParser.from_string(text, Tokenizer('english'))
                    summary = summarizer(parser.document, num_sentences)
                    if not summary:
                        print(f"⚠️ No summary generated for Page {i+1}.")
                        continue
                    summary_text = "\n".join(f"• {sentence}" for sentence in summary)
                    summaries[f"Page {i+1}"] = summary_text
                    print(f"✅ Fallback summary for Page {i+1}: {summary_text[:200]}...")

            doc.close()

            if not summaries:
                print("❌ No summaries generated. Returning error.")
                return JsonResponse({'error': "No summary could be generated. Try a different PDF."}, status=500)

            return JsonResponse({'summaries': summaries}, json_dumps_params={'indent': 4})

        except Exception as e:
            print(f"❌ Error processing PDF: {e}")
            return JsonResponse({'error': str(e)}, status=500)

        finally:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.remove(temp_pdf_path)
                    print(f"🗑️ Deleted temporary file: {temp_pdf_path}")
                except Exception as cleanup_error:
                    print(f"⚠️ Could not delete temp file: {cleanup_error}")
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
