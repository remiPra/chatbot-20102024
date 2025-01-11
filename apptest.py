import os
from pydub import AudioSegment
import speech_recognition as sr
from groq import Groq
import time
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, Response
import base64
import logging
import tempfile
import uuid
import asyncio
from edge_tts import Communicate
from anthropic import Anthropic
import pyaudio
import wave
import numpy as np
from array import array
from datetime import datetime
import asyncio
import tempfile
import json
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import io

# Configuration du logging
logging.basicConfig(level=logging.DEBUG)

# Charger les variables d'environnement
load_dotenv()

app = Flask(__name__)

# Configuration audio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 500
SILENCE_CHUNKS = 10

# Création des dossiers nécessaires
TEMP_AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_audio')
RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recordings')

for directory in [TEMP_AUDIO_DIR, RECORDINGS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Configuration des clients API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("La clé API Groq n'est pas définie dans le fichier .env")
if not ANTHROPIC_API_KEY:
    raise ValueError("La clé API Anthropic n'est pas définie dans le fichier .env")

groq_client = Groq(api_key=GROQ_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Dictionnaire pour stocker les contextes de conversation
conversation_contexts = {}

class AudioDetector:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_running = False
        self.speaking_count = 0
        self.required_detections = 3
        self.noise_threshold = 8000
        
    def start(self):
        if not self.stream:
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            self.is_running = True
            self.speaking_count = 0

    def stop(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.is_running = False
        self.speaking_count = 0
        
    def __del__(self):
        self.stop()
        if self.audio:
            self.audio.terminate()

    def check_speaking(self):
        if not self.stream:
            return False
        try:
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            audio_data = array('h', data)
            volume = max(audio_data)
            if volume > self.noise_threshold:
                self.speaking_count += 1
            else:
                self.speaking_count = max(0, self.speaking_count - 1)
            return self.speaking_count >= self.required_detections
        except Exception as e:
            print(f"Erreur lors de la détection audio: {e}")
            return False

class AudioRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        
    def start_recording(self):
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        self.frames = []
        self.is_recording = True
        
    def stop_recording(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.is_recording = False
        
    def record_until_silence(self):
        silence_count = 0
        recording_started = False
        waiting_for_sound = True
        max_duration = 200
        total_chunks = 0
        
        while self.is_recording:
            data = self.stream.read(CHUNK)
            audio_data = array('h', data)
            current_volume = max(audio_data)
            total_chunks += 1
            
            if waiting_for_sound:
                if current_volume > SILENCE_THRESHOLD:
                    waiting_for_sound = False
                    recording_started = True
                    self.frames.append(data)
                if total_chunks > max_duration:
                    break
                continue
                
            self.frames.append(data)
            
            if current_volume < SILENCE_THRESHOLD:
                silence_count += 1
            else:
                silence_count = 0
                
            if (silence_count > SILENCE_CHUNKS and recording_started) or total_chunks > max_duration:
                break
        
        if not recording_started:
            self.frames = []
            
        self.stop_recording()
        return recording_started
    
    def save_audio(self, filename):
        if self.frames:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            return True
        return False


def get_llm_response(text):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un assistant français serviable et sympathique. Réponds en français."
                },
                {
                    "role": "user",
                    "content": text,
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Erreur avec Groq: {str(e)}"


def transcribe_audio_with_groq(audio_file):
    try:
        audio = AudioSegment.from_wav(audio_file)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            audio.export(temp_file.name, format="wav")
            with open(temp_file.name, "rb") as file:
                transcription = groq_client.audio.transcriptions.create(
                    file=(temp_file.name, file.read()),
                    model="whisper-large-v3-turbo",
                    response_format="json",
                    language="fr",
                    temperature=0.0
                )
        os.unlink(temp_file.name)
        return transcription.text
    except Exception as e:
        app.logger.error(f"Erreur lors de la transcription avec Groq : {str(e)}")
        return "Erreur lors de la transcription audio."

def transcribe_audio_with_anthropic(audio_file):
    try:
        audio = AudioSegment.from_wav(audio_file)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            audio.export(temp_file.name, format="wav")
            with open(temp_file.name, "rb") as file:
                audio_bytes = file.read()
                response = anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    messages=[{
                        "role": "user",
                        "content": "Transcrivez cet audio en français, donnez uniquement la transcription sans autre commentaire.",
                        "file_data": [{
                            "type": "audio",
                            "data": audio_bytes
                        }]
                    }],
                    temperature=0
                )
        os.unlink(temp_file.name)
        return response.content[0].text
    except Exception as e:
        app.logger.error(f"Erreur lors de la transcription avec Claude : {str(e)}")
        return "Erreur lors de la transcription audio."

def transcribe_audio(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language='fr-FR')
            return text
        except sr.UnknownValueError:
            return "Désolé, je n'ai pas compris l'audio"
        except sr.RequestError:
            return "Erreur lors de la requête au service de reconnaissance vocale"

def get_llm_response(conversation_id, user_input, max_tokens=1000):
    context = conversation_contexts.get(conversation_id, [])
    context.append({"role": "user", "content": user_input})
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Vous êtes un assistant IA utile et amical."},
                *context
            ],
            model="mixtral-8x7b-32768",
            max_tokens=max_tokens
        )
        response = chat_completion.choices[0].message.content
        context.append({"role": "assistant", "content": response})
        
        conversation_contexts[conversation_id] = context[-10:]
        
        return response
    except Exception as e:
        app.logger.error(f"Erreur lors de l'appel à l'API Groq : {str(e)}")
        return "Désolé, une erreur s'est produite lors de la génération de la réponse."

def get_anthropic_response(conversation_id, user_input, max_tokens=1000):
    context = conversation_contexts.get(conversation_id, [])
    
    try:
        messages = [{"role": "user", "content": msg["content"]} if msg["role"] == "user" 
                   else {"role": "assistant", "content": msg["content"]} 
                   for msg in context]
        messages.append({"role": "user", "content": user_input})
        
        response = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=max_tokens,
            messages=messages,
            system="Vous êtes un assistant IA utile et amical."
        )
        
        assistant_response = response.content[0].text
        context.append({"role": "user", "content": user_input})
        context.append({"role": "assistant", "content": assistant_response})
        
        conversation_contexts[conversation_id] = context[-10:]
        
        return assistant_response
    except Exception as e:
        app.logger.error(f"Erreur lors de l'appel à l'API Anthropic : {str(e)}")
        return "Désolé, une erreur s'est produite lors de la génération de la réponse."


async def text_to_speech(text):
    try:
        communicate = Communicate(text, "fr-FR-DeniseNeural")
        audio_filename = f"speech_{uuid.uuid4()}.mp3"
        audio_path = os.path.join(TEMP_AUDIO_DIR, audio_filename)
        
        await communicate.save(audio_path)
        return audio_filename
    except Exception as e:
        app.logger.error(f"Erreur lors de la synthèse vocale : {str(e)}")
        return None
    
async def text_to_speech_traduction(text):
    try:
        communicate = Communicate(text, "en-US-AriaNeural")
        audio_filename = f"speech_{uuid.uuid4()}.mp3"
        audio_path = os.path.join(TEMP_AUDIO_DIR, audio_filename)
        
        await communicate.save(audio_path)
        return audio_filename
    except Exception as e:
        app.logger.error(f"Erreur lors de la synthèse vocale : {str(e)}")
        return None

def cleanup_old_audio_files():
    try:
        current_time = time.time()
        for filename in os.listdir(TEMP_AUDIO_DIR):
            file_path = os.path.join(TEMP_AUDIO_DIR, filename)
            if os.path.getmtime(file_path) < current_time - 3600:  # 1 heure
                os.remove(file_path)
    except Exception as e:
        app.logger.error(f"Erreur lors du nettoyage des fichiers audio : {str(e)}")

# Initialisation du détecteur audio global
detector = AudioDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat2')
def chat2():
    return render_template('chat2.html', groq_api_key=GROQ_API_KEY)

@app.route('/jailbreak')
def jailbreak():
    return render_template('jailbreak.html', groq_api_key=GROQ_API_KEY)

@app.route('/chat3')
def chat3():
    return render_template('chat3.html', anthropic_api_key=ANTHROPIC_API_KEY, groq_api_key=GROQ_API_KEY)

@app.route('/translate')
def translatechat():
    return render_template('translatechat.html', anthropic_api_key=ANTHROPIC_API_KEY, groq_api_key=GROQ_API_KEY)

@app.route('/positif')
def positif():
    return render_template('positif.html', anthropic_api_key=ANTHROPIC_API_KEY, groq_api_key=GROQ_API_KEY)

@app.route('/voice')
def voice():
    return render_template('voice.html')

@app.route('/drawer.html')
def drawer_template():
    return send_from_directory('templates', 'drawer.html')

@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    conversation_id = str(uuid.uuid4())
    conversation_contexts[conversation_id] = []
    return jsonify({"conversation_id": conversation_id})

def process_audio(audio_data):
    # Convertir les données audio en fichier WAV
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_audio.write(audio_data)
        temp_audio_path = temp_audio.name
    
    # Transcrire l'audio
    recognizer = sr.Recognizer()
    with sr.AudioFile(temp_audio_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language='fr-FR')
            os.unlink(temp_audio_path)
            return text
        except Exception as e:
            os.unlink(temp_audio_path)
            return str(e)



@app.route('/process_audio', methods=['POST'])
async def process_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'Pas de fichier audio'}), 400
        
        audio_file = request.files['audio']
        
        # Convertir webm en wav
        audio_data = audio_file.read()
        audio = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
        
        # Sauvegarder en wav
        temp_path = os.path.join(RECORDINGS_DIR, f"temp_{uuid.uuid4()}.wav")
        audio.export(temp_path, format="wav")
        
        try:
            # Transcription
            transcription = transcribe_audio(temp_path)
            
            # Obtenir la réponse
            conversation_id = str(uuid.uuid4())
            llm_response = get_llm_response(conversation_id, transcription)
            
            # Générer l'audio
            audio_filename = await text_to_speech(llm_response)
            
            return jsonify({
                'status': 'success',
                'transcription': transcription,
                'llm_response': llm_response,
                'audio_file': audio_filename
            })
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
    except Exception as e:
        app.logger.error(f"Erreur lors du traitement audio : {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/process_input/anthropic', methods=['POST'])
def process_input_anthropic():
    try:
        data = request.json
        conversation_id = data.get('conversation_id')
        user_input = data.get('input')
        
        if not conversation_id:
            return jsonify({"error": "Conversation ID manquant"}), 400
        
        llm_response = get_anthropic_response(conversation_id, user_input)
        audio_file = asyncio.run(text_to_speech(llm_response))
        
        return jsonify({
            'llm_response': llm_response,
            'audio_file': audio_file if audio_file else None
        })
    except Exception as e:
        app.logger.error(f"Erreur lors du traitement de l'entrée avec Anthropic : {str(e)}")
        return jsonify({"error": "Une erreur s'est produite lors du traitement de votre message"}), 500

@app.route('/check_speaking', methods=['POST'])
def check_speaking():
    global detector
    if not detector.is_running:
        detector.start()
    is_speaking = detector.check_speaking()
    return jsonify({"speaking": is_speaking})

@app.route('/stop_detector', methods=['POST'])
def stop_detector():
    global detector
    detector.stop()
    return jsonify({"status": "stopped"})

@app.route('/start_recording', methods=['POST'])
async def start_recording():
    try:
        recorder = AudioRecorder()
        recorder.start_recording()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recordings/audio_{timestamp}.wav"
        
        if recorder.record_until_silence():
            if recorder.save_audio(filename):
                transcription = transcribe_audio(filename)
                llm_response = get_llm_response(str(uuid.uuid4()), transcription)
                
                return jsonify({
                    'status': 'success',
                    'transcription': transcription,
                    'llm_response': llm_response
                })
        
        return jsonify({
            'status': 'no_speech',
            'message': 'Aucune parole détectée'
        })
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        data = request.json
        audio_data = data['audio']
        conversation_id = data.get('conversation_id')
        
        audio_binary = base64.b64decode(audio_data.split(',')[1])
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_binary)
            temp_file_path = temp_file.name
        
        transcription = transcribe_audio_with_groq(temp_file_path)
        os.unlink(temp_file_path)
        
        llm_response = get_llm_response(conversation_id, transcription)
        audio_file = asyncio.run(text_to_speech(llm_response))
        
        return jsonify({
            'transcription': transcription,
            'llm_response': llm_response,
            'audio_file': audio_file if audio_file else None
        })
    except Exception as e:
        app.logger.error(f"Erreur lors de la transcription : {str(e)}")
        return jsonify({
            'error': 'Une erreur est survenue lors de la transcription',
            'details': str(e)
        }), 500

@app.route('/transcribe/anthropic', methods=['POST'])
def transcribe_anthropic():
    try:
        data = request.json
        audio_data = data['audio']
        conversation_id = data.get('conversation_id')
        
        audio_binary = base64.b64decode(audio_data.split(',')[1])
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_binary)
            temp_file_path = temp_file.name
        
        transcription = transcribe_audio_with_anthropic(temp_file_path)
        os.unlink(temp_file_path)
        
        llm_response = get_anthropic_response(conversation_id, transcription)
        audio_file = asyncio.run(text_to_speech(llm_response))
        
        return jsonify({
            'transcription': transcription,
            'llm_response': llm_response,
            'audio_file': audio_file if audio_file else None
        })
    except Exception as e:
        app.logger.error(f"Erreur lors de la transcription : {str(e)}")
        return jsonify({
            'error': 'Une erreur est survenue lors de la transcription',
            'details': str(e)
        }), 500

@app.route('/synthesize', methods=['POST'])
def synthesize():
    try:
        data = request.json
        text = data.get('text')
        
        if not text:
            return jsonify({"error": "Texte manquant"}), 400
        
        audio_file = asyncio.run(text_to_speech(text))
        
        if audio_file:
            return send_from_directory(TEMP_AUDIO_DIR, audio_file, mimetype="audio/mpeg")
        else:
            return jsonify({"error": "Échec de la synthèse vocale"}), 500
    
    except Exception as e:
        app.logger.error(f"Erreur lors de la synthèse vocale : {str(e)}")
        return jsonify({
            'error': 'Une erreur est survenue lors de la synthèse vocale',
            'details': str(e)
        }), 500

@app.route('/synthesizeenglish', methods=['POST'])
def synthesizeenglish():
    try:
        data = request.json
        text = data.get('text')
        
        if not text:
            return jsonify({"error": "Texte manquant"}), 400
        
        audio_file = asyncio.run(text_to_speech_traduction(text))
        
        if audio_file:
            return send_from_directory(TEMP_AUDIO_DIR, audio_file, mimetype="audio/mpeg")
        else:
            return jsonify({"error": "Échec de la synthèse vocale"}), 500
    
    except Exception as e:
        app.logger.error(f"Erreur lors de la synthèse vocale : {str(e)}")
        return jsonify({
            'error': 'Une erreur est survenue lors de la synthèse vocale',
            'details': str(e)
        }), 500


@app.route('/process_audio', methods=['POST'])
async def process_audio_route():
    try:
        # Récupérer les données audio du client
        audio_data = request.files['audio'].read()
        
        # Transcrire l'audio
        transcription = process_audio(audio_data)
        
        # Obtenir la réponse du LLM
        llm_response = get_llm_response(transcription)
        
        # Générer l'audio de réponse
        communicate = Communicate(llm_response, "fr-FR-DeniseNeural")
        audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name
        await communicate.save(audio_file)
        
        return jsonify({
            'status': 'success',
            'transcription': transcription,
            'llm_response': llm_response,
            'audio_path': audio_file
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/audio_response', methods=['POST'])
async def generate_audio_response():
    try:
        text = request.json.get('text')
        communicate = Communicate(text, "fr-FR-DeniseNeural")
        audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name
        await communicate.save(audio_file)
        return send_file(audio_file, mimetype='audio/mpeg')
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/audio/<path:filename>')
def serve_audio(filename):
    try:
        return send_from_directory(TEMP_AUDIO_DIR, filename, mimetype="audio/mpeg")
    except Exception as e:
        app.logger.error(f"Erreur lors de la lecture du fichier audio : {str(e)}")
        return jsonify({"error": "Fichier audio non trouvé"}), 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)