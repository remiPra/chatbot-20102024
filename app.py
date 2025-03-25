import os
from pydub import AudioSegment
import speech_recognition as sr
from groq import Groq
import time
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import base64
import logging
import tempfile
import uuid
import asyncio
from edge_tts import Communicate
from anthropic import Anthropic
from flask_cors import CORS


# Configuration du logging
logging.basicConfig(level=logging.DEBUG)

# Charger les variables d'environnement
load_dotenv()

app = Flask(__name__)
CORS(app)  # Ceci permet toutes les origines

# Création du dossier pour les fichiers audio
TEMP_AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_audio')
if not os.path.exists(TEMP_AUDIO_DIR):
    os.makedirs(TEMP_AUDIO_DIR)

# Configuration Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("La clé API Groq n'est pas définie dans le fichier .env")
groq_client = Groq(api_key=GROQ_API_KEY)

# Configuration Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("La clé API Anthropic n'est pas définie dans le fichier .env")
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Dictionnaire pour stocker les contextes de conversation
conversation_contexts = {}

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
        communicate = Communicate(text, "fr‑FR‑DeniseNeural")
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
    return render_template('chat3.html', anthropic_api_key=ANTHROPIC_API_KEY,groq_api_key=GROQ_API_KEY)

@app.route('/translate')
def translatechat():
    return render_template('translatechat.html', anthropic_api_key=ANTHROPIC_API_KEY,groq_api_key=GROQ_API_KEY)

@app.route('/positif')
def positif():
    return render_template('positif.html', anthropic_api_key=ANTHROPIC_API_KEY,groq_api_key=GROQ_API_KEY)

@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    conversation_id = str(uuid.uuid4())
    conversation_contexts[conversation_id] = []
    return jsonify({"conversation_id": conversation_id})

# Route originale pour Groq
@app.route('/process_input', methods=['POST'])
def process_input():
    try:
        data = request.json
        conversation_id = data.get('conversation_id')
        user_input = data.get('input')
        
        if not conversation_id:
            return jsonify({"error": "Conversation ID manquant"}), 400
        
        llm_response = get_llm_response(conversation_id, user_input)
        audio_file = asyncio.run(text_to_speech(llm_response))
        
        return jsonify({
            'llm_response': llm_response,
            'audio_file': audio_file if audio_file else None
        })
    except Exception as e:
        app.logger.error(f"Erreur lors du traitement de l'entrée : {str(e)}")
        return jsonify({"error": "Une erreur s'est produite lors du traitement de votre message"}), 500

# Nouvelle route pour Anthropic
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

@app.route('/drawer.html')
def drawer_template():
    return send_from_directory('templates', 'drawer.html')

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

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    try:
        return send_from_directory(TEMP_AUDIO_DIR, filename, mimetype="audio/mpeg")
    except Exception as e:
        app.logger.error(f"Erreur lors de la lecture du fichier audio : {str(e)}")
        return jsonify({"error": "Fichier audio non trouvé"}), 404

def cleanup_old_audio_files():
    try:
        current_time = time.time()
        for filename in os.listdir(TEMP_AUDIO_DIR):
            file_path = os.path.join(TEMP_AUDIO_DIR, filename)
            if os.path.getmtime(file_path) < current_time - 3600:  # 1 heure
                os.remove(file_path)
    except Exception as e:
        app.logger.error(f"Erreur lors du nettoyage des fichiers audio : {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
