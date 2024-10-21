import os
from pydub import AudioSegment
import speech_recognition as sr
from groq import Groq
import time
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, send_file
import base64
import logging
import tempfile
import uuid
import asyncio
from edge_tts import Communicate

# Configuration du logging
logging.basicConfig(level=logging.DEBUG)

# Charger les variables d'environnement
load_dotenv()

app = Flask(__name__)

# Configuration Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("La clé API Groq n'est pas définie dans le fichier .env")
groq_client = Groq(api_key=GROQ_API_KEY)

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

def get_llm_response(conversation_id, user_input, max_tokens=1000):
    context = conversation_contexts.get(conversation_id, [])
    context.append({"role": "user", "content": user_input})
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Vous êtes un assistant IA utile et amical."},
                *context
            ],
            model="llama3-8b-8192",
            max_tokens=max_tokens
        )
        response = chat_completion.choices[0].message.content
        context.append({"role": "assistant", "content": response})
        
        # Limiter le contexte aux 10 derniers messages
        conversation_contexts[conversation_id] = context[-10:]
        
        return response
    except Exception as e:
        app.logger.error(f"Erreur lors de l'appel à l'API Groq : {str(e)}")
        return "Désolé, une erreur s'est produite lors de la génération de la réponse."

async def text_to_speech(text):
    try:
        communicate = Communicate(text, "fr-FR-DeniseNeural")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            await communicate.save(temp_file.name)
            return temp_file.name
    except Exception as e:
        app.logger.error(f"Erreur lors de la synthèse vocale : {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat2')
def chat2():
     return render_template('chat2.html', groq_api_key=GROQ_API_KEY)
 
@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    conversation_id = str(uuid.uuid4())
    conversation_contexts[conversation_id] = []
    return jsonify({"conversation_id": conversation_id})

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
            'audio_file': os.path.basename(audio_file) if audio_file else None
        })
    except Exception as e:
        app.logger.error(f"Erreur lors du traitement de l'entrée : {str(e)}")
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
            'audio_file': os.path.basename(audio_file) if audio_file else None
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
            return send_file(audio_file, mimetype="audio/mpeg")
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
        return send_file(filename, mimetype="audio/mpeg")
    except Exception as e:
        app.logger.error(f"Erreur lors de la lecture du fichier audio : {str(e)}")
        return jsonify({"error": "Fichier audio non trouvé"}), 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)