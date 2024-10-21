import os
import pyaudio
import wave
import audioop
from collections import deque
import speech_recognition as sr
from groq import Groq
import time
from dotenv import load_dotenv
import requests
import asyncio
from edge_tts import Communicate
import pygame
import io
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timedelta
import re

# Charger les variables d'environnement
load_dotenv()

# Initialisation de Firebase
cred = credentials.Certificate(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialisation de pygame pour l'audio
pygame.mixer.init()

# Paramètres audio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1
PHRASE_TIMEOUT = 10

# Configuration Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("La clé API Groq n'est pas définie dans le fichier .env")
groq_client = Groq(api_key=GROQ_API_KEY)

async def text_to_speech(text):
    communicate = Communicate(text, "fr-FR-DeniseNeural")
    audio_stream = io.BytesIO()
    
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_stream.write(chunk["data"])
    
    audio_stream.seek(0)
    
    # Charger et jouer l'audio avec pygame
    pygame.mixer.music.load(audio_stream)
    pygame.mixer.music.play()
    
    # Attendre que l'audio soit terminé
    while pygame.mixer.music.get_busy():
        await asyncio.sleep(0.1)

def is_silent(data_chunk):
    rms = audioop.rms(data_chunk, 2)
    return rms < SILENCE_THRESHOLD

def record_phrase(p, timeout=PHRASE_TIMEOUT):
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("En attente de parole...")
    audio_buffer = []
    silence_buffer = deque(maxlen=int(SILENCE_DURATION * RATE / CHUNK))
    is_speaking = False
    timeout_counter = 0

    while True:
        try:
            data = stream.read(CHUNK)
            audio_buffer.append(data)

            if is_silent(data):
                if is_speaking:
                    silence_buffer.append(True)
                if len(silence_buffer) == silence_buffer.maxlen and all(silence_buffer):
                    break
            else:
                silence_buffer.clear()
                is_speaking = True
                timeout_counter = 0

            timeout_counter += 1
            if timeout_counter > timeout * (RATE / CHUNK):
                print("Temps d'attente dépassé.")
                break
        except KeyboardInterrupt:
            print("Enregistrement interrompu par l'utilisateur.")
            break

    print("Enregistrement terminé.")
    stream.stop_stream()
    stream.close()

    if not audio_buffer:
        return None

    filename = f"output_{int(time.time())}.wav"
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(audio_buffer))
    wf.close()

    return filename

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language="fr-FR")
        return text
    except sr.UnknownValueError:
        return "Désolé, je n'ai pas pu comprendre l'audio."
    except sr.RequestError:
        return "Désolé, il y a eu une erreur lors de la requête au service de reconnaissance vocale."

def validate_phone_number(phone_number):
    digits_only = re.sub(r'\D', '', phone_number)
    return len(digits_only) == 10

def validate_and_respond(key, user_input, question):
    if key == "telephone":
        if validate_phone_number(user_input):
            return True, f"Merci, j'ai bien enregistré votre numéro de téléphone : {user_input}"
        else:
            return False, "Ce numéro de téléphone ne semble pas valide. Assurez-vous qu'il contient 10 chiffres."

    prompt = f"""
    Question posée : "{question}"
    Réponse de l'utilisateur : "{user_input}"
    
    Tâche :
    1. Déterminez si la réponse est valide pour la question posée sur le {key}.
    2. Si la réponse est valide, répondez "VALIDE: [Réponse de confirmation]".
    3. Si la réponse n'est pas valide, répondez "INVALIDE: [Explication du problème]".
    
    Votre réponse :
    """
    
    max_retries = 3
    retry_delay = 5  # secondes

    for attempt in range(max_retries):
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}],
                model="llama3-groq-70b-8192-tool-use-preview",
                max_tokens=150
            )

            response = chat_completion.choices[0].message.content.strip()

            if response.upper().startswith('VALIDE:'):
                return True, response[7:].strip()  # Retourne la partie après "VALIDE:"
            elif response.upper().startswith('INVALIDE:'):
                return False, response[9:].strip()  # Retourne la partie après "INVALIDE:"
            else:
                return False, "Je n'ai pas pu valider votre réponse. Pouvez-vous réessayer ?"
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la requête à Groq (tentative {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Nouvelle tentative dans {retry_delay} secondes...")
                time.sleep(retry_delay)
            else:
                print("Échec de la connexion à Groq après plusieurs tentatives.")
                return False, "Je n'ai pas pu valider votre réponse en raison de problèmes techniques. Pouvez-vous réessayer ?"

    return False, "Je n'ai pas pu valider votre réponse en raison de problèmes techniques. Pouvez-vous réessayer ?"

async def check_available_appointments():
    available_slots = []
    today = datetime.now()
    for i in range(28):  # Vérifier les 4 prochaines semaines
        date = (today + timedelta(days=i)).strftime('%Y-%m-%d')
        doc = db.collection('appointments').document(date).get()
        if doc.exists:
            slots = doc.to_dict().get('slots', {})
            for time, info in slots.items():
                if not info.get('booked', True):  # Considérer comme réservé si 'booked' n'existe pas
                    available_slots.append((date, time))
    
    return available_slots

async def get_available_dates(available_slots):
    dates = sorted(set(date for date, _ in available_slots))
    return dates[:5]  # Limiter à 5 dates pour ne pas surcharger la conversation

async def present_available_dates(dates):
    response = "Nous avons des rendez-vous disponibles les jours suivants : "
    for date in dates:
        response += f"le {date}, "
    response += "Quel jour vous conviendrait le mieux ?"
    print(response)
    await text_to_speech(response)

async def get_patient_date_choice(p):
    while True:
        audio_file = record_phrase(p)
        if not audio_file:
            print("Aucun audio enregistré.")
            continue
        
        transcription = transcribe_audio(audio_file)
        os.remove(audio_file)
        
        # Utiliser Groq pour interpréter la réponse du patient
        interpretation = await interpret_patient_response(transcription, "date")
        
        if interpretation['understood']:
            return interpretation['chosen_date']
        else:
            await text_to_speech(interpretation['response'])

async def present_available_times(date, available_slots):
    times = [time for d, time in available_slots if d == date]
    response = f"Pour le {date}, nous avons les horaires suivants : "
    for time in times:
        response += f"{time}, "
    response += "Quelle heure vous conviendrait le mieux ?"
    print(response)
    await text_to_speech(response)

async def get_patient_time_choice(p, date):
    while True:
        audio_file = record_phrase(p)
        if not audio_file:
            print("Aucun audio enregistré.")
            continue
        
        transcription = transcribe_audio(audio_file)
        os.remove(audio_file)
        
        # Utiliser Groq pour interpréter la réponse du patient
        interpretation = await interpret_patient_response(transcription, "time")
        
        if interpretation['understood']:
            return interpretation['chosen_time']
        else:
            await text_to_speech(interpretation['response'])

async def interpret_patient_response(response, context):
    prompt = f"""
    Contexte : Le patient répond à une question sur le choix d'un {context} pour un rendez-vous.
    Réponse du patient : "{response}"
    
    Tâches :
    1. Déterminez si la réponse contient une information claire sur le choix du {context}.
    2. Si oui, extrayez le {context} choisi.
    3. Si non, formulez une réponse pour demander plus de précisions.
    4. Si le patient demande de répéter ou semble confus, proposez de répéter les options.
    
    Format de réponse :
    {{
        "understood": bool,
        "chosen_{context}": str ou null,
        "response": str
    }}
    """
    
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "system", "content": prompt}],
        model="llama3-groq-70b-8192-tool-use-preview",
        max_tokens=150
    )

    interpretation = eval(chat_completion.choices[0].message.content.strip())
    return interpretation

async def confirm_appointment(p, date, time):
    confirmation = f"Voulez-vous confirmer le rendez-vous du {date} à {time} ? Répondez par oui ou non."
    print(confirmation)
    await text_to_speech(confirmation)
    
    while True:
        audio_file = record_phrase(p)
        if not audio_file:
            print("Aucun audio enregistré.")
            continue
        
        response = transcribe_audio(audio_file).lower()
        os.remove(audio_file)
        
        if "oui" in response:
            return True
        elif "non" in response:
            return False
        else:
            await text_to_speech("Je n'ai pas compris. Veuillez répondre par oui ou non.")

def save_appointment_to_firebase(patient_info, date, time):
    appointment_ref = db.collection('appointments').document(date)
    appointment_ref.update({
        f'slots.{time}.booked': True,
        f'slots.{time}.patientName': f"{patient_info['prenom']} {patient_info['nom']}",
        f'slots.{time}.patientPhone': patient_info['telephone']
    })
    print(f"Rendez-vous enregistré pour {patient_info['prenom']} {patient_info['nom']} le {date} à {time}")

async def ask_retry(p):
    await text_to_speech("Souhaitez-vous choisir un autre créneau ? Répondez par oui ou non.")
    while True:
        audio_file = record_phrase(p)
        if not audio_file:
            print("Aucun audio enregistré.")
            continue
        
        response = transcribe_audio(audio_file).lower()
        os.remove(audio_file)
        
        if "oui" in response:
            return True
        elif "non" in response:
            return False
        else:
            await text_to_speech("Je n'ai pas compris. Veuillez répondre par oui ou non.")

async def play_audio_file(file_name):
    pygame.mixer.music.load(file_name)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        await asyncio.sleep(0.1)

async def appointment_booking_flow(patient_info, p):
    while True:
        available_slots = await check_available_appointments()
        available_dates = await get_available_dates(available_slots)
        
        await present_available_dates(available_dates)
        chosen_date = await get_patient_date_choice(p)
        
        if chosen_date not in available_dates:
            await text_to_speech("Je suis désolé, je n'ai pas bien compris la date choisie. Pouvons-nous recommencer ?")
            continue
        
        await present_available_times(chosen_date, available_slots)
        chosen_time = await get_patient_time_choice(p, chosen_date)
        
        if (chosen_date, chosen_time) not in available_slots:
            await text_to_speech("Je suis désolé, cet horaire n'est pas disponible. Pouvons-nous choisir un autre horaire ?")
            continue
        
        confirmed = await confirm_appointment(p, chosen_date, chosen_time)
        
        if confirmed:
            save_appointment_to_firebase(patient_info, chosen_date, chosen_time)
            confirmation_message = f"Parfait ! Votre rendez-vous est confirmé pour le {chosen_date} à {chosen_time}. Merci de votre confiance."
            await text_to_speech(confirmation_message)
            return True
        else:
            retry = await ask_retry(p)
            if not retry:
                await text_to_speech("D'accord, merci d'avoir utilisé notre service. Au revoir.")
                return False

async def conversation_flow():
    # Jouer le fichier d'introduction
    await play_audio_file("presentation.mp3")
    
    p = pyaudio.PyAudio()
    patient_info = {}

    steps = [
        {"question": "Pourriez-vous me donner votre nom, s'il vous plaît ?", "key": "nom"},
        {"question": "Merci. Quel est votre prénom ?", "key": "prenom"},
        {"question": "Merci, et enfin, pourriez-vous me donner votre numéro de portable ? Assurez-vous qu'il contienne 10 chiffres.", "key": "telephone"}
    ]

    for step in steps:
        attempts = 0
        while attempts < 3:  # Limite le nombre de tentatives à 3
            if attempts == 0:
                question = step['question']
            else:
                question = f"Je n'ai pas reçu une réponse valide. Pouvez-vous me redonner votre {step['key']} s'il vous plaît ?"
            
            print(f"\nQuestion : {question}")
            await text_to_speech(question)
            
            try:
                audio_file = record_phrase(p)
                if not audio_file:
                    print("Aucun audio enregistré.")
                    continue

                transcription = transcribe_audio(audio_file)
                print(f"Transcription: {transcription}")
                os.remove(audio_file)

                is_valid, response = validate_and_respond(step['key'], transcription, question)
                
                print(response)
                await text_to_speech(response)

                if is_valid:
                    patient_info[step['key']] = transcription
                    break
                
                attempts += 1
            except KeyboardInterrupt:
                print("\nInterruption détectée. Passage à l'étape suivante.")
                break

        if attempts == 3:
            print(f"Nombre maximum de tentatives atteint pour {step['key']}. Passage à l'étape suivante.")

    # Rechercher les rendez-vous disponibles pendant que l'audio d'attente est joué
    audio_task = asyncio.create_task(play_audio_file("attenrdv.mp3"))
    appointments_task = asyncio.create_task(check_available_appointments())
    
    await audio_task
    available_slots = await appointments_task

    # Lancer le flux de réservation de rendez-vous
    appointment_booked = await appointment_booking_flow(patient_info, p)

    p.terminate()
    return patient_info, appointment_booked

if __name__ == "__main__":
    try:
        patient_info, appointment_booked = asyncio.run(conversation_flow())
        if appointment_booked:
            print("Rendez-vous pris avec succès pour :", patient_info)
        else:
            print("Aucun rendez-vous n'a été pris.")
    except KeyboardInterrupt:
        print("\nProgramme interrompu par l'utilisateur.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")
    finally:
        pygame.mixer.quit()