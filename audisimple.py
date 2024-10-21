import asyncio
from edge_tts import Communicate

async def main():
    communicate = Communicate(f"""
Merci beaucoup pour ces informations. J'ai bien noté votre nom, votre prénom, ainsi que votre numéro de téléphone portable. Ce dernier me permettra de vous envoyer un rappel pour votre rendez-vous. Je suis maintenant en train de rechercher les créneaux horaires disponibles pour la consultation.                              
                              

""", 
"fr-FR-DeniseNeural")
    await communicate.save("attenrdv.mp3")

if __name__ == "__main__":
    asyncio.run(main())