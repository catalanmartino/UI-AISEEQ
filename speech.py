import os
import threading

import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

load_dotenv()

# Set up Azure Speech-to-Text and Text-to-Speech credentials
speech_key = os.getenv("SPEECH_API_KEY")
service_region = "eastus"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
# Set up Azure Text-to-Speech language
speech_config.speech_synthesis_language = "en-NZ"
# Set up Azure Speech-to-Text language recognition
speech_config.speech_recognition_language = "en-NZ"

# Set up the voice configuration
speech_config.speech_synthesis_voice_name = "en-NZ-MollyNeural"
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)


# Define the text-to-speech function
def text_to_speech(text):
    def speak():
        try:
            result = speech_synthesizer.speak_text_async(text).get()
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                print("Text-to-speech conversion successful.")
                return True
            else:
                print(f"Error synthesizing audio: {result}")
                return False
        except Exception as ex:
            print(f"Error synthesizing audio: {ex}")
            return False

    speak_thread = threading.Thread(target=speak)
    speak_thread.start()
