import speech_recognition as sr
import time

# Function to convert audio from the microphone and convert it to text
def listen_to_audio():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Please speak for at least 60 seconds...")
        recognizer.adjust_for_ambient_noise(source)
        
        start_time = time.time()
        
        audio = None
        while time.time() - start_time < 60:
            try:
                print("Listening...")
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=None)
                break
            except sr.WaitTimeoutError:
                pass

    if audio is None:
        print("No audio captured within 60 seconds.")
        return ""
    
    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print(text)
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return ""
    except sr.RequestError:
        print("Could not request results; check your network connection.")
        return ""