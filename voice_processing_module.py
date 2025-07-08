import speech_recognition as sr
import pyttsx3

# --- Text-to-Speech (TTS) Setup ---
tts_engine = None

def initialize_tts():
    """Initializes the TTS engine."""
    global tts_engine
    if tts_engine is None:
        try:
            tts_engine = pyttsx3.init()
            # You can configure properties if needed
            # tts_engine.setProperty('rate', 150)
            # tts_engine.setProperty('volume', 0.9)
            print("TTS engine initialized.")
        except Exception as e:
            print(f"Error initializing TTS engine: {e}")
            tts_engine = None # Ensure it's None if failed
    return tts_engine

def speak_response(text: str):
    """
    Converts the given text into speech.
    """
    global tts_engine
    if tts_engine is None:
        if initialize_tts() is None: # Attempt to initialize if not already
            print("TTS engine not available. Cannot speak.")
            return

    if not text:
        print("No text provided to speak.")
        return

    try:
        print(f"🤖 Speaking: {text}")
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print(f"Error during speech: {e}")

# --- Speech-to-Text (STT) Setup ---
stt_recognizer = None

def initialize_stt():
    """Initializes the STT recognizer."""
    global stt_recognizer
    if stt_recognizer is None:
        try:
            stt_recognizer = sr.Recognizer()
            print("STT recognizer initialized.")
        except Exception as e:
            print(f"Error initializing STT recognizer: {e}")
            stt_recognizer = None
    return stt_recognizer

def listen_for_command(prompt: str = "Listening...", timeout_seconds: int = 5, phrase_time_limit_seconds: int = 10) -> str | None:
    """
    Listens for a voice command from the microphone and returns it as text.
    Returns None if speech is not understood or an error occurs.
    """
    global stt_recognizer
    if stt_recognizer is None:
        if initialize_stt() is None: # Attempt to initialize if not already
            print("STT recognizer not available. Cannot listen.")
            return None

    with sr.Microphone() as source:
        # Adjust for ambient noise once at the beginning if desired,
        # but this can also be problematic if called too often or in quiet environments.
        # try:
        # stt_recognizer.adjust_for_ambient_noise(source, duration=0.5)
        # print("Adjusted for ambient noise.")
        # except Exception as e:
        #     print(f"Could not adjust for ambient noise: {e}")

        if prompt:
            print(f"🎤 {prompt}")
            # Consider speaking the prompt if a TTS engine is available and working
            # speak_response(prompt) # Be careful of recursive calls if prompt is also spoken

        audio = None
        try:
            audio = stt_recognizer.listen(source, timeout=timeout_seconds, phrase_time_limit=phrase_time_limit_seconds)
        except sr.WaitTimeoutError:
            print("No speech detected within the timeout period.")
            return None
        except Exception as e:
            print(f"Error during listening phase: {e}")
            return None

        if audio:
            try:
                print("Recognizing speech...")
                # Using Google Web Speech API by default.
                # This requires an internet connection.
                # Other engines like Sphinx can be used for offline recognition
                # but require more setup (e.g., recognizer.recognize_sphinx(audio))
                text = stt_recognizer.recognize_google(audio)
                print(f"🧑 You said: {text}")
                return text
            except sr.UnknownValueError:
                print("🤖 Google Web Speech API could not understand audio.")
                speak_response("Sorry, I didn't catch that. Could you please repeat?")
                return None
            except sr.RequestError as e:
                print(f"🤖 Could not request results from Google Web Speech API; {e}")
                speak_response("Sorry, I'm having trouble connecting to the speech service.")
                return None
            except Exception as e:
                print(f"An unexpected error occurred during speech recognition: {e}")
                return None
    return None

if __name__ == "__main__":
    print("--- Voice Processing Module Test ---")

    # Initialize engines first
    if not initialize_tts():
        print("Skipping speak test as TTS engine failed to initialize.")
    else:
        speak_response("Hello! This is a test of the text to speech system.")
        speak_response("How are you today?")

    if not initialize_stt():
        print("Skipping listen test as STT recognizer failed to initialize.")
    else:
        print("\n--- Test Listening ---")
        speak_response("Please say something for the listening test after this message.")
        user_input = listen_for_command("Say 'hello' or a short phrase (you have 10 seconds):", timeout_seconds=3, phrase_time_limit_seconds=10)

        if user_input:
            speak_response(f"I heard you say: {user_input}")
        else:
            speak_response("I didn't hear anything, or there was an error.")

        print("\n--- Test Listening for a specific command (e.g., 'exit') ---")
        speak_response("Now, try saying 'exit'.")
        command = listen_for_command("Say 'exit':")
        if command and "exit" in command.lower():
            speak_response("Exit command recognized.")
        elif command:
            speak_response(f"I heard {command}, not 'exit'.")
        else:
            speak_response("No command heard or error.")

    print("\n--- Voice Processing Module Test Complete ---")
