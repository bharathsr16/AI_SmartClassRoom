import speech_recognition as sr
import pyttsx3
import numpy as np
from transformers import AutoTokenizer
from openvino.runtime import Core
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Initialize speech recognizer
recognizer = sr.Recognizer()

def listen(prompt=None):
    with sr.Microphone() as source:
        if prompt:
            print(f"🧑 {prompt}")
            speak(prompt)
        print("🎤 Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"🧑 {text}")
            return text
        except sr.UnknownValueError:
            print("🤖 Could not understand.")
            speak("Sorry, I didn't understand that.")
            return None
        except sr.RequestError:
            print("🤖 Speech recognition `` failed.")
            return None

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")

core = Core()

# ✅ Replace with the correct full path to your model
model_path = "C:/Users/srbha/PycharmProjects/open vino pythonProject7/bert_ir/distilbert_qa.xml"
model = core.read_model(model_path)
compiled_model = core.compile_model(model, "CPU")

input_names = [key.get_any_name() for key in model.inputs]
output_names = [key.get_any_name() for key in model.outputs]

while True:
    # Step 1: Get context paragraph
    context = listen("Please speak your paragraph context.")
    if not context:
        continue

    while True:
        # Step 2: Ask question
        question = listen("Ask your question or say 'new context' to change paragraph.")
        if question is None:
            continue
        if question.lower() in ['exit', 'quit']:
            speak("Goodbye!")
            break
        if "new context" in question.lower():
            break

        # Step 3: Tokenize inputs
        inputs = tokenizer(question, context, return_tensors="np", padding=True)

        # Step 4: Run inference
        outputs = compiled_model({input_names[0]: inputs["input_ids"], input_names[1]: inputs["attention_mask"]})
        start_logits = outputs[output_names[0]]
        end_logits = outputs[output_names[1]]

        # Step 5: Get answer span
        start_index = int(np.argmax(start_logits[0]))
        end_index = int(np.argmax(end_logits[0]))

        if start_index <= end_index:
            answer = tokenizer.decode(inputs["input_ids"][0][start_index:end_index+1], skip_special_tokens=True)
        else:
            answer = "Sorry, I couldn't find an answer."

        print(f"🤖 {answer}")
        speak(answer)

    if question and question.lower() in ['exit', 'quit']:
        break
