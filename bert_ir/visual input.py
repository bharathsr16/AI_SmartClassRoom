import cv2
import pytesseract
from openvino.runtime import Core
from transformers import DistilBertTokenizerFast
import numpy as np
import speech_recognition as sr
import pyttsx3

# Path to Tesseract executable if not in PATH
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ✅ OpenVINO model path (corrected)
model_path = r"C:\Users\srbha\PycharmProjects\open vino pythonProject7\bert_ir\distilbert_qa.xml"

# Initialize components
core = Core()
model = core.read_model(model_path)
compiled_model = core.compile_model(model, "CPU")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Inputs/Outputs
input_layer = compiled_model.input(0)
attention_mask_layer = compiled_model.input(1)
output_start = compiled_model.output(0)
output_end = compiled_model.output(1)

# ⬇️ Voice input setup
recognizer = sr.Recognizer()
engine = pyttsx3.init()


def speak(text):
    engine.say(text)
    engine.runAndWait()


def capture_question():
    with sr.Microphone() as source:
        print("🧑 Speak your question...")
        audio = recognizer.listen(source)
        try:
            question = recognizer.recognize_google(audio)
            print(f"🧑 You said: {question}")
            return question
        except sr.UnknownValueError:
            print("🤖 I didn't catch that. Please try again.")
            return None


def get_visual_context():
    print("📷 Capturing image from camera for OCR...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()
    frame = None
    # frame = cv2.imread("./img.png")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        cv2.imshow("Captured frame", frame)

        # break
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    context = pytesseract.image_to_string(gray)
    print("📝 OCR Extracted Context:")
    print(context)
    return context.strip()


def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="np", padding="max_length", truncation=True, max_length=384)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    result = compiled_model([input_ids, attention_mask])
    start_logits, end_logits = result[output_start], result[output_end]

    start_index = int(np.argmax(start_logits, axis=1)[0])
    end_index = int(np.argmax(end_logits, axis=1)[0])

    if start_index <= end_index:
        answer_ids = input_ids[0][start_index:end_index + 1]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
    else:
        answer = "Sorry, I couldn't find a clear answer."

    print("🤖", answer)
    speak(answer)


# 🧠 Main loop
if __name__ == "__main__":
    context = get_visual_context()
    while True:
        question = capture_question()
        if question and question.lower() == "exit":
            break
        elif question:
            answer_question(question, context)
