import cv2
import time
import os # For checking HAAR_CASCADE_PATH

# Attempt to import 'fer' library
try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    # print("FER library not found. Facial expression analysis will be limited to face detection only (if OpenCV is working).")
    FER_AVAILABLE = False # Keep it silent for now, main_assistant will report
except Exception:
    # print(f"Error importing FER: {e}. Facial expression analysis will be limited.")
    FER_AVAILABLE = False


HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"
face_detector_cv = None

def initialize_face_detector_cv():
    global face_detector_cv
    if face_detector_cv is None:
        if not os.path.exists(HAAR_CASCADE_PATH):
            # This message will be shown if main_assistant doesn't catch it first
            # print(f"Error: Haar Cascade file '{HAAR_CASCADE_PATH}' not found. Face detection will not work.")
            return False
        try:
            face_detector_cv = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
            if face_detector_cv.empty():
                # print(f"Error: Failed to load Haar Cascade from {HAAR_CASCADE_PATH}. Face detection will not work.")
                face_detector_cv = None
                return False
            # print("OpenCV Haar Cascade face detector loaded.")
            return True
        except Exception:
            # print(f"Exception loading Haar Cascade: {e}")
            face_detector_cv = None
            return False
    return True


emotion_detector_fer = None

def initialize_emotion_detector_fer():
    global emotion_detector_fer
    if not FER_AVAILABLE:
        # print("Cannot initialize FER emotion detector: FER library is not available.")
        return False

    if emotion_detector_fer is None:
        # print("Loading FER model for emotion detection...")
        try:
            emotion_detector_fer = FER(mtcnn=False)
            # print("FER emotion detector loaded successfully.")
            return True
        except Exception:
            # print(f"Error loading FER emotion detector: {e}")
            emotion_detector_fer = None
            return False
    return True

# --- Intervention Logic ---
# Global or class members to track state over time for intervention
last_detected_emotion_intervention = None
emotion_start_time_intervention = None
NO_FACE_THRESHOLD_SECONDS_INTERVENTION = 10
SAD_THRESHOLD_SECONDS_INTERVENTION = 7
CONFUSED_THRESHOLD_SECONDS_INTERVENTION = 5 # Example for 'surprise' as proxy
no_face_start_time_intervention = None

# Store intervention messages to avoid repetition in a short time
last_intervention_time = {}
INTERVENTION_COOLDOWN_SECONDS = 30 # Cooldown per intervention type

def _trigger_intervention(reason: str):
    global last_intervention_time
    current_time = time.time()

    if current_time - last_intervention_time.get(reason, 0) < INTERVENTION_COOLDOWN_SECONDS:
        # print(f"Intervention for '{reason}' is on cooldown.")
        return # Cooldown active for this reason

    print(f"INTERVENTION TRIGGERED: {reason}")
    last_intervention_time[reason] = current_time

    intervention_message = ""
    if reason == "confusion_surprise":
        intervention_message = "It looks like that might have been a bit surprising or confusing. Would you like me to explain it differently?"
    elif reason == "disengagement_sad":
        intervention_message = "Is everything alright? You seem a bit down. Maybe we can try another approach?"
    elif reason == "disengagement_no_face":
        intervention_message = "Are you still there? I can't seem to see you."
    else:
        intervention_message = f"I've noticed {reason}. Let's see if we can adjust."

    try:
        # This is a direct call for conceptual demonstration.
        # In a real app, this would be a callback or event to main_assistant.
        import voice_processing_module
        if voice_processing_module.initialize_tts():
            voice_processing_module.speak_response(intervention_message)
        else:
            print(f"TTS not available for intervention: {intervention_message}")
    except ImportError:
        print(f"voice_processing_module not found for intervention speech: {intervention_message}")
    except Exception as e:
        print(f"Error during intervention speech for '{intervention_message}': {e}")


def analyze_engagement_in_frame(frame, draw_on_frame=True):
    global last_detected_emotion_intervention, emotion_start_time_intervention, no_face_start_time_intervention
    results = []

    if face_detector_cv is None: initialize_face_detector_cv() # Ensure it's tried

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces_cv = []
    if face_detector_cv and not face_detector_cv.empty():
        detected_faces_cv = face_detector_cv.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

    current_time = time.time()
    active_fer_session = FER_AVAILABLE and emotion_detector_fer is not None

    # --- FER Path (if available) ---
    if active_fer_session:
        try:
            fer_results = emotion_detector_fer.detect_emotions(frame)
            if fer_results: # If FER detected faces
                if no_face_start_time_intervention: no_face_start_time_intervention = None # Face re-detected by FER

                for fer_res in fer_results:
                    x, y, w, h = fer_res["box"]
                    emotions = fer_res["emotions"]
                    dominant_emotion = max(emotions, key=emotions.get).lower()
                    score = emotions[dominant_emotion]
                    results.append({"box": (x, y, w, h), "emotions": emotions, "detector": "fer"})

                    # Intervention logic for the first detected person by FER
                    if not results or fer_res == fer_results[0]: # Process first face for intervention
                        if dominant_emotion != last_detected_emotion_intervention:
                            last_detected_emotion_intervention = dominant_emotion
                            emotion_start_time_intervention = current_time
                            # print(f"Emotion (FER) changed to: {last_detected_emotion_intervention}")
                        else:
                            duration = current_time - emotion_start_time_intervention
                            if dominant_emotion == "sad" and duration > SAD_THRESHOLD_SECONDS_INTERVENTION:
                                _trigger_intervention("disengagement_sad")
                                emotion_start_time_intervention = current_time
                            elif dominant_emotion == "surprise" and duration > CONFUSED_THRESHOLD_SECONDS_INTERVENTION: # Using surprise as proxy for confusion
                                _trigger_intervention("confusion_surprise")
                                emotion_start_time_intervention = current_time

                    if draw_on_frame:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, f"{dominant_emotion}: {score:.2f}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                return results, frame # Return early if FER processed
            else: # FER active but detected no faces
                if no_face_start_time_intervention is None: no_face_start_time_intervention = current_time
                else:
                    if current_time - no_face_start_time_intervention > NO_FACE_THRESHOLD_SECONDS_INTERVENTION:
                        _trigger_intervention("disengagement_no_face")
                        no_face_start_time_intervention = current_time
                last_detected_emotion_intervention = None


        except Exception: # Fallback if FER errors out at runtime
            # print(f"Runtime error during FER emotion detection: {e_fer}")
            active_fer_session = False # Disable FER for subsequent calls in this session

    # --- OpenCV Haar Cascade Path (if FER not available/failed or detected nothing) ---
    if not active_fer_session or not results: # If FER didn't run or found no faces
        if detected_faces_cv is not None and len(detected_faces_cv) > 0:
            if no_face_start_time_intervention: no_face_start_time_intervention = None # Face re-detected by Haar

            for (x, y, w, h) in detected_faces_cv:
                results.append({"box": (x, y, w, h), "emotions": "N/A (FER not used)", "detector": "opencv_haar"})
                if draw_on_frame:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Face", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            last_detected_emotion_intervention = None # No emotion data from Haar
        else: # No faces by Haar either
            if no_face_start_time_intervention is None: no_face_start_time_intervention = current_time
            else:
                if current_time - no_face_start_time_intervention > NO_FACE_THRESHOLD_SECONDS_INTERVENTION:
                    _trigger_intervention("disengagement_no_face")
                    no_face_start_time_intervention = current_time
            last_detected_emotion_intervention = None

    return results, frame


def monitor_engagement_from_webcam(show_preview=True):
    global last_detected_emotion_intervention, emotion_start_time_intervention, no_face_start_time_intervention, last_intervention_time

    haar_init_ok = initialize_face_detector_cv()
    fer_init_ok = False
    if FER_AVAILABLE:
        fer_init_ok = initialize_emotion_detector_fer()

    if not haar_init_ok and not (FER_AVAILABLE and fer_init_ok):
        print("Engagement Module: Neither OpenCV face detector nor FER could be initialized. Cannot monitor engagement.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Engagement Module: Error: Could not open webcam.")
        return

    # print("Engagement Module: Monitoring engagement from webcam. Press 'q' to quit.")
    # Reset states at the start of monitoring
    last_detected_emotion_intervention = None
    emotion_start_time_intervention = time.time()
    no_face_start_time_intervention = None
    last_intervention_time = {} # Reset cooldowns

    # For FPS calculation
    # frame_count = 0
    # overall_start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Engagement Module: Error: Failed to grab frame.")
            break

        # frame_count += 1
        _engagement_data, processed_frame = analyze_engagement_in_frame(frame.copy(), draw_on_frame=show_preview)

        if show_preview:
            # end_time = time.time()
            # fps = frame_count / (end_time - overall_start_time) if (end_time - overall_start_time) > 0 else 0
            # cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow("Engagement Monitor", processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # print("Engagement Module: Quitting engagement monitor.")
            break

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("--- Engagement Analysis Module Test (with Intervention Logic) ---")

    if not os.path.exists(HAAR_CASCADE_PATH):
        print(f"Please ensure '{HAAR_CASCADE_PATH}' is in the same directory as this script.")
        print("Download from: https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml")
        print("Engagement monitoring will likely fail or be very limited without it.")

    show_preview_main = os.environ.get('DISPLAY') is not None
    if not show_preview_main:
        print("No display environment detected, webcam preview will be disabled for the main test.")

    print("\n--- Test 1: Monitor Engagement from Webcam (with intervention logic) ---")
    try:
        monitor_engagement_from_webcam(show_preview=show_preview_main)
    except Exception as e:
        print(f"Webcam monitoring test failed: {e}")

    print("\n--- Engagement Analysis Module Test Complete ---")
