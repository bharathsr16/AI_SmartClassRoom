# Import the modules created earlier
import threading

import text_qa_module
import voice_processing_module
import visual_module
import engagement_module  # Will likely only have limited functionality

# For system interaction, e.g., exit
import sys
import os  # For checking display for OpenCV windows
from app_ui import main

# State management (very basic)
current_context = "No specific context has been set yet."
last_image_caption = None


def initialize_all_modules():
    """
    Attempts to initialize all core modules.
    """
    print("Initializing AI Assistant Modules...")

    # Text QA
    try:
        if text_qa_module.initialize_qa_model():
            print("Text QA module initialized successfully.")
        else:
            # This path should ideally not be hit if initialize_qa_model raises on critical failure
            print("Text QA module reported non-critical initialization issue.")
    except Exception as e:
        print(f"Text QA module failed to initialize: {e}")

    # Voice Processing
    try:
        if not voice_processing_module.initialize_tts():
            print("TTS could not be initialized. Spoken responses will be unavailable.")
        else:
            print("TTS engine initialized.")
        if not voice_processing_module.initialize_stt():
            print("STT could not be initialized. Voice input will be unavailable.")
        else:
            print("STT recognizer initialized.")
    except Exception as e:
        print(f"Voice processing module failed to initialize: {e}")

    # Visual Module
    try:
        # Check if visual_module has TRANSFORMERS_AVAILABLE, default to False if module itself is missing
        transformers_available_in_visual = getattr(visual_module, 'TRANSFORMERS_AVAILABLE', False)
        if transformers_available_in_visual:
            if not visual_module.initialize_caption_model():
                print("Visual captioning model could not be initialized.")
            else:
                print("Image captioning model loaded successfully.")
        else:
            print("Visual module: Transformers (for captioning) not available or visual_module issue.")
            print("Basic image properties might still work if OpenCV is functional in visual_module.")
    except Exception as e:
        print(f"Visual module failed to initialize: {e}")

    # Engagement Module
    try:
        haar_cascade_path = "haarcascade_frontalface_default.xml"
        if not os.path.exists(haar_cascade_path):
            print(f"WARNING: Haar cascade file '{haar_cascade_path}' not found.")
            print("Face detection in the engagement module will not work without it.")
            print(
                "Download from: https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml")

        if not engagement_module.initialize_face_detector_cv():
            print("Engagement module's OpenCV face detector could not be initialized.")
        # Check if engagement_module has FER_AVAILABLE attribute
        fer_available_in_engagement = getattr(engagement_module, 'FER_AVAILABLE', False)
        if fer_available_in_engagement:
            if not engagement_module.initialize_emotion_detector_fer():
                print("Engagement module's FER emotion detector could not be initialized.")
        else:
            print("Engagement module: FER (for emotion detection) not available or engagement_module issue.")
    except Exception as e:
        print(f"Engagement module failed to initialize: {e}")

    print("\nAll modules initialization attempt complete.")
    print("-----------------------------------------")


def handle_text_query(question, use_voice_output=False):
    global current_context
    print(f"\n🧑 You asked (text): {question}")
    print(f"Context: {current_context}")

    answer = "Error: Text QA module could not produce an answer."
    try:
        answer = text_qa_module.answer_text_question(question, current_context)
    except Exception as e:
        answer = f"Error during text QA call: {e}"

    print(f"🤖 Assistant: {answer}")
    try:
        if use_voice_output and voice_processing_module.tts_engine:
            voice_processing_module.speak_response(answer)
    except Exception as e_tts:
        if use_voice_output: print(f"🤖 TTS Error: {e_tts}. Would speak: {answer}")
    return answer


def handle_voice_query():
    global current_context
    question = None
    try:
        if not voice_processing_module.stt_recognizer:
            print("Voice input is not available (STT recognizer not initialized).")
            if voice_processing_module.tts_engine: voice_processing_module.speak_response(
                "Sorry, my voice input is not working right now.")
            return
        if voice_processing_module.tts_engine:
            voice_processing_module.speak_response("What is your question?")
        else:
            print("🤖 What is your question? (TTS unavailable)")
        question = voice_processing_module.listen_for_command(prompt="Listening for your question...")
    except Exception as e_stt:
        print(f"Voice input error: {e_stt}.")
        question = input("Voice input failed. Type your question: ")

    if question:
        print(f"\n🧑 You asked (voice): {question}")
        print(f"Context: {current_context}")
        answer = "Error: Text QA module could not produce an answer."
        try:
            answer = text_qa_module.answer_text_question(question, current_context)
        except Exception as e:
            answer = f"Error during text QA call: {e}"

        print(f"🤖 Assistant: {answer}")
        try:
            if voice_processing_module.tts_engine: voice_processing_module.speak_response(answer)
        except Exception as e_tts:
            print(f"🤖 TTS Error: {e_tts}. Would speak: {answer}")
    else:
        try:
            if voice_processing_module.tts_engine: voice_processing_module.speak_response(
                "I didn't catch your question.")
        except Exception as e_tts:
            print(f"🤖 TTS Error: {e_tts}. Would speak: I didn't catch your question.")


def handle_visual_analysis_request():
    global current_context, last_image_caption
    try:
        if voice_processing_module.tts_engine:
            voice_processing_module.speak_response("Do you want to analyze an image from a file or use the webcam?")
        else:
            print("🤖 Do you want to analyze an image from a file or use the webcam? (TTS unavailable)")

        choice_text = "Analyze image from (f)ile or (w)ebcam? "
        user_choice = input(choice_text).lower()

        image_frame = None
        image_path = None

        if 'f' in user_choice:
            image_path_input = input("Enter path to image file: ")
            if os.path.exists(image_path_input):
                image_path = image_path_input
            else:
                msg = f"File not found: {image_path_input}"
                print(msg)
                if voice_processing_module.tts_engine:
                    voice_processing_module.speak_response(msg)
                else:
                    print(f"🤖 Would speak: {msg}")
                return
        elif 'w' in user_choice:
            print("Attempting to use webcam...")
            show_preview = os.environ.get('DISPLAY') is not None  # Basic check
            image_frame = visual_module.capture_from_webcam(show_preview=show_preview)
            if image_frame is None:
                msg = "Failed to capture image from webcam."
                print(msg)
                if voice_processing_module.tts_engine:
                    voice_processing_module.speak_response(msg)
                else:
                    print(f"🤖 Would speak: {msg}")
                return
        else:
            msg = "Invalid choice for image source."
            print(msg)
            if voice_processing_module.tts_engine:
                voice_processing_module.speak_response(msg)
            else:
                print(f"🤖 Would speak: {msg}")
            return

        print("\n--- Basic Image Properties ---")
        properties = None
        if image_path:
            properties = visual_module.get_image_properties(image_path=image_path)
        elif image_frame is not None:
            properties = visual_module.get_image_properties(image_frame=image_frame)

        if isinstance(properties, dict):
            prop_msg = f"Image properties: Height={properties.get('height')}, Width={properties.get('width')}, Channels={properties.get('channels')}."
        else:
            prop_msg = f"Could not get image properties: {properties}"
        print(prop_msg)
        if voice_processing_module.tts_engine:
            voice_processing_module.speak_response(prop_msg)
        else:
            print(f"🤖 Would speak: {prop_msg}")

        print("\n--- Image Captioning ---")
        caption = "Image captioning is unavailable or failed."
        if getattr(visual_module, 'TRANSFORMERS_AVAILABLE', False) and \
                (getattr(visual_module, 'caption_model', None) or visual_module.initialize_caption_model()):
            if image_path:
                caption = visual_module.get_image_caption(image_path=image_path)
            elif image_frame is not None:
                caption = visual_module.get_image_caption(image_frame=image_frame)

        print(f"Caption: {caption}")
        if voice_processing_module.tts_engine:
            voice_processing_module.speak_response(f"The image caption is: {caption}")
        else:
            print(f"🤖 Would speak: The image caption is: {caption}")

        if caption and "unavailable" not in caption.lower() and "error" not in caption.lower():
            current_context = f"The user is looking at an image described as: {caption}"
            last_image_caption = caption
            ctx_msg = f"Context updated based on image caption. You can now ask questions about it."
            print(ctx_msg)
            if voice_processing_module.tts_engine:
                voice_processing_module.speak_response(ctx_msg)
            else:
                print(f"🤖 Would speak: {ctx_msg}")
        else:
            # current_context remains unchanged or set to a generic visual analysis message
            # last_image_caption = None # Already None or retains previous if any
            ctx_msg = "No new textual context from image captioning. Previous context remains."
            print(ctx_msg)
            if voice_processing_module.tts_engine:
                voice_processing_module.speak_response(ctx_msg)
            else:
                print(f"🤖 Would speak: {ctx_msg}")

    except Exception as e_main_visual:
        print(f"An error occurred during visual analysis: {e_main_visual}")
        current_context = "Visual analysis encountered an error."


def start_engagement_monitoring_demo():
    try:
        if voice_processing_module.tts_engine:
            voice_processing_module.speak_response(
                "Starting engagement monitoring demo. This will use the webcam. Press Q in the OpenCV window to stop.")
        else:
            print("🤖 Starting engagement monitoring demo... (TTS unavailable)")
        print("\n--- Engagement Monitoring Demo ---")
        print("This requires a webcam and for the Haar cascade XML to be present.")
        print("Emotion detection also requires the FER library to be functional.")

        show_preview = os.environ.get('DISPLAY') is not None  # Basic check
        if not show_preview:
            print(
                "Note: No display detected, so webcam preview window might not show or might cause errors on some systems.")
        engagement_module.monitor_engagement_from_webcam(show_preview=show_preview)
    except Exception as e_main_engage:
        print(f"An error occurred during engagement monitoring: {e_main_engage}")


def main_loop():
    global current_context
    initialize_all_modules()

    try:
        if voice_processing_module.tts_engine:
            voice_processing_module.speak_response("AI Classroom Assistant activated. How can I help you today?")
        else:
            print("AI Classroom Assistant activated. TTS is unavailable.")
    except Exception as e_tts_init:
        print(f"AI Classroom Assistant activated. TTS initialization error: {e_tts_init}")

    while True:
        print("\nChoose an action:")
        print("1. Ask a question (text input)")
        print("2. Ask a question (voice input)")
        print("3. Analyze an image")
        print("4. Start engagement monitoring demo (webcam)")
        print("5. Set text context manually")
        print("6. Show current context")
        print("7. UI")
        print("0. Exit")

        action_prompt = "Enter your choice (1-7, 0): "
        try:
            if voice_processing_module.stt_recognizer and voice_processing_module.tts_engine:
                voice_processing_module.speak_response(
                    "Say 'text question', 'voice question', 'analyze image', 'monitor engagement', 'set context', 'show context','UI' or 'exit'.")
                action_prompt = "Enter your choice (1-7, 0) or speak a command: "
        except Exception:
            pass

        choice = ""
        try:
            choice = input(action_prompt).strip().lower()
        except EOFError:
            print("EOF detected, exiting.");
            sys.exit(0)

        try:
            if not choice and voice_processing_module.stt_recognizer:
                print("Listening for a menu command...")
                spoken_command = voice_processing_module.listen_for_command(prompt="Say a menu option:",
                                                                            timeout_seconds=3)
                if spoken_command:
                    choice = spoken_command.lower();
                    print(f"Heard command: {choice}")
                else:
                    print("No command heard or understood from voice.");
                    continue
        except Exception:
            if not choice: print(
                "No input received (voice STT might be unavailable). Please type a menu option."); continue

        if choice == '1' or "text question" in choice:
            question = input("Type your question: ")
            handle_text_query(question, use_voice_output=(
                    voice_processing_module.tts_engine is not None) if 'voice_processing_module' in globals() else False)
        elif choice == '2' or "voice question" in choice:
            if 'voice_processing_module' in globals() and voice_processing_module.stt_recognizer and voice_processing_module.tts_engine:
                handle_voice_query()
            else:
                print("Voice input/output is not fully available for this option.")
                if 'voice_processing_module' in globals() and voice_processing_module.tts_engine:
                    voice_processing_module.speak_response("Sorry, voice question mode is not fully available.")
                else:
                    print("🤖 Would speak: Sorry, voice question mode is not fully available.")
        elif choice == '3' or "analyze image" in choice:
            handle_visual_analysis_request()
        elif choice == '4' or "monitor engagement" in choice:
            start_engagement_monitoring_demo()
        elif choice == '5' or "set context" in choice:
            context_file_to_check = "context_to_load.txt"
            loaded_from_file = False
            if os.path.exists(context_file_to_check):
                load_choice_prompt = f"File '{context_file_to_check}' found. Load context from this file? (y/n): "
                load_choice = ""
                try:
                    load_choice = input(load_choice_prompt).lower()
                except EOFError:
                    load_choice = 'n'
                if load_choice == 'y':
                    try:
                        with open(context_file_to_check, 'r', encoding='utf-8') as f:
                            current_context = f.read()
                        print(f"Context updated from file: {context_file_to_check}")
                        try:
                            if voice_processing_module.tts_engine: voice_processing_module.speak_response(
                                "Context updated from file.")
                        except Exception:
                            print("🤖 Would speak: Context updated from file.")
                        loaded_from_file = True
                    except Exception as e:
                        print(f"Error reading context file '{context_file_to_check}': {e}")
                        try:
                            if voice_processing_module.tts_engine: voice_processing_module.speak_response(
                                "Sorry, I couldn't read that context file.")
                        except Exception:
                            print("🤖 Would speak: Sorry, I couldn't read that context file.")
            if not loaded_from_file:
                print("Enter your multi-line context. Type 'ENDCONTEXT' on a new line by itself to finish.")
                lines = []
                while True:
                    try:
                        line = input()
                    except EOFError:
                        print("EOF detected, ending context input.");
                        break
                    if line.strip().upper() == 'ENDCONTEXT': break
                    lines.append(line)
                current_context = "\n".join(lines)
                print(f"Context updated from typed input.")
                if not current_context.strip():
                    current_context = "No specific context has been set yet."
                    print("Context is empty, reset to default.")
                try:
                    if voice_processing_module.tts_engine: voice_processing_module.speak_response(
                        "Context updated from typed input.")
                except Exception:
                    print("🤖 Would speak: Context updated from typed input.")
        elif choice == '6' or "show context" in choice:
            print(f"\nCurrent context:\n--------------------\n{current_context}\n--------------------")
            try:
                if voice_processing_module.tts_engine: voice_processing_module.speak_response(
                    f"The current context is: {current_context}")
            except Exception:
                print(f"🤖 Would speak: The current context is: {current_context[:100]}...")
        elif choice == '7':
            thread = threading.Thread(target=main)
            thread.start()
        elif choice == '0' or "exit" in choice:
            print("Exiting AI Assistant. Goodbye!")
            try:
                if voice_processing_module.tts_engine: voice_processing_module.speak_response("Goodbye!")
            except Exception:
                print("🤖 Would speak: Goodbye!")
            sys.exit(0)
        elif not choice and sys.stdin.isatty():
            print("No input received. Please type a menu option.");
            continue
        elif not choice and not sys.stdin.isatty():
            print("Empty choice from piped input, assuming end of commands.");
            break
        else:
            print("Invalid choice. Please try again.")
            try:
                if voice_processing_module.tts_engine: voice_processing_module.speak_response(
                    "Sorry, I didn't understand that command.")
            except Exception:
                print("🤖 Would speak: Sorry, I didn't understand that command.")


if __name__ == "__main__":
    main_loop()
