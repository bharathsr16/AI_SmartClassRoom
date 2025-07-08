import os
import time
import csv  # For saving feedback

# Try to import Gradio, but expect it might fail if not installed
try:
    import gradio as gr

    GRADIO_AVAILABLE = True
except ImportError:
    print("Gradio library not found. UI cannot be launched.")
    print("Please ensure Gradio is installed (e.g., pip install gradio).")
    GRADIO_AVAILABLE = False

# Import our placeholder modules
import lecture_transcriber
import text_summarizer
import text_summarizer_ov

# --- Feedback Saving ---
FEEDBACK_FILE = "feedback_log.csv"


def save_feedback(audio_filename, summary_text, feedback_text, summarizer_used):
    """Saves the feedback to a CSV file."""
    file_exists = os.path.isfile(FEEDBACK_FILE)
    try:
        with open(FEEDBACK_FILE, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'audio_filename', 'summarizer_used', 'summary_preview', 'feedback']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            writer.writerow({
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'audio_filename': os.path.basename(audio_filename) if audio_filename else "N/A",
                'summarizer_used': summarizer_used,
                'summary_preview': summary_text[:100].replace('\n', ' ') + "..." if summary_text else "N/A",
                # First 100 chars
                'feedback': feedback_text
            })
        return f"Feedback saved to {FEEDBACK_FILE}"
    except Exception as e:
        return f"Error saving feedback: {e}"


# --- Main Processing Function for Gradio ---
def process_lecture_audio(audio_file_obj, use_openvino_summarizer, min_len_summary, max_len_summary):
    """
    Processes the uploaded audio file: transcribes and summarizes.
    audio_file_obj is a Gradio File object.
    """
    if audio_file_obj is None:
        return "Please upload an audio file.", "", "", "", "Error: No audio file."

    audio_filepath = audio_file_obj.name  # .name gives the temporary path of the uploaded file

    # Initialize modules (placeholders will just print messages)
    # These should ideally be initialized once when the app starts,
    # but for simplicity in Gradio's function call, we can call them here.
    # They have internal checks to not re-initialize if already done.
    lecture_transcriber.initialize_model()
    text_summarizer.initialize_model()
    text_summarizer_ov.initialize_model()

    # 1. Transcription (Placeholder)
    transcribe_start_time = time.perf_counter()
    # Simulate some work for STT
    time.sleep(0.1)  # Mock init
    transcript = lecture_transcriber.transcribe_audio(audio_filepath)
    time.sleep(0.2)  # Mock transcribe
    transcribe_duration = time.perf_counter() - transcribe_start_time

    if not transcript or "Error:" in transcript:
        # The placeholder transcriber returns "Error: Audio file not found..." if the path is bad.
        # Gradio provides a valid temp path, so this error from the placeholder
        # implies the placeholder itself decided not to proceed or found an issue.
        # For this demo, we'll assume the placeholder gives some text.
        # If it's a critical error like "file not found", the UI should reflect that.
        # The current placeholder for transcribe_audio returns a mock string even for non-existent files,
        # but it includes "Error:" in that string. Let's refine this.
        # For Gradio, the file WILL exist at audio_file_obj.name.
        # So the placeholder should always return its mock content.
        pass  # Assuming placeholder always gives some text back now

    # 2. Summarization (Placeholder)
    summarizer_module = text_summarizer_ov if use_openvino_summarizer else text_summarizer
    summarizer_name = "OpenVINO (Placeholder)" if use_openvino_summarizer else "HuggingFace (Placeholder)"

    summarize_start_time = time.perf_counter()
    # Simulate some work for summarization
    time.sleep(0.05)  # Mock init
    summary = summarizer_module.summarize_text(transcript, int(min_len_summary), int(max_len_summary))
    time.sleep(0.15)  # Mock summarize
    summarize_duration = time.perf_counter() - summarize_start_time

    # Performance metrics (mock)
    perf_metrics = (
        f"Transcription time (mock): {transcribe_duration:.4f}s\n"
        f"Summarization time ({summarizer_name}, mock): {summarize_duration:.4f}s"
    )

    # Store audio_filepath and summary for feedback submission context
    # Gradio's state management can be used for this, or hidden components.
    # For simplicity, we might pass them back to output components that feedback can read from.

    return transcript, summary, perf_metrics, audio_filepath  # Pass audio_filepath for feedback context


# --- Gradio UI Definition ---
def create_gradio_ui():
    with gr.Blocks(title="Lecture Summarization Assistant") as app:
        gr.Markdown("# AI Lecture Summarization & Feedback Assistant")
        gr.Markdown(
            "**Note:** This demo uses *placeholder* AI models due to environment limitations. "
            "Actual transcription and summarization will not occur. "
            "The UI demonstrates the intended workflow."
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.File(label="Upload Lecture Audio (e.g., .wav, .mp3)", type="filepath")

                with gr.Accordion("Summarization Options", open=False):
                    openvino_checkbox = gr.Checkbox(label="Use OpenVINO Summarizer (Placeholder)", value=False)
                    min_len_slider = gr.Slider(minimum=10, maximum=100, value=30, step=5, label="Min Summary Length")
                    max_len_slider = gr.Slider(minimum=50, maximum=300, value=150, step=10, label="Max Summary Length")

                process_button = gr.Button("Process Lecture", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### Results")
                transcript_output = gr.Textbox(label="Full Transcript (Mock)", lines=10, interactive=False)
                summary_output = gr.Textbox(label="Summary (Mock)", lines=5, interactive=False)
                performance_output = gr.Textbox(label="Performance Metrics (Mock)", lines=3, interactive=False)

                # Hidden component to store the original audio filename for feedback
                audio_filename_for_feedback = gr.Textbox(label="Original Filename", visible=False)

        with gr.Accordion("Feedback on Summary", open=True):
            with gr.Row():
                feedback_text_input = gr.Textbox(label="Your Feedback", lines=3,
                                                 placeholder="What did you think of the summary?")
            submit_feedback_button = gr.Button("Submit Feedback")
            feedback_status_output = gr.Textbox(label="Feedback Status", interactive=False)

        # --- Event Handlers ---
        process_button.click(
            fn=process_lecture_audio,
            inputs=[audio_input, openvino_checkbox, min_len_slider, max_len_slider],
            outputs=[transcript_output, summary_output, performance_output, audio_filename_for_feedback]
        )

        submit_feedback_button.click(
            fn=lambda afp, smry, fbk, use_ov: save_feedback(
                afp,  # audio_filename_for_feedback.value (direct component reference for value)
                smry,  # summary_output.value
                fbk,  # feedback_text_input.value
                "OpenVINO (Placeholder)" if use_ov else "HuggingFace (Placeholder)"
            ),
            inputs=[audio_filename_for_feedback, summary_output, feedback_text_input, openvino_checkbox],
            outputs=[feedback_status_output]
        )

    return app


def main():
    if GRADIO_AVAILABLE:
        print("Launching Gradio UI...")
        print("Note: AI functionalities are placeholders.")

        # Initialize placeholder models once at startup (they have internal checks)
        print("Pre-initializing placeholder modules...")
        lecture_transcriber.initialize_model()
        text_summarizer.initialize_model()
        text_summarizer_ov.initialize_model()
        print("Initialization attempt complete.")

        ui = create_gradio_ui()
        try:
            ui.launch(share=False)  # share=True would create a public link if ngrok is available
            print("Gradio UI launched. Open the URL in your browser.")
        except Exception as e:
            print(f"Error launching Gradio UI: {e}")
            print("This might be due to network issues, port conflicts, or other environment problems.")
            print(
                "If you see errors related to ' कोई वस्तु नहीं मिली ', it might be an issue with Gradio finding its frontend files if installation was incomplete.")

    else:
        print("Gradio is not installed or available. Cannot launch the UI.")
        print("To run the UI, please install Gradio: pip install gradio")


if __name__ == "__main__":
    main()
