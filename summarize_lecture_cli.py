import argparse
import os
import time # For mock timing
import lecture_transcriber # Placeholder STT
import text_summarizer   # Placeholder Summarizer (Hugging Face based)
import text_summarizer_ov # Placeholder Summarizer (OpenVINO based)

def main():
    parser = argparse.ArgumentParser(description="Transcribe and summarize a lecture audio file.")
    parser.add_argument("audio_file_path", help="Path to the lecture audio file (e.g., .wav, .mp3).")
    parser.add_argument("--use_openvino", action="store_true", help="Use OpenVINO version for summarization.")
    # Future arguments could include:
    # parser.add_argument("--output_dir", help="Directory to save transcript and summary files.", default=".")
    parser.add_argument("--min_summary_len", type=int, default=30, help="Minimum length for the summary.")
    parser.add_argument("--max_summary_len", type=int, default=150, help="Maximum length for the summary.")

    args = parser.parse_args()

    audio_path = args.audio_file_path # Original path variable
    use_ov = args.use_openvino
    min_len = args.min_summary_len
    max_len = args.max_summary_len

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at '{audio_path}'")
        return

    transcript = None # Initialize transcript variable

    print("Initializing services...")
    if not lecture_transcriber.initialize_model():
        print("Failed to initialize lecture transcriber. Exiting.")
        return

    summarizer_module = None
    summarizer_name = ""
    if use_ov:
        if not text_summarizer_ov.initialize_model():
            print("Failed to initialize OpenVINO text summarizer. Exiting.")
            return
        summarizer_module = text_summarizer_ov
        summarizer_name = "OpenVINO Summarizer (Placeholder)"
    else:
        if not text_summarizer.initialize_model():
            print("Failed to initialize Hugging Face text summarizer. Exiting.")
            return
        summarizer_module = text_summarizer
        summarizer_name = "Hugging Face Summarizer (Placeholder)"

    print(f"Using: {summarizer_name}")
    print(f"\nProcessing audio file: {original_audio_path}") # Show user the original path they specified
    print("-" * 30)

    # 1. Transcribe audio
    transcribe_start_time = time.perf_counter()
    if hardcoded_transcript_for_testing: # Check if hack is active
        print("Step 1: Using pre-defined dummy transcript for testing CLI summarizer logic.")
        # Call the placeholder transcriber with the dummy path for its side effects (e.g., printing file not found for the dummy path)
        # We will ignore its return value.
        _ = lecture_transcriber.transcribe_audio(audio_path_to_process)
        transcript = hardcoded_transcript_for_testing # Ensure our hardcoded one is used
    else: # Hack is not active, audio file should exist and be processed
        print("Step 1: Transcribing audio (using placeholder)...")
        time.sleep(0.1) # Mock init time
        transcript = lecture_transcriber.transcribe_audio(audio_path_to_process) # Use the correct path variable
        time.sleep(0.2) # Mock transcribe time

    transcribe_end_time = time.perf_counter()
    transcribe_duration = transcribe_end_time - transcribe_start_time

    # Simplified check: if transcript is None or contains "Error:", it's problematic.
    # The placeholder transcriber for a non-existent file returns "Error: Audio file not found..."
    # If using hardcoded transcript, this check should pass unless hardcoded_transcript_for_testing was None.
    if transcript is None or ("Error:" in transcript and transcript != hardcoded_transcript_for_testing):
        print(f"Transcription failed or produced an error: {transcript}")
        return

    print("\n--- Full Transcript (Using Actual or Dummy) ---")
    print(transcript)
    print(f"(Mock transcription time: {transcribe_duration:.4f} seconds)")
    print("-" * 30)

    # 2. Summarize transcript
    summarize_start_time = time.perf_counter()
    print(f"\nStep 2: Summarizing transcript (using {summarizer_name})...")
    # Simulate work for summarizer initialization and processing
    time.sleep(0.05) # Mock init time (already done, but for consistency)
    summary = summarizer_module.summarize_text(transcript, max_length=max_len, min_length=min_len)
    time.sleep(0.15) # Mock summarize time
    summarize_end_time = time.perf_counter()
    summarize_duration = summarize_end_time - summarize_start_time


    if not summary or "Error:" in summary:
        print(f"Summarization failed or produced an error: {summary}")
        return

    print("\n--- Summary (Mock) ---")
    print(summary)
    print("-" * 30)

    print("\nProcessing complete.")

if __name__ == "__main__":
    # To test this CLI script from the command line:
    # 1. Make sure 'sample_lecture_audio.wav' (or any .wav file) exists or is created by lecture_transcriber.py's test.
    #    If lecture_transcriber.py's test block fails to create it (due to missing soundfile/numpy),
    #    you'll need to manually place a WAV file.
    #
    # 2. Run: python summarize_lecture_cli.py sample_lecture_audio.wav
    #
    # Example of how the test within lecture_transcriber.py tries to create a dummy file:
    # (This is just for context, not run here)
    # try:
    #     import soundfile as sf
    #     import numpy as np
    #     _sample_audio_file = "sample_lecture_audio.wav"
    #     if not os.path.exists(_sample_audio_file):
    #         samplerate = 16000; duration = 1; amplitude = 0.0
    #         data = np.full(int(samplerate * duration), amplitude)
    #         sf.write(_sample_audio_file, data, samplerate)
    #         print(f"INFO (CLI): Dummy '{_sample_audio_file}' would be created by lecture_transcriber.py if dependencies met.")
    # except ImportError:
    #     print("INFO (CLI): soundfile/numpy not found; dummy audio creation by lecture_transcriber.py would fail.")
    #     print("INFO (CLI): Ensure 'sample_lecture_audio.wav' exists for testing this CLI.")

    main()
