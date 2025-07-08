# AI-Powered Interactive Learning Assistant for Classrooms

**Objective**: Build a Multimodal AI assistant for classrooms to dynamically answer queries using text, voice, and visuals while improving student engagement with personalized responses.

**Current Status (IMPORTANT):**
This project has been developed in a sandboxed environment where critical dependencies could not be installed due to limitations (e.g., disk space, OS-level package installation). As a result, **most of the core AI functionalities are currently NOT OPERATIONAL when running the `main_assistant.py` script or individual modules directly from this sandbox.**

The code represents the intended architecture and logic for such an assistant. To run this project successfully, you will need to set up a local Python environment that meets all the prerequisites listed below.

## Project Structure

The project is organized into several Python modules:

*   **`main_assistant.py`**: The central orchestrator for the assistant. It integrates all other modules and manages the primary user interaction loop. (Currently fails to launch due to `torch` import errors from `text_qa_module`).
*   **`text_qa_module.py`**: Handles text-based Question Answering using Hugging Face Transformers. (Requires `torch`, `transformers`).
*   **`voice_processing_module.py`**: Manages Speech-to-Text (STT) using `speech_recognition` and Text-to-Speech (TTS) using `pyttsx3`. (Requires `PyAudio` for microphone STT, and an OS-level TTS engine like `espeak-ng` for `pyttsx3` on Linux).
*   **`visual_module.py`**: Processes visual input. Includes basic image property extraction (OpenCV) and is designed for image captioning (Hugging Face Transformers). (Captioning requires `torch`, `transformers`. Webcam access depends on OS and OpenCV setup).
*   **`engagement_module.py`**: Aims to perform facial expression analysis for engagement monitoring. Uses OpenCV for face detection and is designed to use the `fer` library for emotion recognition. (Face detection requires `haarcascade_frontalface_default.xml`. Emotion recognition requires `fer` and `tensorflow`).
*   **`visual_aid_generator.py`**: Designed to generate simple visual aids (e.g., bar charts) using Matplotlib. (Requires `matplotlib`).
*   **`requirements.txt`**: Lists the Python dependencies.
*   **`convert_bert.py`**: A utility script (from the original codebase) to convert a Hugging Face BERT model to OpenVINO format. The main application modules currently use Hugging Face models directly, but this shows an optimization path.
*   **`bert_ir/`**: Directory containing OpenVINO models (as per original codebase structure) and some initial example scripts (`visual input.py`, `with voice.py`).

## Prerequisites

To run this project in your own environment, you will need:

1.  **Python 3.9+**
2.  **Pip** (Python package installer)
3.  **Tesseract OCR Engine**:
    *   Required by `pytesseract` (used in the original `bert_ir/visual input.py` and potentially useful for advanced visual queries).
    *   Installation varies by OS (e.g., `sudo apt install tesseract-ocr` on Debian/Ubuntu).
    *   Ensure `tesseract` is in your system's PATH or configure the path in code if needed.
4.  **PortAudio**:
    *   Required by `PyAudio` (a dependency for `speech_recognition` to use the microphone).
    *   Installation varies by OS (e.g., `sudo apt install portaudio19-dev` on Debian/Ubuntu).
5.  **eSpeak NG (or another TTS engine for `pyttsx3` on Linux)**:
    *   `pyttsx3` often relies on `espeak-ng` on Linux.
    *   Installation: `sudo apt install espeak-ng` on Debian/Ubuntu.
6.  **Haar Cascade XML File**:
    *   The file `haarcascade_frontalface_default.xml` is required for basic face detection in `engagement_module.py`.
    *   Download it from [OpenCV's GitHub repository](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml) and place it in the root directory of the project.
7.  **Python Libraries**: Install using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure you have enough disk space and that any OS-level dependencies for these Python packages (especially for `torch`, `tensorflow` (via `fer`), `opencv-python`, `PyAudio`) are met prior to running pip install.*

## How to (Attempt to) Run

1.  **Set up your environment** by installing all prerequisites listed above.
2.  **Place `haarcascade_frontalface_default.xml`** in the project root.
3.  **Run the main assistant**:
    ```bash
    python main_assistant.py
    ```
4.  Alternatively, you can try running individual modules (e.g., `python text_qa_module.py`) to test them, but they also depend on the prerequisites.

## Challenges & Limitations (as observed in the development sandbox)

*   **Dependency Installation**: The primary blocker was the inability to install large libraries like `torch`, `tensorflow`, and `matplotlib` due to sandbox limitations. This rendered most AI features non-functional.
*   **Hardware Access**: Webcam, microphone, and speaker access can be problematic in restricted environments and require correct OS-level drivers and permissions.
*   **Model Availability**: Pre-trained models from Hugging Face are downloaded at runtime if not cached. This requires an internet connection and sufficient disk space.
*   **Real-time Performance**: True real-time interaction with multiple AI models (QA, STT, TTS, Vision, FER) is computationally intensive and requires optimization (like using OpenVINO, as hinted in `convert_bert.py`) and potentially powerful hardware (GPU/NPU).

This project structure provides a foundation. Getting it fully operational requires a robust local development environment where all dependencies can be correctly installed and hardware can be accessed.Tool output for `create_file_with_block`:
