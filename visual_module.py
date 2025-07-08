import cv2
from PIL import Image # Pillow is often better for transformers image processing
# We will attempt to use transformers, but it might fail due to environment issues
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Transformers library not found. Visual processing will be limited.")
    TRANSFORMERS_AVAILABLE = False
except Exception as e:
    print(f"Error importing transformers: {e}. Visual processing will be limited.")
    TRANSFORMERS_AVAILABLE = False


# --- Image Captioning Setup (if transformers are available) ---
caption_processor = None
caption_model = None
CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-large"

def initialize_caption_model():
    """Initializes the image captioning model and processor."""
    global caption_processor, caption_model
    if not TRANSFORMERS_AVAILABLE:
        print("Cannot initialize caption model: Transformers library is not available.")
        return False

    if caption_processor is None or caption_model is None:
        print(f"Loading image captioning model '{CAPTION_MODEL_NAME}'...")
        try:
            caption_processor = BlipProcessor.from_pretrained(CAPTION_MODEL_NAME)
            caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_NAME)
            # If a GPU is available and torch is correctly set up with CUDA:
            # if torch.cuda.is_available():
            #     caption_model.to("cuda")
            print("Image captioning model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading image captioning model: {e}")
            caption_processor = None
            caption_model = None
            return False
    return True

def get_image_caption(image_path: str = None, image_frame = None) -> str | None:
    """
    Generates a caption for the given image.
    Provide either image_path or image_frame (OpenCV BGR ndarray).
    """
    if not TRANSFORMERS_AVAILABLE or caption_model is None or caption_processor is None:
        if not initialize_caption_model(): # Attempt to initialize
             return "Image captioning service not available."

    raw_image = None
    try:
        if image_path:
            raw_image = Image.open(image_path).convert('RGB')
        elif image_frame is not None:
            raw_image = Image.fromarray(cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB))
        else:
            return "No image provided for captioning."
    except FileNotFoundError:
        return f"Error: Image file not found at {image_path}."
    except Exception as e:
        return f"Error loading image: {e}"

    if raw_image:
        try:
            # Conditional captioning (optional text prompt)
            # text = "a photography of"
            # inputs = caption_processor(raw_image, text, return_tensors="pt")

            # Unconditional captioning
            inputs = caption_processor(raw_image, return_tensors="pt")

            # If using GPU:
            # if torch.cuda.is_available():
            #     inputs = inputs.to("cuda")

            outputs = caption_model.generate(**inputs, max_length=150) # Increased max_length
            caption = caption_processor.decode(outputs[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            return f"Error generating caption: {e}"
    return "Could not generate caption."

# --- Basic Image Properties (Fallback/Alternative) ---
def get_image_properties(image_path: str = None, image_frame = None) -> dict | str:
    """
    Returns basic properties of the image (dimensions, channels).
    Provide either image_path or image_frame (OpenCV BGR ndarray).
    """
    img = None
    try:
        if image_path:
            img = cv2.imread(image_path)
            if img is None:
                return f"Error: Could not read image from {image_path}."
        elif image_frame is not None:
            img = image_frame
        else:
            return "No image provided for properties."
    except Exception as e:
        return f"Error loading image with OpenCV: {e}"

    if img is not None:
        height, width = img.shape[:2]
        channels = img.shape[2] if len(img.shape) == 3 else 1
        return {"height": height, "width": width, "channels": channels}
    return "Could not retrieve image properties."


# --- Webcam Capture ---
def capture_from_webcam(show_preview=True):
    """
    Captures a single frame from the webcam.
    Returns the frame as an OpenCV ndarray (BGR).
    Returns None if capture fails.
    """
    cap = cv2.VideoCapture(0) # 0 is usually the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    print("Press 'c' to capture frame, 'q' to quit without capture.")
    frame = None
    while True:
        ret, current_frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from webcam.")
            cap.release()
            if show_preview: cv2.destroyAllWindows()
            return None

        if show_preview:
            display_frame = current_frame.copy()
            cv2.putText(display_frame, "Press 'c' to capture, 'q' to quit", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("Webcam - Press 'c' to capture, 'q' to quit", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            frame = current_frame
            print("Frame captured.")
            break
        elif key == ord('q'):
            print("Quitting webcam capture without saving.")
            break

    cap.release()
    if show_preview: cv2.destroyAllWindows()
    return frame

if __name__ == "__main__":
    print("--- Visual Module Test ---")

    # Test with a placeholder image file if one exists or can be created.
    # For now, we'll rely on webcam or skip if no image.
    # Create a dummy image for testing if OpenCV is available
    dummy_image_path = "dummy_test_image.png"
    try:
        import numpy as np
        dummy_img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(dummy_img_array, "TEST", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imwrite(dummy_image_path, dummy_img_array)
        print(f"Created dummy image: {dummy_image_path}")
        IMAGE_AVAILABLE = True
    except Exception as e:
        print(f"Could not create dummy image using OpenCV: {e}")
        print("Skipping file-based image tests if dummy image creation failed.")
        IMAGE_AVAILABLE = False


    if IMAGE_AVAILABLE:
        print("\n--- Test 1: Image Properties from File ---")
        properties = get_image_properties(image_path=dummy_image_path)
        print(f"Properties of '{dummy_image_path}': {properties}")

        if TRANSFORMERS_AVAILABLE:
            print("\n--- Test 2: Image Captioning from File ---")
            # Initialize model first (important if not done globally)
            if initialize_caption_model():
                caption = get_image_caption(image_path=dummy_image_path)
                print(f"Caption for '{dummy_image_path}': {caption}")
            else:
                print("Skipping captioning test as model initialization failed.")
        else:
            print("\nSkipping Test 2 (Image Captioning) as Transformers library is not available/functional.")

    # Test with webcam (this might fail in many sandboxed environments)
    print("\n--- Test 3: Webcam Capture and Info ---")
    print("Attempting to access webcam. This may not work in all environments.")
    print("If a window appears, press 'c' to capture an image, or 'q' to quit.")

    captured_frame = None
    try:
        # Check if display is available for cv2.imshow
        # This is a common issue in headless environments
        # Forcing show_preview=False if no display (heuristic)
        import os
        show_preview_webcam = os.environ.get('DISPLAY') is not None
        if not show_preview_webcam:
            print("No display environment detected, webcam preview will be disabled.")

        captured_frame = capture_from_webcam(show_preview=show_preview_webcam)
    except Exception as e:
        print(f"Webcam test failed: {e}")

    if captured_frame is not None:
        print("\n--- Test 3a: Image Properties from Webcam Frame ---")
        properties_webcam = get_image_properties(image_frame=captured_frame)
        print(f"Properties of captured frame: {properties_webcam}")

        if TRANSFORMERS_AVAILABLE:
            print("\n--- Test 3b: Image Captioning from Webcam Frame ---")
            if caption_model or initialize_caption_model(): # Ensure model is loaded
                caption_webcam = get_image_caption(image_frame=captured_frame)
                print(f"Caption for captured frame: {caption_webcam}")
            else:
                print("Skipping webcam captioning as model is not available.")
        else:
            print("\nSkipping Test 3b (Image Captioning from Webcam) as Transformers library is not available/functional.")

        # Optionally save the captured frame
        try:
            cv2.imwrite("webcam_capture.png", captured_frame)
            print("Webcam frame saved to webcam_capture.png")
        except Exception as e:
            print(f"Could not save webcam_capture.png: {e}")

    else:
        print("No frame captured from webcam, or webcam not accessible.")

    print("\n--- Visual Module Test Complete ---")
