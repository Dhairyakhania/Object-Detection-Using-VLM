# OWL-ViT Zero-Shot Object Detection with Gradio

This project demonstrates zero-shot object detection using Google's OWL-ViT (Vision Transformer for Open-World Localization) model. It provides a simple web interface built with Gradio that allows you to detect arbitrary objects in both images and videos based on free-text prompts.

![Gradio Interface Placeholder](placeholder.png)
*(Suggestion: Replace placeholder.png with an actual screenshot of your Gradio app)*

## Features

*   **Zero-Shot Detection:** Detect objects without needing to train the model on them beforehand. Simply provide text descriptions (e.g., "a red car", "two dogs playing").
*   **Image Detection:** Upload an image and specify object prompts to get an output image with bounding boxes and labels.
*   **Video Detection:** Upload a video and specify object prompts. The script processes the video frame by frame and outputs a new video with detections overlaid.
*   **Web UI:** Easy-to-use interface powered by Gradio, with separate tabs for image and video processing.
*   **Model:** Utilizes the `google/owlvit-base-patch32` model from the Hugging Face Hub.

## Requirements

*   Python 3.8+
*   PyTorch
*   Transformers
*   Gradio
*   OpenCV-Python
*   Pillow (PIL Fork)
*   NumPy
*   **FFmpeg (Recommended):** For robust video processing, especially for writing output videos with the H.264 codec (`avc1`) which is widely compatible with web browsers. Ensure OpenCV can access it.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    # Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Create a `requirements.txt` file** with the following content:
    ```txt
    torch
    torchvision
    transformers
    gradio
    opencv-python
    Pillow
    numpy
    ```

4.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **(Optional but Recommended) Install FFmpeg:**
    *   **Linux (Debian/Ubuntu):** `sudo apt update && sudo apt install ffmpeg`
    *   **macOS (using Homebrew):** `brew install ffmpeg`
    *   **Windows:** Download from the [official FFmpeg website](https://ffmpeg.org/download.html) and add it to your system's PATH.
    *   **Conda:** `conda install ffmpeg -c conda-forge` (if using a Conda environment)

## Usage

1.  **Run the Python script:**
    ```bash
    python owl_vit.py
    ```
    (Replace `your_script_name.py` with the actual name of your Python file).

2.  **Open your web browser** and navigate to the URL provided by Gradio (usually `http://127.0.0.1:7860` or similar).

3.  **Use the Interface:**
    *   Select the "Image Detection" or "Video Detection" tab.
    *   Upload your image or video file.
    *   Enter comma-separated text prompts for the objects you want to detect (e.g., `a photo of a cat, a remote control`).
    *   Click the "Submit" button.
    *   The results (image or video with bounding boxes) will be displayed in the output area.

## How it Works

1.  **Model Loading:** The `OwlViTProcessor` and `OwlViTForObjectDetection` model are loaded from the Hugging Face Hub.
2.  **Input Processing:**
    *   **Images:** PIL Images are directly processed.
    *   **Videos:** The script reads the video frame by frame using OpenCV. Each frame is converted to a PIL Image.
3.  **Text Prompts:** The comma-separated text prompts are prepared for the model.
4.  **Inference:** The processor prepares the image(s) and text prompts, and the model performs inference to predict object locations and corresponding labels based on the prompts.
5.  **Post-processing:** The model outputs are processed to get bounding boxes, confidence scores, and predicted labels (indices corresponding to the input prompts). A threshold (e.g., 0.3) is applied to filter low-confidence detections.
6.  **Visualization:** Bounding boxes and labels (with scores) are drawn onto the images/frames using Pillow (PIL).
7.  **Video Output:** For videos, the processed frames are written to a temporary MP4 file using OpenCV's `VideoWriter`. The H.264 (`avc1`) codec is preferred for browser compatibility, with a fallback to `mp4v`.
8.  **Gradio Interface:** Gradio handles the file uploads, text inputs, function calls, and displaying the output images/videos.

## Troubleshooting

*   **Video Output Not Playing:** This is often due to browser codec incompatibility. The script attempts to use the H.264 (`avc1`) codec, which requires `ffmpeg` with `libx264` support to be correctly installed and accessible by OpenCV. If `avc1` fails, it falls back to `mp4v`, which might not play in all browsers. Ensure FFmpeg is installed correctly.
*   **Slow Video Processing:** Object detection on every frame can be computationally intensive, especially without a GPU. Processing may take a long time for longer videos. Consider modifying the code to process only every Nth frame for faster (but less smooth) results.
*   **Font Errors/Missing Fonts:** The script tries to load a default font or "arial.ttf". If you encounter errors related to fonts, ensure the specified font file exists or modify the code in `detect_image` and `detect_video` to use a font available on your system.
*   **CUDA/GPU Issues:** Ensure you have the correct version of PyTorch installed for your CUDA toolkit if you intend to use a GPU.

## License

(Optional: Add a license here, e.g., MIT, Apache 2.0)
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

*   Google Research for the OWL-ViT model.
*   Hugging Face for the Transformers library and model hosting.
*   The Gradio team for the easy-to-use UI library.
*   OpenCV team.