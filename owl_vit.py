import gradio as gr
import torch
import torchvision.transforms as T
import cv2
import numpy as np
import tempfile
from PIL import Image, ImageDraw, ImageFont
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import os 

# Loading the model and processor 

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Function for image detection

def detect_image(img, text_prompt):
    prompts = [p.strip() for p in text_prompt.split(",")]

    inputs = processor(text=prompts, images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.Tensor([img.size[::-1]])  # H, W
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)[0]

    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 15) # Example using Arial
    except IOError:
        print("Default font not found, using load_default().")
        font = ImageFont.load_default()


    for box, label_idx, score in zip(results["boxes"], results["labels"], results["scores"]):
        label = prompts[label_idx]
        box = [int(b) for b in box.tolist()]
        draw.rectangle(box, outline="lime", width=3)
        
        text_y = box[1] - 15 if box[1] - 15 > 0 else box[1] + 5
        draw.text((box[0], text_y), f"{label} ({score:.2f})", fill="lime", font=font)

    return img


# Function for video detection
def detect_video(video_path, text_prompt):
    if not video_path:
        print("No video path provided.")
        return None

    prompts = [p.strip() for p in text_prompt.split(",")]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None 

    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fps = 30 if fps <= 0 else fps
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output_path = output_file.name
    output_file.close() 

    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    fourcc_h264 = cv2.VideoWriter_fourcc(*'avc1') 

    print(f"Attempting to use codec: {'avc1'}")
    out = cv2.VideoWriter(output_path, fourcc_h264, fps, (width, height))

    if not out.isOpened():
        print(f"Warning: Could not open video writer with codec {'avc1'}. Trying 'mp4v' fallback...")
        fourcc_mp4v = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc_mp4v, fps, (width, height))
        if not out.isOpened():
             print(f"Error: Could not open video writer with fallback codec {'mp4v'}.")
             cap.release()
             try:
                 os.remove(output_path)
             except OSError as e:
                 print(f"Error removing temp file {output_path}: {e}")
             return None 

    print(f"VideoWriter opened successfully with codec {'avc1' if fourcc_h264 == out.getBackendName() else 'mp4v'}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Convert frame to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Process the frame
        inputs = processor(text=prompts, images=pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.Tensor([pil_image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)[0]

        # Draw bounding boxes
        draw = ImageDraw.Draw(pil_image)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        for box, label_idx, score in zip(results["boxes"], results["labels"], results["scores"]):
            
            if label_idx < len(prompts):
                label = prompts[label_idx]
                box = [int(b) for b in box.tolist()]
                draw.rectangle(box, outline="lime", width=3)
                text_y = box[1] - 15 if box[1] - 15 > 0 else box[1] + 5
                draw.text((box[0], text_y), f"{label} ({score:.2f})", fill="lime", font=font)
            else:
                print(f"Warning: label_idx {label_idx} out of bounds for prompts list (len={len(prompts)})")


        # Converting back to OpenCV format and write to output video
        processed_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        out.write(processed_frame)

    print(f"Processed {frame_count} frames.")
    cap.release()
    out.release()
    print(f"Video saved to: {output_path}")
    return output_path



# Gradio interface

image_tab = gr.Interface(
    fn=detect_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(lines=1, placeholder="e.g. a tiger, a monitor", label="Object Prompts (comma-separated)")
    ],
    outputs=gr.Image(type="pil", label="Detected Image"),
)

video_tab = gr.Interface(
    fn=detect_video,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Textbox(lines=1, placeholder="e.g. a tiger, a chair", label="Object Prompts (comma-separated)")
    ],
    outputs=gr.Video(label="Detected Video"), # Expects a file path
)

gr.TabbedInterface(
    interface_list=[image_tab, video_tab],
    tab_names=["Image Detection", "Video Detection"],
    title="Zero-Shot Object Detection with OWL-ViT"
).launch(debug=True) 
