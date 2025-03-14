import os
import gradio as gr
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer
import torch
import uuid
import json
from datetime import datetime

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models and processors
print("Loading caption model...")
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

print("Loading translation model...")
translator_model_name = "Helsinki-NLP/opus-mt-en-ar"
translator_tokenizer = MarianTokenizer.from_pretrained(translator_model_name)
translator_model = MarianMTModel.from_pretrained(translator_model_name).to(device)

# Create directory for saved captions
SAVE_DIR = "saved_captions"
os.makedirs(SAVE_DIR, exist_ok=True)

# JSON file to store all captions
HISTORY_FILE = os.path.join(SAVE_DIR, "caption_history.json")

# Initialize history file if it doesn't exist
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

def load_history():
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def save_to_history(entry):
    history = load_history()
    history.append(entry)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def generate_caption(image, num_beams=5, max_length=30):
    """Generate a caption for the image with beam search for better quality"""
    if image is None:
        return "", ""
    
    # Process image
    inputs = caption_processor(images=image, return_tensors="pt").to(device)
    
    # Generate caption with beam search
    out = caption_model.generate(
        **inputs,
        num_beams=num_beams,
        max_length=max_length,
        early_stopping=True
    )
    
    # Decode caption
    caption = caption_processor.batch_decode(out, skip_special_tokens=True)[0]
    
    # Translate to Arabic
    inputs = translator_tokenizer(caption, return_tensors="pt", padding=True, truncation=True).to(device)
    translated = translator_model.generate(**inputs, num_beams=4, max_length=50)
    arabic_caption = translator_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    
    return caption, arabic_caption

def save_caption(image, english_caption, arabic_caption):
    """Save the image and its captions"""
    if image is None or not english_caption:
        return "No image or caption to save."
    
    # Create a unique ID for this caption
    caption_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the image
    image_filename = f"{timestamp}_{caption_id}.jpg"
    image_path = os.path.join(SAVE_DIR, image_filename)
    image.save(image_path)
    
    # Save entry to history
    entry = {
        "id": caption_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_path": image_path,
        "english_caption": english_caption,
        "arabic_caption": arabic_caption
    }
    save_to_history(entry)
    
    return f"Caption saved successfully! ID: {caption_id}"

def process_image(image, num_beams, max_length):
    """Process the image and generate captions"""
    if image is None:
        return None, "", "", ""
    
    # Generate captions
    english_caption, arabic_caption = generate_caption(image, num_beams, max_length)
    
    # Return results
    return image, english_caption, arabic_caption, ""

def view_history():
    """Load and display caption history"""
    history = load_history()
    if not history:
        return "No captions saved yet."
    
    # Format history for display
    formatted_history = ""
    for entry in history:
        formatted_history += f"ID: {entry['id']} - {entry['timestamp']}\n"
        formatted_history += f"English: {entry['english_caption']}\n"
        formatted_history += f"Arabic: {entry['arabic_caption']}\n"
        formatted_history += "-" * 40 + "\n"
    
    return formatted_history

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# EELU PROJECT NLP NOUR'S TEAM")
    gr.Markdown("Upload an image to generate captions in English and Arabic")
    
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Tab("Generate Captions"):
                # Input components
                with gr.Row():
                    image_input = gr.Image(type="pil", label="Upload Image")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        num_beams = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Beam Size (Higher = Better Quality)")
                        max_length = gr.Slider(minimum=10, maximum=50, value=30, step=1, label="Max Caption Length")
                    with gr.Column(scale=1):
                        generate_btn = gr.Button("Generate Caption", variant="primary")
                
                # Output components
                with gr.Row():
                    with gr.Column():
                        english_output = gr.Textbox(label="English Caption", lines=2)
                    with gr.Column():
                        arabic_output = gr.Textbox(label="Arabic Caption", lines=2)
                
                with gr.Row():
                    save_btn = gr.Button("💾 Save Caption", variant="secondary")
                    save_status = gr.Textbox(label="Save Status")
            
            with gr.Tab("Saved Captions"):
                history_btn = gr.Button("View Saved Captions")
                history_output = gr.Textbox(label="Caption History", lines=20)
        
        with gr.Column(scale=1):
            gr.Markdown("### How to use:")
            gr.Markdown("""
            1. Upload an image using the uploader
            2. Adjust beam size and max length if needed
            3. Click "Generate Caption"
            4. Save the caption if you like it
            5. View your saved captions in the history tab
            
            ### Tips:
            - Higher beam size gives better quality but slower generation
            - Adjust max length based on image complexity
            - All captions are saved with unique IDs for reference
            """)
    
    # Set up event handlers
    generate_btn.click(
        fn=process_image,
        inputs=[image_input, num_beams, max_length],
        outputs=[image_input, english_output, arabic_output, save_status]
    )
    
    save_btn.click(
        fn=save_caption,
        inputs=[image_input, english_output, arabic_output],
        outputs=[save_status]
    )
    
    history_btn.click(
        fn=view_history,
        inputs=[],
        outputs=[history_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)
