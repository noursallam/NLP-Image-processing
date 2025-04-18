# Install necessary libraries
!pip install transformers torch pillow sentencepiece nltk tqdm
!python -c "import nltk; nltk.download('punkt')"

import torch
from PIL import Image
import numpy as np
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration,
    ViTImageProcessor,
    ViTForImageClassification,
    MarianTokenizer, 
    MarianMTModel
)
from google.colab import files
import logging
from tqdm import tqdm
import time
import gc
from nltk.tokenize import word_tokenize

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def load_models():
    """Load all models needed for the four algorithms"""
    models = {}
    
    try:
        # 1. Load BLIP model (Algorithm 1)
        logger.info("Loading BLIP model...")
        models["blip_processor"] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        models["blip_model"] = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device)
        
        # 2. Load ViT image classifier (for Algorithm 2)
        logger.info("Loading ViT image classifier...")
        models["vit_processor"] = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        models["vit_model"] = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device)
        
        # 3. Load translation model
        logger.info("Loading translation model...")
        model_name = "Helsinki-NLP/opus-mt-en-ar"
        models["translator"] = MarianTokenizer.from_pretrained(model_name)
        models["translation_model"] = MarianMTModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device)
        
        return models
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

def algorithm1_blip_caption(image_path, models):
    """Algorithm 1: Generate caption using BLIP model"""
    try:
        start_time = time.time()
        logger.info("Processing image with Algorithm 1 (BLIP)...")
        
        # Process with smaller image size for faster processing
        image = Image.open(image_path).convert("RGB")
        image = image.resize((384, 384), Image.LANCZOS)
        
        inputs = models["blip_processor"](images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = models["blip_model"].generate(
                **inputs,
                max_length=30,
                num_beams=2
            )
            
        caption = models["blip_processor"].decode(output[0], skip_special_tokens=True)
        logger.info(f"Algorithm 1 completed in {time.time() - start_time:.2f} seconds")
        return caption
    except Exception as e:
        logger.error(f"Error in Algorithm 1: {e}")
        return "Error generating caption with Algorithm 1"

def algorithm2_image_classification(image_path, models):
    """Algorithm 2: Generate caption using image classification"""
    try:
        start_time = time.time()
        logger.info("Processing image with Algorithm 2 (Image Classification)...")
        
        image = Image.open(image_path).convert("RGB")
        inputs = models["vit_processor"](images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = models["vit_model"](**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            
        label = models["vit_model"].config.id2label[predicted_class_idx]
        caption = f"This image appears to show {label}."
        logger.info(f"Algorithm 2 completed in {time.time() - start_time:.2f} seconds")
        return caption
    except Exception as e:
        logger.error(f"Error in Algorithm 2: {e}")
        return "Error generating caption with Algorithm 2"

def algorithm3_template_based(image_path, models):
    """Algorithm 3: Template-based caption using BLIP + classification results"""
    try:
        start_time = time.time()
        logger.info("Processing image with Algorithm 3 (Template-based)...")
        
        # Get classification result
        image = Image.open(image_path).convert("RGB")
        inputs = models["vit_processor"](images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = models["vit_model"](**inputs)
            logits = outputs.logits
            # Get top 3 predictions
            topk_values, topk_indices = torch.topk(logits, k=3, dim=-1)
            
        # Get top 3 labels
        labels = [models["vit_model"].config.id2label[idx.item()] for idx in topk_indices[0]]
        
        # Create template-based caption
        caption = f"A photograph showing {labels[0]}, with possible elements of {labels[1]} and {labels[2]}."
        logger.info(f"Algorithm 3 completed in {time.time() - start_time:.2f} seconds")
        return caption
    except Exception as e:
        logger.error(f"Error in Algorithm 3: {e}")
        return "Error generating caption with Algorithm 3"

def algorithm4_features_based(image_path, models):
    """Algorithm 4: Color and composition analysis with basic image processing"""
    try:
        start_time = time.time()
        logger.info("Processing image with Algorithm 4 (Features-based)...")
        
        image = Image.open(image_path).convert("RGB")
        image = image.resize((128, 128))  # Small size for faster processing
        
        # Basic image analysis
        img_array = np.array(image)
        
        # Color analysis
        avg_color = img_array.mean(axis=(0, 1))
        brightness = avg_color.mean()
        
        # Brightness description
        if brightness < 85:
            brightness_desc = "dark"
        elif brightness < 170:
            brightness_desc = "moderately lit"
        else:
            brightness_desc = "bright"
        
        # Get main caption from BLIP for the content description
        blip_caption = algorithm1_blip_caption(image_path, models)
        words = word_tokenize(blip_caption)
        
        # Create a more analytical caption
        caption = f"A {brightness_desc} image that depicts {' '.join(words[1:])}"
        logger.info(f"Algorithm 4 completed in {time.time() - start_time:.2f} seconds")
        return caption
    except Exception as e:
        logger.error(f"Error in Algorithm 4: {e}")
        return "Error generating caption with Algorithm 4"

def translate_to_arabic(caption, models):
    """Translate caption to Arabic"""
    try:
        start_time = time.time()
        logger.info("Translating caption...")
        
        translation_inputs = models["translator"](caption, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            translated = models["translation_model"].generate(
                **translation_inputs,
                max_length=50,
                num_beams=2
            )
            
        arabic_caption = models["translator"].decode(translated[0], skip_special_tokens=True)
        logger.info(f"Translation completed in {time.time() - start_time:.2f} seconds")
        return arabic_caption
    except Exception as e:
        logger.error(f"Error translating caption: {e}")
        return "خطأ في الترجمة"  # Error in translation

def clear_memory():
    """Clear CUDA memory to prevent OOM errors"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def main():
    try:
        # Upload image file from the user's device
        print("Please upload an image file:")
        uploaded = files.upload()
        
        if not uploaded:
            logger.error("No file was uploaded")
            return
        
        # Get the image path
        image_path = list(uploaded.keys())[0]
        
        # Load models
        print("\nLoading models (optimized for performance)...")
        models = load_models()
        
        # Generate captions using different algorithms
        print("\nGenerating captions with 4 different algorithms...")
        
        captions = {}
        arabic_captions = {}
        
        # Algorithm 1: BLIP
        print("\nRunning Algorithm 1: BLIP Caption Generator...")
        captions["blip"] = algorithm1_blip_caption(image_path, models)
        
        # Algorithm 2: Image Classification
        print("Running Algorithm 2: Image Classification Based Caption...")
        captions["classification"] = algorithm2_image_classification(image_path, models)
        
        # Algorithm 3: Template-based
        print("Running Algorithm 3: Template-Based Caption...")
        captions["template"] = algorithm3_template_based(image_path, models)
        
        # Algorithm 4: Features-based
        print("Running Algorithm 4: Image Features Analysis...")
        captions["features"] = algorithm4_features_based(image_path, models)
        
        # Translate captions to Arabic
        print("\nTranslating all captions to Arabic...")
        for key, caption in captions.items():
            print(f"Translating {key} caption...")
            arabic_captions[key] = translate_to_arabic(caption, models)
        
        # Display results
        print("\n===== RESULTS =====")
        
        print("\n🔹 Algorithm 1: BLIP Caption")
        print("English:", captions["blip"])
        print("Arabic:", arabic_captions["blip"])
        
        print("\n🔹 Algorithm 2: Classification-Based Caption")
        print("English:", captions["classification"])
        print("Arabic:", arabic_captions["classification"])
        
        print("\n🔹 Algorithm 3: Template-Based Caption")
        print("English:", captions["template"])
        print("Arabic:", arabic_captions["template"])
        
        print("\n🔹 Algorithm 4: Features-Based Caption")
        print("English:", captions["features"])
        print("Arabic:", arabic_captions["features"])
        
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")
    finally:
        # Clean up resources
        clear_memory()

if __name__ == "__main__":
    main()
