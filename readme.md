# 🖼️ Advanced Image Caption Generator with Arabic Translation

## EELU PROJECT NLP NOUR'S TEAM

This application generates captions for images in both English and Arabic using state-of-the-art deep learning models.

## Features

- **Automatic Image Captioning**: Generates natural language descriptions of uploaded images
- **Bilingual Output**: Provides captions in both English and Arabic
- **Adjustable Parameters**: Control beam size and maximum caption length for quality vs. speed
- **Caption Storage**: Save and review your generated captions
- **User-friendly Interface**: Simple Gradio-based UI for easy interaction

## Technical Implementation

The system uses the following components:

- **BLIP Model**: Salesforce's BLIP (Bootstrapping Language-Image Pre-training) model for generating English captions
- **Neural Machine Translation**: Helsinki-NLP's MarianMT model for translating captions to Arabic
- **Gradio**: Web interface for easy interaction with the models
- **PyTorch**: Deep learning framework for model inference

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/image-caption-generator.git
cd image-caption-generator

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- Gradio
- Pillow
- CUDA-capable GPU (recommended for faster performance)

## Usage

1. Run the application:
   ```bash
   python app.py
   ```

2. Open the provided URL in your web browser

3. Upload an image using the interface

4. Adjust beam size and maximum caption length parameters if desired

5. Click "Generate Caption" to process the image

6. Review the English and Arabic captions

7. Save captions you like for future reference

## Project Structure

```
.
├── app.py                  # Main application file
├── saved_captions/         # Directory for saved captions and history
│   └── caption_history.json # JSON file storing caption history
├── requirements.txt        # Project dependencies
├── LICENSE                 # MIT License file
└── README.md               # This file
```

## How It Works

1. The application loads pre-trained BLIP and MarianMT models
2. When an image is uploaded, it's processed by the BLIP model to generate an English caption
3. The English caption is then translated to Arabic using the MarianMT model
4. Both captions are displayed to the user and can be saved with a timestamp and unique ID
5. Saved captions are stored in a JSON file for future reference

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Salesforce for the BLIP image captioning model
- Helsinki-NLP for the MarianMT translation model
- Hugging Face for the Transformers library
- Gradio team for the web interface framework

## Contact

EELU PROJECT NLP NOUR'S TEAM - [Your Contact Information]
