# Music Genre Classification using Vision Transformers

This project implements a machine learning system that analyzes audio files and classifies them into their primary musical genre. By converting audio into visual representations (spectrograms), the system leverages computer vision techniques to identify audio patterns and predict the correct genre.

## What It Predicts

The primary goal of this application is to take an audio sample—such as a song or a short music clip—and accurately predict its genre. 

The model has been specifically trained to recognize and classify the following 10 genres:
- Blues
- Classical
- Country
- Disco
- Hip-Hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

When an audio clip is processed, the model provides the top three most likely matches along with their probability scores.

## Code Infrastructure

The project is structured into three main components: data processing, model training, and the user interface.

### 1. Data Processing and Training (`ViT_music_classification.ipynb`)
This Jupyter Notebook contains the complete experimentation and training pipeline:
- **Audio Processing:** Uses the `librosa` library to read audio files and convert them into Mel-spectrograms. A Mel-spectrogram is a visual layout of audio frequencies over time.
- **Model Architecture:** Instead of using traditional audio models, this project uses a Vision Transformer (ViT). By treating the spectrograms as images, the Vision Transformer learns visual patterns that correspond to different musical characteristics.
- **Training:** The notebook handles the downloading of the dataset, processing the audio stems, training the custom model, and evaluating its performance.

### 2. The Web Interface (`app.py`)
This script runs the interactive tracking and prediction environment. Built using Gradio, it allows users to easily test the model. 
- It sets up a local web server where users can upload their own audio files.
- It applies the exact same preprocessing steps (converting the uploaded audio into a 5-second Mel-spectrogram image).
- It passes the image to the trained Vision Transformer to receive live predictions.

### 3. Model Storage (`vit-mashup-best-local/`)
This directory holds the final, trained model weights (`model.safetensors`) and the model configurations required to load the model properly without having to retrain from scratch. It is loaded automatically when the web interface starts.

## How to Run

To run the user interface locally, ensure you have Python installed along with the project's dependencies:

```bash
# Install the required libraries
pip install -r requirements.txt

# Start the web interface
python app.py
```

Once running, you can access the interface through your web browser, upload an audio clip, and watch the model predict the genre.
