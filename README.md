---
title: ViT Music Genre Classification
emoji: 🎶
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

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

## Project Structure

This repository contains all necessary scripts, configuration files, and saved weights to immediately run the genre classification interface.

- **`ViT_music_classification.ipynb`**: This is the core Jupyter Notebook containing the research, dataset parsing, and Model training loop. It converts the audio into Mel-spectrograms (using Librosa) and explicitly bridges audio data to image models.
- **`app.py`**: The main web interface script built with Gradio. It uses this code to serve the frontend interface, quickly processing user-uploaded `.wav` or `.mp3` files into spectrograms, and loading the model to return clear predictions.
- **`requirements.txt`**: A list of all required Python libraries—like `torch`, `transformers`, and `librosa`—needed to run this project smoothly.
- **`vit-mashup-best-local/`**: This directory stores our local fine-tuned model weights and configurations.
  - *Local Model Details*: This contains a Vision Transformer model that was locally fine-tuned directly on our curated music dataset. It was successfully trained for 10 epochs, effectively learning to map the visual patterns of spectrograms to our 10 distinct musical genres.

## How to Run

To run the user interface locally, ensure you have Python installed along with the project's dependencies:

```bash
# Install the required libraries
pip install -r requirements.txt

# Start the web interface
python app.py
```

Once running, you can access the interface through your web browser, upload an audio clip, and watch the model predict the genre.
