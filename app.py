import os
import torch
import numpy as np
import librosa
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
import gradio as gr

MODEL_PATH = "vit-mashup-best-local"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SR = 16_000
CHUNK_SECONDS = 5
SAMPLES = SR * CHUNK_SECONDS

print(f"Loading model on {DEVICE}...")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTForImageClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()
print("Model loaded successfully.")

def to_mel(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        fmin=20,
        fmax=SR // 2
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    return mel.astype(np.float32)

def mel_to_image(mel):
    mel_norm = mel - mel.min()
    mel_max = mel_norm.max()
    if mel_max > 0:
        mel_norm = (mel_norm / mel_max * 255).astype(np.uint8)
    else:
        mel_norm = np.zeros_like(mel, dtype=np.uint8)
    return Image.fromarray(mel_norm).resize((224, 224), Image.BILINEAR).convert("RGB")

def predict_genre(audio_path):
    if audio_path is None:
        return {}
        
    audio, _ = librosa.load(audio_path, sr=SR, duration=CHUNK_SECONDS)
    if len(audio) < SAMPLES: 
        audio = np.pad(audio, (0, SAMPLES - len(audio)))

    img = mel_to_image(to_mel(audio))
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    
    labels_dict = {}
    for i in range(len(probs)):
        label = model.config.id2label.get(i, f"LABEL_{i}")
        # ID2Label json keys are sometimes parsed as strings, so let's check str(i) too
        if isinstance(model.config.id2label, dict):
            if i in model.config.id2label:
                label = model.config.id2label[i]
            elif str(i) in model.config.id2label:
                label = model.config.id2label[str(i)]
        labels_dict[label] = float(probs[i])
        
    return labels_dict

# Gradio Interface
iface = gr.Interface(
    fn=predict_genre,
    inputs=gr.Audio(type="filepath", label="Upload Music File"),
    outputs=gr.Label(num_top_classes=3, label="Predicted Genre"),
    title="ViT Music Genre Classification",
    description="Upload a song or audio clip. The Vision Transformer (ViT) model will convert it to a Mel-spectrogram and predict the top 3 most likely genres."
)

if __name__ == "__main__":
    iface.launch(share=False)
