import os
import pickle
import numpy as np
import torch
import librosa
import re
import matplotlib.pyplot as plt
from CLAP import CLAP

# Load the CAV, scaler, and classifier
with open("pkl files/footstep_speed_rcv.pkl", "rb") as f:
    cav_data = pickle.load(f)['rcv']
with open("pkl files/footstep_speed_rcv.pkl", "rb") as f:
    scaler = pickle.load(f)['scaler']

# Initialize CLAP model
clap_model = CLAP()

# Directory with audio files
audio_dir = "ffxFootstepsGenData"
audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.aac'))]

def extract_speed(filename):
    match = re.search(r"spe_(\d+\.?\d*)", filename)
    return float(match.group(1)) if match else None

speeds = []
scores = []
file_labels = []

for audio_file in audio_files:
    speed = extract_speed(audio_file)
    if speed is None:
        continue
    try:
        # Load and embed audio
        audio_data, sr = librosa.load(audio_file, sr=44100)
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
        embedding = clap_model.get_audio_embedding(audio_tensor, sr)
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
        embedding = embedding.flatten()
        # Standardize
        embedding_scaled = scaler.transform(embedding.reshape(1, -1))[0]
        # Compute attribute score
        score = np.dot(embedding_scaled, cav_data)
        speeds.append(speed)
        scores.append(score)
        file_labels.append(os.path.basename(audio_file))
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(speeds, scores, alpha=0.7)
plt.xlabel("Speed (from filename, spe_{value})")
plt.ylabel("Attribute Score (CAV alignment)")
plt.title("CAV Attribute Score vs. Footstep Speed")
plt.grid(True)

# Optionally, fit and plot a trend line
if len(speeds) > 1:
    z = np.polyfit(speeds, scores, 1)
    p = np.poly1d(z)
    plt.plot(sorted(speeds), p(sorted(speeds)), "r--", label="Trend line")
    plt.legend()

plt.tight_layout()
plt.savefig("cav_score_vs_speed_more data.png", dpi=150)
plt.show()