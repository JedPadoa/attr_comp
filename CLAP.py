import numpy as np
import librosa
import torch
import laion_clap
import os
import torchaudio
import torch.nn.functional as F
import librosa as li

def dummy_load(name):
    """
    Preprocess function that takes one audio path, crops it to 6 seconds,
    and returns it as 3 chunks of 2 seconds each (88200 samples per chunk at 44100 Hz).
    """
    target_length = 6 * 44100
    # Load audio at 44.1kHz
    x = li.load(name, sr=44100)[0]
    #x = load_audio_mono(name)
    
    # Calculate padding needed to make length divisible by 16
    remainder = target_length % 16
    if remainder != 0:
        padding = 16 - remainder
        target_length += padding
    
    # Crop or pad to exactly target length
    if len(x) > target_length:
        x = x[:target_length]
    elif len(x) < target_length:
        x = np.pad(x, (0, target_length - len(x)))
    
    # Reshape into chunks
    #x = x.reshape(int(config.AUDIO_LENGTH/2), -1)  # -1 will automatically calculate the correct chunk size
    
    if x.shape[0]:
        return x
    else:
        return None

class CLAP:
    def __init__(self, audio_dir = None, texts = None):
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt()
        self.audio_dir = audio_dir
        self.texts = texts
        
        if texts is not None:
            self.text_embeddings = self.get_text_embeddings(texts)
        
            diffs = []

            for i in range(len(self.text_embeddings) - 1):
                diff = self.subtract_embeddings(self.text_embeddings[i], self.text_embeddings[-1])
                diffs.append(diff)
            
            self.attr_vector = F.normalize(torch.mean(torch.stack(diffs), dim=0), p=2, dim=0)
        #print(f'attr_vector shape: {self.attr_vector.shape}')

    def get_audio_embedding(self, audio_data, sr):
        #audio_data, sr = crop_audio(audio_file)
        if sr != 48000:
            resampler = torchaudio.transforms.Resample(sr, 48000)
            audio_data = resampler(audio_data)
        audio_data = audio_data.reshape(1, -1)
        audio_embed = self.model.get_audio_embedding_from_data(x = audio_data, use_tensor=True)
        return audio_embed
    
    def get_audio_embeddings(self):
        assert self.audio_dir is not None, "audio_dir must be provided"
        audio_files = [
            f'{self.audio_dir}/{f}' for f in os.listdir(self.audio_dir) if f.endswith('.wav')
        ]
        audio_embeddings = self.model.get_audio_embedding_from_filelist(x = audio_files, use_tensor=True)
        return audio_embeddings
    
    def get_text_embedding(self, text):
        text_embed = self.model.get_text_embedding(text, use_tensor=False)
        return text_embed
    
    def get_text_embeddings(self, texts = None):
        if texts is None:
            text_embeddings = self.model.get_text_embedding(self.texts, use_tensor=True)
        else:
            text_embeddings = self.model.get_text_embedding(texts, use_tensor=True)
        return text_embeddings
    
    def get_text_from_audio(self, audio_file, texts):
        with torch.no_grad():
            audio_embed = self.get_audio_embedding(audio_file)
            text_embeddings = self.get_text_embedding(texts)
            #sorts indices of text embeddings by similarity to audio embedding
            ranking = torch.argsort((torch.tensor(audio_embed) @ torch.tensor(text_embeddings).t()).squeeze(), descending=True)
            ranking = ranking.tolist()
            best_match = texts[ranking[0]]
            
        return best_match, ranking
    
    def subtract_embeddings(self, E_1, E_2):
        return torch.subtract(E_1, E_2)
    
    def get_attribute_score(self, audio, descriptors, resampled_length):
        with torch.no_grad():  # Don't track gradients
            audio_embed = torch.squeeze(self.get_audio_embedding(audio, sr=44100))
            score = torch.dot(audio_embed, self.attr_vector)
            features = torch.full((resampled_length,), score.item()).numpy()
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return features
        
    
    
if __name__ == "__main__":
    clap = CLAP(audio_dir='audio', texts=None)
    audio = torch.tensor(dummy_load('all5hrs_footsteps/1-ur_15.wav'))
    score = clap.get_attribute_score(audio)
    print(f'score: {score}')
    