import os
import shutil
import re
import random
import numpy as np
import lmdb
import pickle
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import librosa
import torch
from CLAP import CLAP
from fs_lmdb import LMDBFootstepDataset

def find_footstep_files(base_dir: str = ".") -> List[str]:
    """Find all audio files with attribute labels in their names."""
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac']
    footstep_files = []
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if (any(file.lower().endswith(ext) for ext in audio_extensions) and 
                'spe_' in file and 'con_' in file and 'woo_' in file and 'gra_' in file):
                footstep_files.append(os.path.join(root, file))
    
    return footstep_files

def extract_attributes_from_filename(filename: str) -> Dict[str, float]:
    """
    Extract attribute values from filename.
    Expected format: steps_spe_0.20_con_0.00_woo_0.50_gra_0.00.wav
    
    Args:
        filename: Name of the audio file
        
    Returns:
        Dictionary with attribute values or None if parsing fails
    """
    attributes = {}
    
    # Define attribute patterns
    patterns = {
        'speed': r'spe_(\d+\.?\d*)',
        'concrete': r'con_(\d+\.?\d*)',
        'wood': r'woo_(\d+\.?\d*)',
        'grass': r'gra_(\d+\.?\d*)'
    }
    
    try:
        for attr_name, pattern in patterns.items():
            match = re.search(pattern, filename)
            if match:
                attributes[attr_name] = float(match.group(1))
            else:
                print(f"Warning: Could not find {attr_name} in {filename}")
                return None
        
        return attributes
        
    except (ValueError, AttributeError) as e:
        print(f"Error parsing attributes from {filename}: {e}")
        return None

def group_files_by_attributes(files: List[str]) -> Dict[tuple, List[str]]:
    """Group files by their attribute combinations."""
    attribute_groups = {}
    
    for file_path in files:
        filename = os.path.basename(file_path)
        attributes = extract_attributes_from_filename(filename)
        
        if attributes is not None:
            # Create a tuple key from all attribute values
            attr_key = (attributes['speed'], attributes['concrete'], 
                       attributes['wood'], attributes['grass'])
            
            if attr_key not in attribute_groups:
                attribute_groups[attr_key] = []
            attribute_groups[attr_key].append(file_path)
    
    return attribute_groups

def create_full_lmdb_dataset(audio_dir: str = ".", 
                            db_path: str = "footsteps_full",
                            clap_model: Optional[CLAP] = None) -> str:
    """
    Create LMDB dataset with CLAP embeddings for all available attribute combinations.
    
    Args:
        audio_dir: Directory to search for audio files
        db_path: Path to LMDB database
        clap_model: Pre-initialized CLAP model
        
    Returns:
        Path to created LMDB database
    """
    # Initialize CLAP model if not provided
    if clap_model is None:
        print("Initializing CLAP model...")
        clap_model = CLAP()
    
    # Find all footstep files
    print(f"Searching for footstep files in: {audio_dir}")
    all_files = find_footstep_files(audio_dir)
    
    if not all_files:
        print("No footstep files with attribute labels found!")
        return None
    
    print(f"Found {len(all_files)} footstep files")
    
    # Group files by attribute combinations
    attribute_groups = group_files_by_attributes(all_files)
    print(f"Found {len(attribute_groups)} unique attribute combinations")
    
    # Count total files across all combinations
    total_files = sum(len(files) for files in attribute_groups.values())
    print(f"Creating LMDB dataset with {total_files} samples")
    
    # Remove existing database
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    
    print(f"Creating LMDB dataset at: {db_path}")
    
    # Create LMDB dataset
    with LMDBFootstepDataset(db_path, readonly=False) as dataset:
        sample_count = 0
        
        for attr_combo, files in sorted(attribute_groups.items()):
            speed, concrete, wood, grass = attr_combo
            print(f"Processing combination: speed={speed:.2f}, concrete={concrete:.2f}, "
                  f"wood={wood:.2f}, grass={grass:.2f} ({len(files)} files)")
            
            for i, audio_file in enumerate(files):
                try:
                    # Load audio
                    audio_data, sr = librosa.load(audio_file, sr=44100)
                    audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
                    
                    # Get CLAP embedding
                    embedding = clap_model.get_audio_embedding(audio_tensor, sr)
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.detach().cpu().numpy()
                    embedding = embedding.flatten()
                    
                    # Create comprehensive metadata with all attributes
                    attributes = {
                        'speed': speed,
                        'concrete': concrete,
                        'wood': wood,
                        'grass': grass
                    }
                    
                    metadata = {
                        'original_file': audio_file,
                        'filename': os.path.basename(audio_file),
                        'attributes': attributes,
                        'audio_duration': len(audio_data) / sr,
                        'sample_rate': sr
                    }
                    
                    # Store in LMDB with composite key
                    key = f"spe_{speed:.2f}_con_{concrete:.2f}_woo_{wood:.2f}_gra_{grass:.2f}_{i:03d}"
                    
                    # Use speed as primary label for compatibility, but include all attributes in metadata
                    dataset.put_sample(key, embedding, attributes, metadata)
                    
                    sample_count += 1
                    if sample_count % 50 == 0:
                        print(f"  Processed {sample_count}/{total_files} samples")
                        
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
                    continue
    
    # Print final statistics
    with LMDBFootstepDataset(db_path, readonly=True) as dataset:
        print(f"\n=== Full LMDB Dataset Created ===")
        print(f"Database path: {db_path}")
        print(f"Total samples: {len(dataset)}")
    
    
    return db_path

def main():
    """Main function to create the footstep LMDB dataset."""
    print("=== Footstep Multi-Attribute LMDB Dataset Creator ===\n")
    
    # Configuration
    AUDIO_DIR = "ffxFootstepsGenData"  # Change this to your audio directory
    DB_PATH = "footsteps_full"
    
    # Create the dataset
    db_path = create_full_lmdb_dataset(AUDIO_DIR, DB_PATH)
    
    if db_path:
        print(f"\n=== Processing Complete ===")
        print(f"LMDB database created: {db_path}")
        print(f"You can now load data with RegCAVTrainer.load_continuous_data_from_lmdb()")
    else:
        print("Failed to create dataset")

if __name__ == "__main__":
    main()