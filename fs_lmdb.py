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

class LMDBFootstepDataset:
    """LMDB-based dataset for footstep audio embeddings and labels."""
    
    def __init__(self, db_path: str, readonly: bool = True):
        self.db_path = db_path
        self.readonly = readonly
        self._env = None
        self._txn = None
        
    def __enter__(self):
        # Create directory if it doesn't exist
        if not self.readonly:
            os.makedirs(self.db_path, exist_ok=True)
            
        # Open LMDB environment
        self._env = lmdb.open(
            self.db_path,
            readonly=self.readonly,
            lock=False,
            map_size=10 * 1024**3  # 10GB max size
        )
        self._txn = self._env.begin(write=not self.readonly)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._txn:
            if not self.readonly and exc_type is None:
                self._txn.commit()
            else:
                self._txn.abort()
        if self._env:
            self._env.close()
            
    def put_sample(self, key: str, embedding: np.ndarray, label: int, metadata: Dict):
        """Store a sample in the database."""
        if self.readonly:
            raise ValueError("Cannot write to readonly database")
            
        # Prepare data
        data = {
            'embedding': embedding.tobytes(),
            'embedding_shape': embedding.shape,
            'embedding_dtype': str(embedding.dtype),
            'label': label,
            'metadata': metadata
        }
        
        # Serialize and store
        serialized = pickle.dumps(data)
        self._txn.put(key.encode(), serialized)
        
    def get_sample(self, key: str) -> Optional[Tuple[np.ndarray, int, Dict]]:
        """Retrieve a sample from the database."""
        serialized = self._txn.get(key.encode())
        if serialized is None:
            return None
            
        data = pickle.loads(serialized)
        
        # Reconstruct embedding
        embedding = np.frombuffer(
            data['embedding'], 
            dtype=data['embedding_dtype']
        ).reshape(data['embedding_shape'])
        
        return embedding, data['label'], data['metadata']
    
    def get_all_keys(self) -> List[str]:
        """Get all keys in the database."""
        keys = []
        cursor = self._txn.cursor()
        for key, _ in cursor:
            keys.append(key.decode())
        return keys
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        stats = self._env.stat()
        
        # Count samples and get label distribution
        total_samples = 0
        label_counts = {0: 0, 1: 0}
        
        cursor = self._txn.cursor()
        for key, value in cursor:
            total_samples += 1
            try:
                data = pickle.loads(value)
                label = data['label']
                if label in label_counts:
                    label_counts[label] += 1
            except:
                continue
                
        return {
            'total_samples': total_samples,
            'positive_samples': label_counts[1],
            'negative_samples': label_counts[0],
            'db_size_bytes': stats['psize'] * stats['depth'],
            'db_path': self.db_path
        }

def find_footstep_files(base_dir: str = ".") -> List[str]:
    """Find all audio files with speed labels (spe_{value}) in their names."""
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac']
    footstep_files = []
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if (any(file.lower().endswith(ext) for ext in audio_extensions) and 
                'spe_' in file):
                footstep_files.append(os.path.join(root, file))
    
    return footstep_files

def extract_speed_from_filename(filename: str) -> float:
    """Extract speed value from filename containing 'spe_{value}' pattern."""
    pattern = r'spe_(\d+\.?\d*)'
    match = re.search(pattern, filename)
    return float(match.group(1)) if match else None

def group_files_by_speed(files: List[str]) -> Dict[float, List[str]]:
    """Group files by their speed values."""
    speed_groups = {}
    
    for file_path in files:
        filename = os.path.basename(file_path)
        speed = extract_speed_from_filename(filename)
        
        if speed is not None:
            if speed not in speed_groups:
                speed_groups[speed] = []
            speed_groups[speed].append(file_path)
    
    return speed_groups

def create_full_lmdb_dataset(speed_groups: Dict[float, List[str]], 
                            db_path: str = "footstep_full.lmdb",
                            clap_model: Optional[CLAP] = None) -> str:
    """
    Create LMDB dataset with CLAP embeddings for all available speed values and their metadata.
    
    Args:
        speed_groups: Dictionary mapping speeds to file lists
        db_path: Path to LMDB database
        clap_model: Pre-initialized CLAP model
        
    Returns:
        Path to created LMDB database
    """
    # Initialize CLAP model if not provided
    if clap_model is None:
        print("Initializing CLAP model...")
        clap_model = CLAP()
    
    # Count total files across all speeds
    total_files = sum(len(files) for files in speed_groups.values())
    print(f"Creating full LMDB dataset with {total_files} samples across {len(speed_groups)} different speeds")
    
    # Remove existing database
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    
    print(f"Creating LMDB dataset at: {db_path}")
    
    # Create LMDB dataset
    with LMDBFootstepDataset(db_path, readonly=False) as dataset:
        sample_count = 0
        
        for speed, files in sorted(speed_groups.items()):
            print(f"Processing speed {speed:.2f} with {len(files)} files...")
            
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
                    
                    # Create comprehensive metadata
                    metadata = {
                        'original_file': audio_file,
                        'speed': speed,
                        'filename': os.path.basename(audio_file),
                        'speed_category': 'slow' if speed <= 0.5 else 'medium' if speed <= 0.8 else 'fast',
                        'sample_rate': sr,
                        'audio_duration': len(audio_data) / sr
                    }
                    
                    # Store in LMDB with speed as label for compatibility
                    key = f"speed_{speed:.2f}_{i:05d}"
                    dataset.put_sample(key, embedding, speed, metadata)
                    
                    sample_count += 1
                    if sample_count % 50 == 0:
                        print(f"  Processed {sample_count}/{total_files} samples")
                        
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
                    continue
    
    # Print final statistics
    with LMDBFootstepDataset(db_path, readonly=True) as dataset:
        stats = dataset.get_stats()
        print(f"\n=== Full LMDB Dataset Created ===")
        print(f"Database path: {db_path}")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Database size: {stats['db_size_bytes'] / 1024**2:.1f} MB")
        
        # Print speed distribution
        print(f"\nSpeed distribution in dataset:")
        speed_counts = {}
        with LMDBFootstepDataset(db_path, readonly=True) as dataset_read:
            for key in dataset_read.get_all_keys():
                _, _, metadata = dataset_read.get_sample(key)
                speed = metadata['speed']
                speed_counts[speed] = speed_counts.get(speed, 0) + 1
        
        for speed in sorted(speed_counts.keys()):
            print(f"  Speed {speed:.2f}: {speed_counts[speed]} samples")
    
    return db_path

def load_all_data_from_lmdb(db_path: str) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Load all embeddings, labels (speeds), and metadata from LMDB dataset.
    
    Args:
        db_path: Path to LMDB database
        
    Returns:
        Tuple of (embeddings, speeds, metadata_list)
    """
    embeddings = []
    speeds = []
    metadata_list = []
    
    with LMDBFootstepDataset(db_path, readonly=True) as dataset:
        all_keys = dataset.get_all_keys()
        print(f"Loading {len(all_keys)} samples from LMDB...")
        
        for i, key in enumerate(all_keys):
            try:
                embedding, speed, metadata = dataset.get_sample(key)
                embeddings.append(embedding)
                speeds.append(speed)
                metadata_list.append(metadata)
                
                if (i + 1) % 100 == 0:
                    print(f"  Loaded {i + 1}/{len(all_keys)} samples")
                    
            except Exception as e:
                print(f"Error loading sample {key}: {e}")
                continue
    
    embeddings = np.array(embeddings)
    speeds = np.array(speeds)
    
    print(f"Loaded {len(embeddings)} embeddings with shape {embeddings[0].shape}")
    print(f"Speed range: {speeds.min():.2f} to {speeds.max():.2f}")
    print(f"Unique speeds: {sorted(np.unique(speeds))}")
    
    return embeddings, speeds, metadata_list

def filter_lmdb_by_speeds(db_path: str, target_speeds: List[float]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Load embeddings for specific speed values only.
    
    Args:
        db_path: Path to LMDB database
        target_speeds: List of speed values to include
        
    Returns:
        Tuple of (embeddings, speeds, metadata_list) for target speeds only
    """
    embeddings = []
    speeds = []
    metadata_list = []
    
    with LMDBFootstepDataset(db_path, readonly=True) as dataset:
        all_keys = dataset.get_all_keys()
        print(f"Filtering for speeds {target_speeds} from {len(all_keys)} total samples...")
        
        for key in all_keys:
            try:
                embedding, speed, metadata = dataset.get_sample(key)
                if speed in target_speeds:
                    embeddings.append(embedding)
                    speeds.append(speed)
                    metadata_list.append(metadata)
                    
            except Exception as e:
                print(f"Error loading sample {key}: {e}")
                continue
    
    embeddings = np.array(embeddings)
    speeds = np.array(speeds)
    
    print(f"Filtered to {len(embeddings)} embeddings for target speeds")
    for speed in target_speeds:
        count = np.sum(speeds == speed)
        print(f"  Speed {speed:.2f}: {count} samples")
    
    return embeddings, speeds, metadata_list

def main():
    """Main function to process footstep data and create comprehensive LMDB dataset."""
    print("=== Footstep Full LMDB Dataset Creator ===\n")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Configuration
    DB_PATH = "footsteps_full.lmdb"
    
    # Search for footstep files
    print("Searching for footstep files with speed labels...")
    
    search_dirs = [
        'ffxFootstepsGenData'
    ]
    
    all_files = []
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            print(f"Searching in: {search_dir}")
            files = find_footstep_files(search_dir)
            all_files.extend(files)
            print(f"Found {len(files)} files in {search_dir}")
    
    if not all_files:
        print("No footstep files with speed labels found!")
        return
    
    print(f"\nTotal footstep files found: {len(all_files)}")
    
    # Group files by speed
    speed_groups = group_files_by_speed(all_files)
    
    # Print distribution
    print(f"\nFound {len(speed_groups)} different speed values:")
    for speed in sorted(speed_groups.keys()):
        count = len(speed_groups[speed])
        print(f"  Speed {speed:4.2f}: {count:4d} files")
    
    # Create full LMDB dataset with all speeds
    db_path = create_full_lmdb_dataset(speed_groups, DB_PATH)
    
    print(f"\n=== Processing Complete ===")
    print(f"Full LMDB database created: {db_path}")
    print(f"You can now load all data or filter by specific speeds.")

if __name__ == "__main__":
    main()