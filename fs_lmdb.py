import os
import shutil
import re
import random
import numpy as np
import lmdb
import pickle
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
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
        self._keys = None  # Cache keys for indexing
        
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
        
        # Cache all keys for indexing
        self._keys = self.get_all_keys()
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._txn:
            if not self.readonly and exc_type is None:
                self._txn.commit()
            else:
                self._txn.abort()
        if self._env:
            self._env.close()
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self._keys is None:
            # If not in context manager, temporarily open to get length
            with lmdb.open(self.db_path, readonly=True, lock=False) as env:
                with env.begin() as txn:
                    return txn.stat()['entries']
        return len(self._keys)
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, Dict, Dict]:
        """Get a sample by index."""
        if self._keys is None:
            raise RuntimeError("Dataset must be used as context manager for indexing")
        
        if index < 0:
            index = len(self._keys) + index  # Support negative indexing
            
        if index >= len(self._keys):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self._keys)}")
        
        key = self._keys[index]
        return self.get_sample(key)
        
    def put_sample(self, key: str, embedding: np.ndarray, 
                   attributes: Dict, metadata: Dict):
        """
        Store a sample in the database.
        
        Args:
            key: Unique key for the sample
            embedding: Audio embedding as numpy array
            attributes: Dictionary of attributes (speed, concrete, wood, grass, etc.)
            metadata: Additional metadata dictionary
        """
        if self.readonly:
            raise ValueError("Cannot write to readonly database")
            
        # Prepare data
        data = {
            'embedding': embedding.tobytes(),
            'embedding_shape': embedding.shape,
            'embedding_dtype': str(embedding.dtype),
            'attributes': attributes,
            'metadata': metadata
        }
        
        # Serialize and store
        serialized = pickle.dumps(data)
        self._txn.put(key.encode(), serialized)
        
    def get_sample(self, key: str) -> Optional[Tuple[np.ndarray, Dict, Dict]]:
        """
        Retrieve a sample from the database.
        
        Returns:
            Tuple of (embedding, attributes_dict, metadata)
        """
        serialized = self._txn.get(key.encode())
        if serialized is None:
            return None
            
        data = pickle.loads(serialized)
        
        # Reconstruct embedding
        embedding = np.frombuffer(
            data['embedding'], 
            dtype=data['embedding_dtype']
        ).reshape(data['embedding_shape'])
        
        # Handle both old and new formats
        if 'attributes' in data:
            # New format - attributes stored separately
            attributes = data['attributes']
        elif 'label' in data and isinstance(data['label'], dict):
            # Transition format - label is attributes dict
            attributes = data['label']
        elif 'label' in data:
            # Old format - single label, try to extract from metadata
            if 'attributes' in data['metadata']:
                attributes = data['metadata']['attributes']
            else:
                # Fallback - create attributes dict with speed
                attributes = {'speed': data['label']}
        else:
            # Default fallback
            attributes = {}
        
        return embedding, attributes, data['metadata']
    
    def get_all_keys(self) -> List[str]:
        """Get all keys in the database."""
        keys = []
        cursor = self._txn.cursor()
        for key, _ in cursor:
            keys.append(key.decode())
        return keys
    
    def get_attribute_stats(self) -> Dict:
        """Get statistics about attributes in the dataset."""
        if self._keys is None:
            raise RuntimeError("Dataset must be used as context manager")
            
        attr_stats = {}
        sample_count = 0
        
        for key in self._keys:
            try:
                _, attributes, metadata = self.get_sample(key)
                
                for attr_name, value in attributes.items():
                    if attr_name not in attr_stats:
                        attr_stats[attr_name] = {'values': set(), 'count': 0}
                    attr_stats[attr_name]['values'].add(value)
                    attr_stats[attr_name]['count'] += 1
                
                sample_count += 1
                if sample_count >= 1000:  # Limit sampling for performance
                    break
                    
            except Exception as e:
                continue
        
        # Convert sets to sorted lists
        for attr_name in attr_stats:
            attr_stats[attr_name]['values'] = sorted(list(attr_stats[attr_name]['values']))
            
        return attr_stats
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        stats = self._env.stat()
        
        # Count samples
        total_samples = len(self._keys) if self._keys else 0
        
        return {
            'total_samples': total_samples,
            'db_size_bytes': stats['psize'] * stats['depth'],
            'db_path': self.db_path
        }