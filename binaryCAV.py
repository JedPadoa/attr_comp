from CLAP import CLAP
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import pickle
import json
import fs_lmdb
from typing import List, Tuple, Dict, Optional
import librosa
from fs_lmdb import LMDBFootstepDataset

class LMDBCAVTrainer:
    """CAV trainer that works with LMDB datasets."""
    
    def __init__(self, clap_model: Optional[CLAP] = None):
        """
        Initialize CAV trainer with a CLAP model.
        
        Args:
            clap_model: Pre-initialized CLAP model. If None, creates a new one.
        """
        self.clap_model = clap_model or CLAP()
        self.scaler = StandardScaler()
        self.classifier = None
        self.cav = None
        
    def load_data_from_lmdb(self, db_path: str, 
                           sample_keys: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load embeddings and labels from LMDB dataset.
        
        Args:
            db_path: Path to LMDB database
            sample_keys: Optional list of specific keys to load. If None, loads all.
            
        Returns:
            Tuple of (embeddings, labels) as numpy arrays
        """
        embeddings = []
        labels = []
        
        with LMDBFootstepDataset(db_path, readonly=True) as dataset:
            if sample_keys is None:
                sample_keys = dataset.get_all_keys()
            
            print(f"Loading {len(sample_keys)} samples from LMDB...")
            
            for i, key in enumerate(sample_keys):
                try:
                    embedding, label, metadata = dataset.get_sample(key)
                    embeddings.append(embedding)
                    labels.append(label)
                    
                    if (i + 1) % 100 == 0:
                        print(f"  Loaded {i + 1}/{len(sample_keys)} samples")
                        
                except Exception as e:
                    print(f"Error loading sample {key}: {e}")
                    continue
        
        embeddings = np.array(embeddings)
        labels = np.array(labels)
        
        print(f"Loaded {len(embeddings)} embeddings with shape {embeddings[0].shape}")
        return embeddings, labels
    
    def train_cav_from_lmdb(self, db_path: str,
                           test_size: float = 0.2, 
                           random_state: int = 42,
                           sgd_params: Optional[Dict] = None) -> Dict:
        """
        Train CAV directly from LMDB dataset.
        
        Args:
            db_path: Path to LMDB database
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            sgd_params: Parameters for SGDClassifier
            
        Returns:
            Dictionary with training results and metrics
        """
        # Load data from LMDB
        embeddings, labels = self.load_data_from_lmdb(db_path)
        
        # Train CAV using loaded data
        return self.train_cav(embeddings, labels, test_size, random_state, sgd_params)
    
    def train_cav(self, embeddings: np.ndarray, labels: np.ndarray, 
                  test_size: float = 0.2, random_state: int = 42,
                  sgd_params: Optional[Dict] = None) -> Dict:
        """
        Train a linear classifier and extract the CAV.
        
        Args:
            embeddings: Array of CLAP embeddings
            labels: Binary labels (0/1)
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            sgd_params: Parameters for SGDClassifier
            
        Returns:
            Dictionary with training results and metrics
        """
        if sgd_params is None:
            sgd_params = {
                'loss': 'hinge',  # SVM-like loss
                'alpha': 0.0001,  # Regularization
                'max_iter': 1000,
                'random_state': random_state,
                'tol': 1e-3
            }
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=test_size, 
            random_state=random_state, stratify=labels
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Positive examples: {np.sum(y_train)} / {len(y_train)} (train), {np.sum(y_test)} / {len(y_test)} (test)")
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classifier
        print("Training SGD classifier...")
        self.classifier = SGDClassifier(**sgd_params)
        self.classifier.fit(X_train_scaled, y_train)
        
        # Generate predictions
        y_train_pred = self.classifier.predict(X_train_scaled)
        y_test_pred = self.classifier.predict(X_test_scaled)
        
        # Extract and normalize CAV (weight vector)
        raw_cav = self.classifier.coef_[0]  # Get weight vector
        self.cav = raw_cav / np.linalg.norm(raw_cav)  # Normalize to unit vector
        
        # Calculate metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        results = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'cav_magnitude': np.linalg.norm(raw_cav),
            'cav_shape': self.cav.shape,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'classification_report': classification_report(y_test, y_test_pred, output_dict=True)
        }
        
        print(f"Training accuracy: {train_acc:.3f}")
        print(f"Test accuracy: {test_acc:.3f}")
        print(f"CAV shape: {self.cav.shape}")
        
        return results
    
    def compute_cav_score(self, audio_file: str) -> float:
        """Compute CAV activation score for a single audio file."""
        if self.cav is None:
            raise ValueError("CAV not trained yet. Call train_cav() first.")
        
        # Load and embed audio
        audio_data, sr = librosa.load(audio_file, sr=44100)
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
        embedding = self.clap_model.get_audio_embedding(audio_tensor, sr)
        
        # Convert to numpy and flatten
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
        embedding = embedding.flatten()
        
        # Standardize using the same scaler used in training
        embedding_scaled = self.scaler.transform(embedding.reshape(1, -1))[0]
        
        # Compute dot product with CAV
        score = np.dot(embedding_scaled, self.cav)
        
        return float(score)
    
    def evaluate_on_lmdb(self, db_path: str, sample_keys: Optional[List[str]] = None) -> Dict:
        """
        Evaluate trained CAV on LMDB dataset.
        
        Args:
            db_path: Path to LMDB database
            sample_keys: Optional list of specific keys to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.cav is None:
            raise ValueError("CAV not trained yet.")
        
        embeddings, labels = self.load_data_from_lmdb(db_path, sample_keys)
        
        # Standardize embeddings
        embeddings_scaled = self.scaler.transform(embeddings)
        
        # Compute CAV scores
        cav_scores = np.dot(embeddings_scaled, self.cav)
        
        # Get predictions (positive if score > 0)
        predictions = (cav_scores > 0).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        
        # Separate scores by true label
        pos_scores = cav_scores[labels == 1]
        neg_scores = cav_scores[labels == 0]
        
        results = {
            'accuracy': accuracy,
            'mean_positive_score': np.mean(pos_scores) if len(pos_scores) > 0 else 0,
            'mean_negative_score': np.mean(neg_scores) if len(neg_scores) > 0 else 0,
            'std_positive_score': np.std(pos_scores) if len(pos_scores) > 0 else 0,
            'std_negative_score': np.std(neg_scores) if len(neg_scores) > 0 else 0,
            'score_separation': np.mean(pos_scores) - np.mean(neg_scores) if len(pos_scores) > 0 and len(neg_scores) > 0 else 0,
            'n_positive': len(pos_scores),
            'n_negative': len(neg_scores),
            'all_scores': cav_scores.tolist(),
            'all_labels': labels.tolist()
        }
        
        print(f"Evaluation Results:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Mean positive score: {results['mean_positive_score']:.3f} ± {results['std_positive_score']:.3f}")
        print(f"  Mean negative score: {results['mean_negative_score']:.3f} ± {results['std_negative_score']:.3f}")
        print(f"  Score separation: {results['score_separation']:.3f}")
        
        return results
    
    def save_cav(self, filepath: str):
        """Save the trained CAV and associated components."""
        if self.cav is None:
            raise ValueError("No CAV to save. Train a CAV first.")
        
        save_data = {
            'cav': self.cav,
            'scaler': self.scaler,
            'classifier': self.classifier
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"CAV saved to {filepath}")
    
    def load_cav(self, filepath: str):
        """Load a previously trained CAV."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.cav = save_data['cav']
        self.scaler = save_data['scaler']
        self.classifier = save_data['classifier']
        
        print(f"CAV loaded from {filepath}")
        print(f"CAV shape: {self.cav.shape}")

# Example usage
if __name__ == "__main__":
    # Initialize trainer
    trainer = LMDBCAVTrainer()
    
    # Path to LMDB database (created by preprocess_data.py)
    db_path = "footstep_embeddings.lmdb"
    
    if os.path.exists(db_path):
        print("Training CAV from LMDB dataset...")
        
        # Train CAV directly from LMDB
        results = trainer.train_cav_from_lmdb(db_path)
        
        # Save trained CAV
        trainer.save_cav("footstep_speed_cav.pkl")
        
        # Evaluate on the same dataset
        eval_results = trainer.evaluate_on_lmdb(db_path)
        
        print("\n=== Training Complete ===")
        print(f"CAV trained and saved to: footstep_speed_cav.pkl")
        
    else:
        print(f"LMDB database not found at {db_path}")
        print("Please run preprocess_data.py first to create the dataset.") 