from CLAP import CLAP
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import pickle
import json
import lmdb
from typing import List, Tuple, Dict, Optional
import librosa
from fs_lmdb import *

class RegCAVTrainer:
    """Regression-based CAV trainer for continuous concept learning."""
    
    def __init__(self, clap_model: Optional[CLAP] = None):
        """
        Initialize regression CAV trainer with a CLAP model.
        
        Args:
            clap_model: Pre-initialized CLAP model. If None, creates a new one.
        """
        #self.clap_model = clap_model or CLAP()
        self.scaler = StandardScaler()
        self.regressor = None
        self.rcv = None  # Regression Concept Vector
        
    def load_continuous_data_from_files(self, audio_dir: str, 
                                       cache_file: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load audio files and generate CLAP embeddings with continuous speed labels.
        
        Args:
            audio_dir: Directory containing audio files with speed labels
            cache_file: Optional file to cache embeddings
            
        Returns:
            Tuple of (embeddings, speed_values) as numpy arrays
        """
        # Try to load from cache first
        if cache_file and os.path.exists(cache_file):
            print(f"Loading embeddings from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                return cached_data['embeddings'], cached_data['speed_values']
        
        # Find all audio files with speed labels
        audio_files = find_footstep_files(audio_dir)
        
        print(f"Generating embeddings for {len(audio_files)} audio files...")
        embeddings = []
        speed_values = []
        valid_files = []
        
        for i, audio_file in enumerate(audio_files):
            try:
                # Extract speed from filename
                speed = extract_speed_from_filename(os.path.basename(audio_file))
                if speed is None:
                    continue
                
                # Load audio using the same preprocessing as CLAP
                audio_data, sr = librosa.load(audio_file, sr=44100)
                audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
                
                # Get CLAP embedding
                embedding = self.clap_model.get_audio_embedding(audio_tensor, sr)
                
                # Convert to numpy and flatten
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.detach().cpu().numpy()
                embedding = embedding.flatten()
                
                embeddings.append(embedding)
                speed_values.append(speed)
                valid_files.append(audio_file)
                
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(audio_files)} files")
                    
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
        
        embeddings = np.array(embeddings)
        speed_values = np.array(speed_values)
        
        print(f"Generated {len(embeddings)} embeddings with shape {embeddings[0].shape}")
        print(f"Speed range: {speed_values.min():.2f} to {speed_values.max():.2f}")
        
        # Cache the embeddings
        if cache_file:
            print(f"Caching embeddings to: {cache_file}")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'embeddings': embeddings,
                    'speed_values': speed_values,
                    'audio_files': valid_files
                }, f)
        
        return embeddings, speed_values
    
    def train_rcv(self, embeddings: np.ndarray, speed_values: np.ndarray, 
                  test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train a linear regression model and extract the RCV (Regression Concept Vector).
        
        Args:
            embeddings: Array of CLAP embeddings
            speed_values: Continuous speed values
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training results and metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, speed_values, test_size=test_size, 
            random_state=random_state
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Speed range (train): {y_train.min():.2f} to {y_train.max():.2f}")
        print(f"Speed range (test): {y_test.min():.2f} to {y_test.max():.2f}")
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train regressor
        print("Training linear regression...")
        self.regressor = LinearRegression()
        self.regressor.fit(X_train_scaled, y_train)
        
        # Generate predictions
        y_train_pred = self.regressor.predict(X_train_scaled)
        y_test_pred = self.regressor.predict(X_test_scaled)
        
        # Extract and normalize RCV (coefficient vector)
        raw_rcv = self.regressor.coef_  # Get coefficient vector
        self.rcv = raw_rcv / np.linalg.norm(raw_rcv)  # Normalize to unit vector
        
        # Calculate regression metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        results = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'rcv_magnitude': np.linalg.norm(raw_rcv),
            'rcv_shape': self.rcv.shape,
            'intercept': self.regressor.intercept_,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        }
        
        print(f"Training R²: {train_r2:.3f}")
        print(f"Test R²: {test_r2:.3f}")
        print(f"Training RMSE: {train_rmse:.3f}")
        print(f"Test RMSE: {test_rmse:.3f}")
        print(f"RCV shape: {self.rcv.shape}")
        
        return results
    
    def compute_speed_score(self, audio_file: str) -> float:
        """
        Compute predicted speed for a single audio file using the RCV.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Predicted speed value
        """
        if self.rcv is None:
            raise ValueError("RCV not trained yet. Call train_rcv() first.")
        
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
        
        # Predict speed using the full regression model
        predicted_speed = self.regressor.predict(embedding_scaled.reshape(1, -1))[0]
        
        return float(predicted_speed)
    
    def compute_rcv_activation(self, audio_file: str) -> float:
        """
        Compute RCV activation score (dot product with RCV) for a single audio file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            RCV activation score (directional alignment with speed concept)
        """
        if self.rcv is None:
            raise ValueError("RCV not trained yet. Call train_rcv() first.")
        
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
        
        # Compute dot product with RCV (directional activation)
        activation = np.dot(embedding_scaled, self.rcv)
        
        return float(activation)
    
    def evaluate_on_directory(self, audio_dir: str) -> Dict:
        """
        Evaluate trained RCV on all files in a directory.
        
        Args:
            audio_dir: Directory containing audio files with speed labels
            
        Returns:
            Dictionary with evaluation metrics and predictions
        """
        if self.rcv is None:
            raise ValueError("RCV not trained yet.")
        
        # Load all files and extract speeds
        audio_files = find_footstep_files(audio_dir)
        true_speeds = []
        predicted_speeds = []
        rcv_activations = []
        valid_files = []
        
        print(f"Evaluating on {len(audio_files)} files...")
        
        for audio_file in audio_files:
            try:
                # Extract true speed
                true_speed = extract_speed_from_filename(os.path.basename(audio_file))
                if true_speed is None:
                    continue
                
                # Predict speed
                pred_speed = self.compute_speed_score(audio_file)
                rcv_activation = self.compute_rcv_activation(audio_file)
                
                true_speeds.append(true_speed)
                predicted_speeds.append(pred_speed)
                rcv_activations.append(rcv_activation)
                valid_files.append(audio_file)
                
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
        
        true_speeds = np.array(true_speeds)
        predicted_speeds = np.array(predicted_speeds)
        rcv_activations = np.array(rcv_activations)
        
        # Calculate metrics
        r2 = r2_score(true_speeds, predicted_speeds)
        rmse = np.sqrt(mean_squared_error(true_speeds, predicted_speeds))
        mae = mean_absolute_error(true_speeds, predicted_speeds)
        
        # Correlation between true speed and RCV activation
        speed_activation_corr = np.corrcoef(true_speeds, rcv_activations)[0, 1]
        
        results = {
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'speed_activation_correlation': speed_activation_corr,
            'n_samples': len(true_speeds),
            'true_speeds': true_speeds.tolist(),
            'predicted_speeds': predicted_speeds.tolist(),
            'rcv_activations': rcv_activations.tolist(),
            'filenames': [os.path.basename(f) for f in valid_files]
        }
        
        print(f"Evaluation Results:")
        print(f"  R² Score: {r2:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  MAE: {mae:.3f}")
        print(f"  Speed-Activation Correlation: {speed_activation_corr:.3f}")
        
        return results
    
    def save_rcv(self, filepath: str):
        """Save the trained RCV and associated components."""
        if self.rcv is None:
            raise ValueError("No RCV to save. Train an RCV first.")
        
        save_data = {
            'rcv': self.rcv,
            'scaler': self.scaler,
            'regressor': self.regressor
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"RCV saved to {filepath}")
    
    def load_rcv(self, filepath: str):
        """Load a previously trained RCV."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.rcv = save_data['rcv']
        self.scaler = save_data['scaler']
        self.regressor = save_data['regressor']
        
        print(f"RCV loaded from {filepath}")
        print(f"RCV shape: {self.rcv.shape}")

# Example usage
if __name__ == "__main__":
    # Initialize trainer
    trainer = RegCAVTrainer()
    
    # Path to LMDB database 
    db_path = "footsteps_full"
    
    if os.path.exists(db_path):
        print("Training RCV from LMDB dataset...")
        
        # Load all data from LMDB
        embeddings, speeds, metadata = load_all_data_from_lmdb(db_path)
        
        # Train RCV
        results = trainer.train_rcv(embeddings, speeds)
        
        # Save trained RCV
        trainer.save_rcv("footstep_speed_rcv.pkl")
        
        print("\n=== Training Complete ===")
        print(f"RCV trained and saved to: footstep_speed_rcv.pkl")
        
    else:
        print(f"LMDB database not found at {db_path}")
        print("Please run fs_lmdb.py first to create the dataset.")
    
