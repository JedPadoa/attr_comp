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
from fs_lmdb import LMDBFootstepDataset
import matplotlib.pyplot as plt

class RegCAVTrainer:
    """Regression-based CAV trainer for continuous concept learning on any attribute."""
    
    def __init__(self, clap_model: Optional[CLAP] = None):
        """
        Initialize regression CAV trainer with a CLAP model.
        
        Args:
            clap_model: Pre-initialized CLAP model. If None, creates a new one.
        """
        self.clap_model = clap_model  # Allow None for now
        self.scaler = StandardScaler()
        self.regressor = None
        self.rcv = None  # Regression Concept Vector
        self.attribute_name = None  # Track which attribute this trainer is for
    
    def load_continuous_data_from_lmdb(self, db_path: str, 
                                     attribute_name: str,
                                     target_values: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load CLAP embeddings and continuous attribute labels from LMDB database.
        
        Args:
            db_path: Path to LMDB database created by fs_lmdb.py
            attribute_name: Name of the attribute to extract ('speed', 'wood', 'concrete', 'grass')
            target_values: Optional list of specific attribute values to filter for.
                          If None, loads all available values.
            
        Returns:
            Tuple of (embeddings, attribute_values) as numpy arrays
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"LMDB database not found at: {db_path}")
        
        embeddings = []
        attribute_values = []
        
        with LMDBFootstepDataset(db_path, readonly=True) as dataset:
            for i in range(len(dataset)):
                try:
                    embedding, attributes, metadata = dataset[i]
                    
                    # Extract the specified attribute
                    if attribute_name not in attributes:
                        print(f"Warning: Attribute '{attribute_name}' not found in sample {i}")
                        continue
                    
                    attr_value = attributes[attribute_name]
                    
                    # Filter by target values if specified
                    if target_values is not None and attr_value not in target_values:
                        continue
                    
                    embeddings.append(embedding)
                    attribute_values.append(attr_value)
                    
                except Exception as e:
                    continue
        
        if len(embeddings) == 0:
            raise ValueError(f"No data loaded from LMDB database for attribute '{attribute_name}'")
        
        embeddings = np.array(embeddings)
        attribute_values = np.array(attribute_values)
        
        # Store which attribute this trainer is for
        self.attribute_name = attribute_name
        
        print(f"✓ Loaded {len(embeddings)} samples for attribute '{attribute_name}'")
        print(f"{attribute_name.capitalize()} range: {attribute_values.min():.2f} to {attribute_values.max():.2f}")
        
        # Show distribution
        unique_values, counts = np.unique(attribute_values, return_counts=True)
        print(f"{attribute_name.capitalize()} distribution:")
        for value, count in zip(unique_values, counts):
            print(f"  {attribute_name}={value:.2f}: {count} samples")
        
        return embeddings, attribute_values
    
    def train_rcv(self, embeddings: np.ndarray, attribute_values: np.ndarray, 
                  test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train a linear regression model and extract the RCV (Regression Concept Vector).
        
        Args:
            embeddings: Array of CLAP embeddings
            attribute_values: Continuous attribute values
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training results and metrics
        """
        if self.attribute_name is None:
            self.attribute_name = "attribute"  # fallback
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, attribute_values, test_size=test_size, 
            random_state=random_state
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"{self.attribute_name.capitalize()} range (train): {y_train.min():.2f} to {y_train.max():.2f}")
        print(f"{self.attribute_name.capitalize()} range (test): {y_test.min():.2f} to {y_test.max():.2f}")
        
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
        
        # Plot training performance
        plt.figure(figsize=(8, 6))
        plt.scatter(y_train, y_train_pred, alpha=0.6, s=20)
        
        # Add perfect prediction line
        min_val = min(y_train.min(), y_train_pred.min())
        max_val = max(y_train.max(), y_train_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
                 linewidth=2, label='Perfect Prediction')
        
        plt.xlabel(f'True {self.attribute_name.capitalize()}')
        plt.ylabel(f'Predicted {self.attribute_name.capitalize()}')
        plt.title(f'{self.attribute_name.capitalize()} Training Performance (R² = {train_r2:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        
        results = {
            'attribute_name': self.attribute_name,
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
    
    def compute_attribute_score(self, audio_file: str) -> float:
        """
        Compute predicted attribute value for a single audio file using the RCV.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Predicted attribute value
        """
        if self.rcv is None:
            raise ValueError("RCV not trained yet. Call train_rcv() first.")
        if self.clap_model is None:
            raise ValueError("CLAP model not provided. Cannot process audio files.")
        
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
        
        # Predict attribute value using the full regression model
        predicted_value = self.regressor.predict(embedding_scaled.reshape(1, -1))[0]
        
        return float(predicted_value)
    
    def compute_rcv_activation(self, audio_file: str) -> float:
        """
        Compute RCV activation score (dot product with RCV) for a single audio file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            RCV activation score (directional alignment with attribute concept)
        """
        if self.rcv is None:
            raise ValueError("RCV not trained yet. Call train_rcv() first.")
        if self.clap_model is None:
            raise ValueError("CLAP model not provided. Cannot process audio files.")
        
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
    
    def save_rcv(self, filepath: str):
        """Save the trained RCV and associated components."""
        if self.rcv is None:
            raise ValueError("No RCV to save. Train an RCV first.")
        
        save_data = {
            'rcv': self.rcv,
            'scaler': self.scaler,
            'regressor': self.regressor,
            'attribute_name': self.attribute_name
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"RCV for '{self.attribute_name}' saved to {filepath}")
    
    def load_rcv(self, filepath: str):
        """Load a previously trained RCV."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.rcv = save_data['rcv']
        self.scaler = save_data['scaler']
        self.regressor = save_data['regressor']
        self.attribute_name = save_data.get('attribute_name', 'unknown')
        
        print(f"RCV for '{self.attribute_name}' loaded from {filepath}")
        print(f"RCV shape: {self.rcv.shape}")

# Example usage
if __name__ == "__main__":
    # Initialize trainer
    clap = CLAP()
    trainer = RegCAVTrainer(clap)
    
    # Path to LMDB database 
    db_path = "footsteps_full"
    
    if os.path.exists(db_path):
        # Train for different attributes
        for attribute in ['speed', 'wood', 'concrete', 'grass']:
            print(f"\n=== Training RCV for {attribute} ===")
            
            # Load data for this attribute
            embeddings, values = trainer.load_continuous_data_from_lmdb(db_path, attribute)
            
            # Train RCV
            results = trainer.train_rcv(embeddings, values)
            
            # Save trained RCV
            trainer.save_rcv(f"{attribute}_rcv.pkl")
            
            print(f"RCV for {attribute} trained and saved")
    else:
        print(f"LMDB database not found at {db_path}")
    
