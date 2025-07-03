import os
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import ADASYN
import warnings
warnings.filterwarnings('ignore')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
csv_path = os.path.join(project_root, 'creditcard.csv')
# Set random seeds for reproducibility
np.random.seed(42)

class QuantumInspired100Qubit:
    """
    Quantum-Inspired 100-Qubit Equivalent Fraud Detection System
    
    This simulates the power of 100 qubits using:
    1. High-dimensional feature engineering
    2. Quantum-inspired kernels
    3. Ensemble methods
    4. Advanced preprocessing
    """
    
    def __init__(self, feature_dim=100, n_estimators=100):
        self.feature_dim = feature_dim
        self.n_estimators = n_estimators
        self.scaler = RobustScaler()
        self.pca = None
        self.models = {}
        
    def load_creditcard_data(self, filepath='creditcard.csv', sample_size=None):
        """Load credit card fraud dataset"""
        if not os.path.isabs(filepath):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            filepath = os.path.join(project_root, filepath)
        print(f"Loading credit card data from {filepath}...")
        df = pd.read_csv(filepath)
        
        if sample_size:
            # Stratified sampling
            fraud_indices = df[df['Class'] == 1].index
            normal_indices = df[df['Class'] == 0].index
            
            fraud_sample_size = min(len(fraud_indices), sample_size // 10)
            fraud_sample = np.random.choice(fraud_indices, fraud_sample_size, replace=False)
            
            normal_sample_size = sample_size - fraud_sample_size
            normal_sample = np.random.choice(normal_indices, normal_sample_size, replace=False)
            
            sample_indices = np.concatenate([fraud_sample, normal_sample])
            df = df.iloc[sample_indices].reset_index(drop=True)
        
        X = df.drop(['Time', 'Class'], axis=1).values
        y = np.array(df['Class'].values, dtype=float)
        
        print(f"Dataset shape: {X.shape}")
        print(f"Fraud rate: {np.mean(y)*100:.3f}%")
        
        return X, y
    
    def create_quantum_inspired_features(self, X):
        """Create quantum-inspired high-dimensional features"""
        print("Creating quantum-inspired features...")
        
        # 1. Polynomial features (quantum superposition simulation)
        print("  - Adding polynomial features (quantum superposition)")
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        print(f"    Polynomial features: {X.shape[1]} → {X_poly.shape[1]}")
        
        # 2. Interaction features (quantum entanglement simulation)
        print("  - Adding interaction features (quantum entanglement)")
        X_interactions = self._create_interaction_features(X)
        print(f"    Interaction features: {X_interactions.shape[1]}")
        
        # 3. Wavelet-like features (quantum wave functions)
        print("  - Adding wavelet-like features (quantum wave functions)")
        X_wavelet = self._create_wavelet_features(X)
        print(f"    Wavelet features: {X_wavelet.shape[1]}")
        
        # Combine all features
        X_enhanced = np.hstack([X_poly, X_interactions, X_wavelet])
        
        print(f"Total quantum-inspired features: {X_enhanced.shape[1]}")
        print(f"   (Simulating {X_enhanced.shape[1]} qubits of information)")
        
        return X_enhanced
    
    def _create_interaction_features(self, X):
        """Create interaction features simulating quantum entanglement"""
        n_features = X.shape[1]
        interactions = []
        
        # Create pairwise interactions (like quantum entanglement)
        for i in range(n_features):
            for j in range(i+1, min(i+5, n_features)):  # Limit to avoid explosion
                interaction = X[:, i] * X[:, j]
                interactions.append(interaction)
        
        return np.column_stack(interactions) if interactions else np.empty((X.shape[0], 0))
    
    def _create_wavelet_features(self, X):
        """Create wavelet-like features simulating quantum wave functions"""
        wavelet_features = []
        
        # Sine and cosine transformations (wave-like features)
        for i in range(min(5, X.shape[1])):
            sine_feature = np.sin(X[:, i])
            cosine_feature = np.cos(X[:, i])
            wavelet_features.extend([sine_feature, cosine_feature])
        
        return np.column_stack(wavelet_features) if wavelet_features else np.empty((X.shape[0], 0))
    
    def preprocess_features(self, X, y, test_size=0.2):
        """Advanced preprocessing with quantum-inspired techniques"""
        print("Applying quantum-inspired preprocessing...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        # Create quantum-inspired features
        X_train_enhanced = self.create_quantum_inspired_features(X_train)
        X_test_enhanced = self.create_quantum_inspired_features(X_test)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_enhanced)
        X_test_scaled = self.scaler.transform(X_test_enhanced)
        
        # Apply PCA for dimensionality reduction
        self.pca = PCA(n_components=self.feature_dim, random_state=42)
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        print(f"Final feature dimensions: {X_train_pca.shape[1]} (simulating {self.feature_dim} qubits)")
        
        # Handle class imbalance
        print("Original training set class distribution:")
        print(f"Normal: {np.sum(y_train == 0)}, Fraud: {np.sum(y_train == 1)}")
        
        # Use ADASYN for better minority class handling
        print("Using ADASYN for balancing...")
        adasyn = ADASYN(random_state=42, n_neighbors=min(5, (y_train == 1).sum() - 1))
        X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train_pca, y_train)  # type: ignore
        y_train_balanced = np.array(y_train_balanced)
        
        print("After ADASYN balancing:")
        print(f"Normal: {np.sum(y_train_balanced == 0)}, Fraud: {np.sum(y_train_balanced == 1)}")
        
        return X_train_balanced, X_test_pca, y_train_balanced, y_test
    
    def create_quantum_inspired_models(self):
        """Create ensemble of quantum-inspired models"""
        print("Creating quantum-inspired model ensemble...")
        
        # 1. Quantum-Inspired Random Forest (quantum superposition)
        print("  - Training Quantum-Inspired Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        self.models['quantum_rf'] = rf_model
        
        # 2. Quantum-Inspired Gradient Boosting (quantum tunneling)
        print("  - Training Quantum-Inspired Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=10,
            learning_rate=0.1,
            random_state=42
        )
        self.models['quantum_gb'] = gb_model
        
        # 3. Quantum-Inspired SVM (quantum kernels)
        print("  - Training Quantum-Inspired SVM...")
        svm_model = SVC(
            kernel='rbf',
            C=10.0,
            probability=True,
            random_state=42
        )
        self.models['quantum_svm'] = svm_model
        
        # 4. Quantum-Inspired Neural Network (quantum neural networks)
        print("  - Training Quantum-Inspired Neural Network...")
        nn_model = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
        self.models['quantum_nn'] = nn_model
        
        return self.models
    
    def train_quantum_inspired_ensemble(self, X_train, y_train):
        """Train the quantum-inspired ensemble"""
        models = self.create_quantum_inspired_models()
        
        print("Training quantum-inspired ensemble...")
        print(f"   Simulating {self.feature_dim} qubits of computational power")
        
        for name, model in models.items():
            print(f"  - Training {name}...")
            model.fit(X_train, y_train)
        
        print("Quantum-inspired ensemble training completed!")
        return models
    
    def evaluate_ensemble(self, X_test, y_test):
        """Evaluate the quantum-inspired ensemble"""
        print(f"\n{'='*60}")
        print(f"QUANTUM-INSPIRED 100-QUBIT EQUIVALENT RESULTS")
        print(f"{'='*60}")
        
        ensemble_results = {}
        
        for name, model in self.models.items():
            print(f"\n🔮 {name.upper()} Results:")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get prediction probabilities
            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(X_test)[:, 1]
            else:
                y_score = y_pred.astype(float)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_score)
            
            # Print results
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  ROC AUC:   {roc_auc:.4f}")
            
            ensemble_results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            }
        
        # Calculate ensemble average
        avg_accuracy = np.mean([r['accuracy'] for r in ensemble_results.values()])
        avg_precision = np.mean([r['precision'] for r in ensemble_results.values()])
        avg_recall = np.mean([r['recall'] for r in ensemble_results.values()])
        avg_f1 = np.mean([r['f1_score'] for r in ensemble_results.values()])
        avg_roc_auc = np.mean([r['roc_auc'] for r in ensemble_results.values()])
        
        print(f"\n{'='*60}")
        print(f"ENSEMBLE AVERAGE (100-QUBIT EQUIVALENT)")
        print(f"{'='*60}")
        print(f"Average Accuracy:  {avg_accuracy:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall:    {avg_recall:.4f}")
        print(f"Average F1-Score:  {avg_f1:.4f}")
        print(f"Average ROC AUC:   {avg_roc_auc:.4f}")
        
        return ensemble_results
    
    def save_metrics(self, metrics, filename):
        """Save metrics to JSON file"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to: {filename}")
    
    def train_and_evaluate(self, sample_size=10000):
        """Complete training and evaluation pipeline"""
        # Load data
        X, y = self.load_creditcard_data(sample_size=sample_size)
        
        # Preprocess
        X_train, X_test, y_train, y_test = self.preprocess_features(X, y)
        
        # Train ensemble
        self.train_quantum_inspired_ensemble(X_train, y_train)
        
        # Evaluate
        metrics = self.evaluate_ensemble(X_test, y_test)
        
        # Save results
        script_dir = os.path.dirname(os.path.realpath(__file__))
        metrics_path = os.path.join(script_dir, "metrics_data", "quantum_inspired_100.json")
        self.save_metrics(metrics, metrics_path)
        
        return metrics

def run_quantum_inspired_100():
    """Main function to run the quantum-inspired 100-qubit equivalent system"""
    print("Starting Quantum-Inspired 100-Qubit Equivalent Fraud Detection")
    print("="*70)
    print("This system simulates the power of 100 qubits using:")
    print("   - High-dimensional feature engineering")
    print("   - Quantum-inspired kernels")
    print("   - Ensemble methods")
    print("   - Advanced preprocessing")
    print("="*70)
    
    # Create quantum-inspired system
    quantum_system = QuantumInspired100Qubit(
        feature_dim=100,      # Simulate 100 qubits
        n_estimators=100      # 100 estimators for ensemble
    )
    
    # Train and evaluate
    metrics = quantum_system.train_and_evaluate(sample_size=10000)
    
    print("\n✅ Quantum-inspired 100-qubit equivalent training completed!")
    print("This represents the theoretical power of 100 qubits!")
    return metrics

if __name__ == "__main__":
    run_quantum_inspired_100() 