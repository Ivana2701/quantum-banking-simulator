import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, TwoLocal
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel as QuantumKernel
from qiskit.algorithms.optimizers import COBYLA, SPSA
import warnings
warnings.filterwarnings('ignore')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
csv_path = os.path.join(project_root, 'creditcard.csv')

# Set random seeds for reproducibility
np.random.seed(42)

class CreditCardQSVM:
    def __init__(self, n_qubits=6, feature_map_type='zz', reps=2, entanglement='linear'):
        """
        Initialize Quantum SVM for Credit Card Fraud Detection
        
        Args:
            n_qubits: Number of qubits (increased from 2 to 6)
            feature_map_type: 'zz' or 'pauli' feature map
            reps: Number of repetitions in feature map
            entanglement: Entanglement pattern ('linear', 'circular', 'full')
        """
        self.n_qubits = n_qubits
        self.feature_map_type = feature_map_type
        self.reps = reps
        self.entanglement = entanglement
        self.scaler = RobustScaler()  # More robust to outliers
        self.pca = None
        self.qsvc = None
        self.feature_map = None
        self.quantum_kernel = None
        
    def load_creditcard_data(self, filepath='creditcard.csv', sample_size=None):
        """
        Load and preprocess credit card fraud dataset
        
        Args:
            filepath: Path to creditcard.csv
            sample_size: If provided, sample this many transactions for faster training
        """
        if not os.path.isabs(filepath):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            filepath = os.path.join(project_root, filepath)
        print(f"Loading credit card data from {filepath}...")
        df = pd.read_csv(filepath)
        
        if sample_size:
            # Stratified sampling to maintain fraud ratio
            fraud_indices = df[df['Class'] == 1].index
            normal_indices = df[df['Class'] == 0].index
            
            # Sample fraud cases (keep all if sample_size is large)
            fraud_sample_size = min(len(fraud_indices), sample_size // 10)
            fraud_sample = np.random.choice(fraud_indices, fraud_sample_size, replace=False)
            
            # Sample normal cases
            normal_sample_size = sample_size - fraud_sample_size
            normal_sample = np.random.choice(normal_indices, normal_sample_size, replace=False)
            
            # Combine samples
            sample_indices = np.concatenate([fraud_sample, normal_sample])
            df = df.iloc[sample_indices].reset_index(drop=True)
        
        # Separate features and target
        X = df.drop(['Time', 'Class'], axis=1).values  # Remove Time, keep V1-V28 + Amount
        y = df['Class'].values
        
        print(f"Dataset shape: {X.shape}")
        print(f"Fraud rate: {y.mean()*100:.3f}%")
        print(f"Fraud cases: {y.sum()}, Normal cases: {(y==0).sum()}")
        
        return X, y
    
    def preprocess_features(self, X, y, test_size=0.2, n_components=None):
        """
        Preprocess features with scaling, PCA, and balancing
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Fraction for test set
            n_components: Number of PCA components (default: min(n_qubits, n_features))
        """
        if n_components is None:
            n_components = min(self.n_qubits, X.shape[1])
        
        # Split data first
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Scale features (robust to outliers)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply PCA for dimensionality reduction
        self.pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        # Handle class imbalance with SMOTE
        print("Original training set class distribution:")
        print(f"Normal: {(y_train == 0).sum()}, Fraud: {(y_train == 1).sum()}")
        
        # Use SMOTE for oversampling minority class
        smote = SMOTE(random_state=42, k_neighbors=min(5, (y_train == 1).sum() - 1))
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_pca, y_train)
        
        print("After SMOTE balancing:")
        print(f"Normal: {(y_train_balanced == 0).sum()}, Fraud: {(y_train_balanced == 1).sum()}")
        
        return X_train_balanced, X_test_pca, y_train_balanced, y_test
    
    def create_quantum_feature_map(self, feature_dim):
        """
        Create quantum feature map with increased qubits
        
        Args:
            feature_dim: Dimension of input features
        """
        if self.feature_map_type == 'zz':
            self.feature_map = ZZFeatureMap(
                feature_dimension=feature_dim,
                reps=self.reps,
                entanglement=self.entanglement
            )
        elif self.feature_map_type == 'pauli':
            self.feature_map = PauliFeatureMap(
                feature_dimension=feature_dim,
                reps=self.reps,
                paulis=['Z', 'X', 'ZZ', 'XX', 'ZX']
            )
        else:
            raise ValueError(f"Unknown feature map type: {self.feature_map_type}")
        
        print(f"Created {self.feature_map_type.upper()} feature map:")
        print(f"  - Qubits: {self.feature_map.num_qubits}")
        print(f"  - Repetitions: {self.reps}")
        print(f"  - Entanglement: {self.entanglement}")
        
        return self.feature_map
    
    def train_quantum_svm(self, X_train, y_train):
        """
        Train Quantum SVM with optimized parameters
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        # Create quantum feature map
        feature_map = self.create_quantum_feature_map(X_train.shape[1])
        
        # Create quantum kernel (newer API doesn't need sampler parameter)
        self.quantum_kernel = QuantumKernel(
            feature_map=feature_map
        )
        
        # Create and train QSVC
        self.qsvc = QSVC(quantum_kernel=self.quantum_kernel)
        
        print("Training Quantum SVM...")
        self.qsvc.fit(X_train, y_train)
        print("Training completed!")
        
        return self.qsvc
    
    def evaluate_model(self, X_test, y_test, model_name="Quantum SVM"):
        """
        Evaluate model performance with comprehensive metrics
        
        Args:
            X_test: Test features
            y_test: Test labels
            model_name: Name for reporting
        """
        if self.qsvc is None:
            raise ValueError("Model not trained yet!")
        
        # Make predictions
        y_pred = self.qsvc.predict(X_test)
        y_score = self.qsvc.decision_function(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_score)
        
        # Print results
        print(f"\n{'='*50}")
        print(f"{model_name} Results")
        print(f"{'='*50}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC AUC:   {roc_auc:.4f}")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🔍 Confusion Matrix:")
        print(cm)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_score)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': roc_auc
            },
            'predictions': y_pred.tolist(),
            'scores': y_score.tolist()
        }
    
    def save_metrics(self, metrics, filename):
        """Save metrics to JSON file"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to: {filename}")
    
    def train_and_evaluate(self, sample_size=10000):
        """
        Complete training and evaluation pipeline
        
        Args:
            sample_size: Number of samples to use (None for full dataset)
        """
        # Load data
        X, y = self.load_creditcard_data(sample_size=sample_size)
        
        # Preprocess
        X_train, X_test, y_train, y_test = self.preprocess_features(X, y)
        
        # Train model
        self.train_quantum_svm(X_train, y_train)
        
        # Evaluate
        metrics = self.evaluate_model(X_test, y_test)
        
        # Save results
        script_dir = os.path.dirname(os.path.realpath(__file__))
        metrics_path = os.path.join(script_dir, "metrics_data", "creditcard_qsvm.json")
        self.save_metrics(metrics, metrics_path)
        
        return metrics

def run_creditcard_qsvm():
    """Main function to run the credit card QSVM"""
    print("Starting Credit Card Fraud Detection with Quantum SVM")
    print("="*60)
    
    # Create model with increased qubits
    qsvm = CreditCardQSVM(
        n_qubits=8,           # Increased from 6 to 8 qubits
        feature_map_type='zz', # ZZ feature map
        reps=4,               # Increased repetitions from 3 to 4
        entanglement='linear'
    )
    
    # Train and evaluate (using sample for faster execution)
    metrics = qsvm.train_and_evaluate(sample_size=10000)
    
    print("\n✅ Training and evaluation completed!")
    return metrics

if __name__ == "__main__":
    run_creditcard_qsvm() 