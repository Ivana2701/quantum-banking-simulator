import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

class CreditCardClassical:
    def __init__(self, model_type='random_forest'):
        """
        Initialize Classical ML models for Credit Card Fraud Detection
        
        Args:
            model_type: 'random_forest', 'svm', or 'logistic'
        """
        self.model_type = model_type
        self.scaler = RobustScaler()
        self.pca = None
        self.model = None
        
    def load_creditcard_data(self, filepath='creditcard.csv', sample_size=None):
        """
        Load and preprocess credit card fraud dataset
        
        Args:
            filepath: Path to creditcard.csv (relative to project root or absolute)
            sample_size: If provided, sample this many transactions for faster training
        """
        # Always resolve the path relative to the project root
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
        y = np.array(df['Class'].values, dtype=float)
        
        print(f"Dataset shape: {X.shape}")
        print(f"Fraud rate: {np.mean(y)*100:.3f}%")
        print(f"Fraud cases: {np.sum(y==1)}, Normal cases: {np.sum(y==0)}")
        
        return X, y
    
    def preprocess_features(self, X, y, test_size=0.2, n_components=6):
        """
        Preprocess features with scaling, PCA, and balancing
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Fraction for test set
            n_components: Number of PCA components
        """
        # Split data first
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        # Scale features (robust to outliers)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply PCA for dimensionality reduction
        self.pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        # Handle class imbalance with SMOTE
        print("Original training set class distribution:")
        print(f"Normal: {np.sum(y_train == 0)}, Fraud: {np.sum(y_train == 1)}")
        
        # Use SMOTE for oversampling minority class
        smote = SMOTE(random_state=42, k_neighbors=min(5, int(np.sum(y_train == 1)) - 1))
        resampled = smote.fit_resample(X_train_pca, y_train)
        if len(resampled) == 2:
            X_train_balanced, y_train_balanced = resampled
        else:
            X_train_balanced, y_train_balanced, *_ = resampled
        y_train_balanced = np.array(y_train_balanced)
        
        print("After SMOTE balancing:")
        print(f"Normal: {np.sum(y_train_balanced == 0)}, Fraud: {np.sum(y_train_balanced == 1)}")
        
        return X_train_balanced, X_test_pca, y_train_balanced, y_test
    
    def create_model(self):
        """
        Create classical ML model based on specified type
        
        Returns:
            Model instance
        """
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"Created {self.model_type.upper()} model")
        return self.model
    
    def train_model(self, X_train, y_train):
        """
        Train classical ML model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        # Create model
        model = self.create_model()
        
        print(f"Training {self.model_type.upper()}...")
        model.fit(X_train, y_train)
        print("Training completed!")
        
        self.model = model
        return model
    
    def evaluate_model(self, X_test, y_test, model_name=None):
        """
        Evaluate model performance with comprehensive metrics
        
        Args:
            X_test: Test features
            y_test: Test labels
            model_name: Name for reporting
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        if model_name is None:
            model_name = f"Classical {self.model_type.upper()}"
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Get prediction probabilities
        try:
            if hasattr(self.model, 'predict_proba'):
                y_score = self.model.predict_proba(X_test)
                # Handle case where predict_proba returns a list (e.g., VotingClassifier)
                if isinstance(y_score, list):
                    import numpy as np
                    y_score = np.mean([proba[:, 1] if proba.shape[1] > 1 else proba.ravel() for proba in y_score], axis=0)
                else:
                    # If binary classification, take probability of class 1
                    if len(y_score.shape) > 1 and y_score.shape[1] > 1:
                        y_score = y_score[:, 1]
                    else:
                        y_score = y_score.ravel()
            else:
                y_score = y_pred.astype(float)
        except Exception as e:
            print(f"Error obtaining prediction probabilities: {e}")
            y_score = y_pred.astype(float)
        
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
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_score),
            'confusion_matrix': cm.tolist(),
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': roc_auc_score(y_test, y_score)
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
        self.train_model(X_train, y_train)
        
        # Evaluate
        metrics = self.evaluate_model(X_test, y_test)
        
        # Save results
        script_dir = os.path.dirname(os.path.realpath(__file__))
        metrics_path = os.path.join(script_dir, "metrics_data", f"creditcard_{self.model_type}.json")
        self.save_metrics(metrics, metrics_path)
        
        return metrics

def run_all_classical_models():
    """Run all classical models for comparison"""
    print("🚀 Starting Credit Card Fraud Detection with Classical ML Models")
    print("="*65)
    
    models = ['random_forest', 'svm', 'logistic']
    results = {}
    
    for model_type in models:
        print(f"\n{'='*20} {model_type.upper()} {'='*20}")
        
        # Create and train model
        classical = CreditCardClassical(model_type=model_type)
        metrics = classical.train_and_evaluate(sample_size=10000)
        results[model_type] = metrics
        
        print(f"✅ {model_type.upper()} completed!")
    
    # Compare results
    print(f"\n{'='*60}")
    print("📊 CLASSICAL MODELS COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC AUC':<10}")
    print("-" * 70)
    
    for model_type, metrics in results.items():
        accuracy = metrics.get('accuracy', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        f1 = metrics.get('f1_score', 0)
        roc_auc = metrics.get('roc_auc', 0)
        print(f"{model_type.upper():<15} {accuracy:<10.4f} {precision:<10.4f} "
              f"{recall:<10.4f} {f1:<10.4f} {roc_auc:<10.4f}")
    
    return results

if __name__ == "__main__":
    run_all_classical_models() 