import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelComparisonVisualizer:
    def __init__(self, metrics_dir="./metrics_data"):
        """
        Initialize the model comparison visualizer
        
        Args:
            metrics_dir: Directory containing model metrics JSON files
        """
        self.metrics_dir = Path(metrics_dir)
        self.models_data = {}
        self.load_all_metrics()
        
    def load_all_metrics(self):
        """Load metrics from all available model files"""
        model_files = {
            'quantum_qsvm': 'quantum_qsvm.json',
            'quantum_vqc': 'quantum_vqc.json',
            'creditcard_qsvm': 'creditcard_qsvm.json',
            'creditcard_vqc': 'creditcard_vqc.json',
            'creditcard_random_forest': 'creditcard_random_forest.json',
            'creditcard_svm': 'creditcard_svm.json',
            'creditcard_logistic': 'creditcard_logistic.json'
        }
        
        for model_name, filename in model_files.items():
            filepath = self.metrics_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        self.models_data[model_name] = json.load(f)
                    print(f"Loaded {model_name}")
                except Exception as e:
                    print(f"Error loading {model_name}: {e}")
            else:
                print(f"{model_name} metrics not found")
    
    def create_performance_comparison(self, save_path="./qsvm_vqc_performance_comparison.png"):
        """Create comprehensive performance comparison visualization"""
        if not self.models_data:
            print("No model data available!")
            return
        
        # Prepare data for plotting
        metrics_df = []
        for model_name, metrics in self.models_data.items():
            if 'error' not in metrics:
                metrics_df.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1-Score': metrics.get('f1_score', 0),
                    'ROC AUC': metrics.get('roc_auc', 0)
                })
        
        if not metrics_df:
            print("No valid metrics found!")
            return
            
        df = pd.DataFrame(metrics_df)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Credit Card Fraud Detection: Model Performance Comparison', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Color coding for model types
        quantum_colors = ['#FF6B6B', '#4ECDC4']  # Red, Teal
        classical_colors = ['#45B7D1', '#96CEB4', '#FFEAA7']  # Blue, Green, Yellow
        
        # 1. Accuracy Comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(df['Model'], df['Accuracy'], 
                       color=['#FF6B6B' if 'Quantum' in model else '#45B7D1' for model in df['Model']])
        ax1.set_title('Accuracy Comparison', fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        self._add_value_labels(ax1, bars1)
        
        # 2. Precision Comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(df['Model'], df['Precision'],
                       color=['#FF6B6B' if 'Quantum' in model else '#45B7D1' for model in df['Model']])
        ax2.set_title('Precision Comparison', fontweight='bold')
        ax2.set_ylabel('Precision')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        self._add_value_labels(ax2, bars2)
        
        # 3. Recall Comparison
        ax3 = axes[0, 2]
        bars3 = ax3.bar(df['Model'], df['Recall'],
                       color=['#FF6B6B' if 'Quantum' in model else '#45B7D1' for model in df['Model']])
        ax3.set_title('Recall Comparison', fontweight='bold')
        ax3.set_ylabel('Recall')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        self._add_value_labels(ax3, bars3)
        
        # 4. F1-Score Comparison
        ax4 = axes[1, 0]
        bars4 = ax4.bar(df['Model'], df['F1-Score'],
                       color=['#FF6B6B' if 'Quantum' in model else '#45B7D1' for model in df['Model']])
        ax4.set_title('F1-Score Comparison', fontweight='bold')
        ax4.set_ylabel('F1-Score')
        ax4.set_ylim(0, 1)
        ax4.tick_params(axis='x', rotation=45)
        self._add_value_labels(ax4, bars4)
        
        # 5. ROC AUC Comparison
        ax5 = axes[1, 1]
        bars5 = ax5.bar(df['Model'], df['ROC AUC'],
                       color=['#FF6B6B' if 'Quantum' in model else '#45B7D1' for model in df['Model']])
        ax5.set_title('ROC AUC Comparison', fontweight='bold')
        ax5.set_ylabel('ROC AUC')
        ax5.set_ylim(0, 1)
        ax5.tick_params(axis='x', rotation=45)
        self._add_value_labels(ax5, bars5)
        
        # 6. Radar Chart for best models
        ax6 = axes[1, 2]
        self._create_radar_chart(ax6, df.head(5))  # Top 5 models
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison chart saved to: {save_path}")
        plt.show()
        
    def _add_value_labels(self, ax, bars):
        """Add value labels on top of bars"""
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    def _create_radar_chart(self, ax, df):
        """Create radar chart for top models"""
        # Prepare data for radar chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot each model
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        for i, (_, row) in enumerate(df.iterrows()):
            values = [row[metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Top Models Radar Chart', fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
    
    def create_roc_curves_comparison(self, save_path="./roc_curves_comparison.png"):
        """Create ROC curves comparison"""
        if not self.models_data:
            print("No model data available!")
            return
        
        plt.figure(figsize=(12, 8))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
        color_idx = 0
        
        for model_name, metrics in self.models_data.items():
            if 'roc_curve' in metrics and 'error' not in metrics:
                fpr = metrics['roc_curve']['fpr']
                tpr = metrics['roc_curve']['tpr']
                auc = metrics['roc_curve']['auc']
                
                display_name = model_name.replace('_', ' ').title()
                plt.plot(fpr, tpr, 
                        label=f'{display_name} (AUC = {auc:.3f})',
                        color=colors[color_idx % len(colors)],
                        linewidth=2)
                color_idx += 1
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison', fontweight='bold', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to: {save_path}")
        plt.show()
    
    def create_confusion_matrices(self, save_path="./confusion_matrices.png"):
        """Create confusion matrices for all models"""
        if not self.models_data:
            print("No model data available!")
            return
        
        n_models = len([m for m in self.models_data.values() if 'confusion_matrix' in m and 'error' not in m])
        if n_models == 0:
            print("No confusion matrices available!")
            return
        
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        fig.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold')
        
        model_idx = 0
        for model_name, metrics in self.models_data.items():
            if 'confusion_matrix' in metrics and 'error' not in metrics:
                cm = np.array(metrics['confusion_matrix'])
                ax = axes[model_idx]
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['Normal', 'Fraud'],
                           yticklabels=['Normal', 'Fraud'])
                
                display_name = model_name.replace('_', ' ').title()
                ax.set_title(f'{display_name}\nAccuracy: {metrics.get("accuracy", 0):.3f}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                
                model_idx += 1
        
        # Hide empty subplots
        for i in range(model_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved to: {save_path}")
        plt.show()
    
    def create_summary_table(self, save_path="./model_summary_table.png"):
        """Create a summary table visualization"""
        if not self.models_data:
            print("No model data available!")
            return
        
        # Prepare data
        summary_data = []
        for model_name, metrics in self.models_data.items():
            if 'error' not in metrics:
                summary_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
                    'Precision': f"{metrics.get('precision', 0):.4f}",
                    'Recall': f"{metrics.get('recall', 0):.4f}",
                    'F1-Score': f"{metrics.get('f1_score', 0):.4f}",
                    'ROC AUC': f"{metrics.get('roc_auc', 0):.4f}"
                })
        
        if not summary_data:
            print("No valid metrics found!")
            return
        
        df = pd.DataFrame(summary_data)
        
        # Create table
        fig, ax = plt.subplots(figsize=(12, len(df) * 0.5 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table with colors
        table = ax.table(cellText=df.values,
                        colLabels=df.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Color header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows alternately
        for i in range(1, len(df) + 1):
            color = '#F7F7F7' if i % 2 == 0 else 'white'
            for j in range(len(df.columns)):
                table[(i, j)].set_facecolor(color)
        
        plt.title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary table saved to: {save_path}")
        plt.show()
    
    def create_all_visualizations(self):
        """Create all visualizations"""
        print("Creating comprehensive model comparison visualizations...")
        
        # Ensure output directory exists
        output_dir = Path("./visualizations")
        output_dir.mkdir(exist_ok=True)
        
        # Create all visualizations
        self.create_performance_comparison(str(output_dir / "qsvm_vqc_performance_comparison.png"))
        self.create_roc_curves_comparison(str(output_dir / "roc_curves_comparison.png"))
        self.create_confusion_matrices(str(output_dir / "confusion_matrices.png"))
        self.create_summary_table(str(output_dir / "summary_table.png"))
        
        print("All visualizations created successfully!")
        print(f"Check the '{output_dir}' directory for all charts")

def main():
    """Main function to run the visualization"""
    print("Starting Model Comparison Visualization")
    print("="*50)
    
    # Create visualizer
    visualizer = ModelComparisonVisualizer()
    
    # Create all visualizations
    visualizer.create_all_visualizations()
    
    # Print summary
    print("\n📊 Available Models:")
    for model_name in visualizer.models_data.keys():
        print(f"  - {model_name}")

if __name__ == "__main__":
    main() 