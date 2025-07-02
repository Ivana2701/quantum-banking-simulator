import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class Quantum100ComparisonVisualizer:
    """
    Comprehensive visualization comparing quantum-inspired 100-qubit equivalent
    with all previous quantum and classical models
    """
    
    def __init__(self):
        self.metrics_dir = "metrics_data"
        self.output_dir = "visualizations"
        self.results = {}
        
    def load_all_metrics(self):
        """Load all available metrics from JSON files"""
        print("📊 Loading all model metrics...")
        
        # Load quantum-inspired 100-qubit results
        quantum_100_path = os.path.join(self.metrics_dir, "quantum_inspired_100.json")
        if os.path.exists(quantum_100_path):
            with open(quantum_100_path, 'r') as f:
                quantum_100_data = json.load(f)
                
            # Calculate ensemble averages
            accuracies = [model['accuracy'] for model in quantum_100_data.values()]
            precisions = [model['precision'] for model in quantum_100_data.values()]
            recalls = [model['recall'] for model in quantum_100_data.values()]
            f1_scores = [model['f1_score'] for model in quantum_100_data.values()]
            roc_aucs = [model['roc_auc'] for model in quantum_100_data.values()]
            
            self.results['Quantum-Inspired 100-Qubit (Ensemble)'] = {
                'accuracy': np.mean(accuracies),
                'precision': np.mean(precisions),
                'recall': np.mean(recalls),
                'f1_score': np.mean(f1_scores),
                'roc_auc': np.mean(roc_aucs),
                'type': 'quantum_inspired'
            }
            
            # Add individual models
            for name, metrics in quantum_100_data.items():
                self.results[f'Quantum-Inspired {name.replace("_", " ").title()}'] = {
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'roc_auc': metrics['roc_auc'],
                    'type': 'quantum_inspired'
                }
        
        # Load classical models
        classical_models = ['creditcard_random_forest', 'creditcard_svm', 'creditcard_logistic']
        for model in classical_models:
            model_path = os.path.join(self.metrics_dir, f"{model}.json")
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract metrics from the large JSON files
                    if 'accuracy' in data:
                        model_name = model.replace('creditcard_', '').replace('_', ' ').title()
                        self.results[f'Classical {model_name}'] = {
                            'accuracy': data.get('accuracy', 0),
                            'precision': data.get('precision', 0),
                            'recall': data.get('recall', 0),
                            'f1_score': data.get('f1_score', 0),
                            'roc_auc': data.get('roc_auc', 0),
                            'type': 'classical'
                        }
                except Exception as e:
                    print(f"Warning: Could not load {model}: {e}")
        
        # Load quantum models
        quantum_models = ['quantum_qsvm', 'quantum_vqc']
        for model in quantum_models:
            model_path = os.path.join(self.metrics_dir, f"{model}.json")
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'r') as f:
                        data = json.load(f)
                    
                    if 'accuracy' in data:
                        model_name = model.replace('quantum_', '').upper()
                        self.results[f'Quantum {model_name}'] = {
                            'accuracy': data.get('accuracy', 0),
                            'precision': data.get('precision', 0),
                            'recall': data.get('recall', 0),
                            'f1_score': data.get('f1_score', 0),
                            'roc_auc': data.get('roc_auc', 0),
                            'type': 'quantum'
                        }
                except Exception as e:
                    print(f"Warning: Could not load {model}: {e}")
        
        print(f"✅ Loaded {len(self.results)} models")
        return self.results
    
    def create_performance_comparison_chart(self):
        """Create comprehensive performance comparison chart"""
        print("📈 Creating performance comparison chart...")
        
        # Prepare data
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('🔬 Quantum-Inspired 100-Qubit vs All Models Performance Comparison', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Color coding
        colors = {
            'quantum_inspired': '#FF6B6B',  # Red for quantum-inspired
            'quantum': '#4ECDC4',           # Teal for quantum
            'classical': '#45B7D1'          # Blue for classical
        }
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i//3, i%3]
            
            # Get values and colors
            values = [self.results[model][metric] for model in models]
            model_colors = [colors[self.results[model]['type']] for model in models]
            
            # Create bar plot
            bars = ax.bar(range(len(models)), values, color=model_colors, alpha=0.8)
            
            # Highlight quantum-inspired models
            for j, (model, color) in enumerate(zip(models, model_colors)):
                if color == colors['quantum_inspired']:
                    bars[j].set_edgecolor('red')
                    bars[j].set_linewidth(2)
            
            # Customize plot
            ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_ylim(0, 1)
            
            # Rotate x-axis labels
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                ax.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Add grid
            ax.grid(True, alpha=0.3)
        
        # Remove the last subplot
        axes[1, 2].remove()
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor=colors['quantum_inspired'], label='Quantum-Inspired 100-Qubit'),
            plt.Rectangle((0,0),1,1, facecolor=colors['quantum'], label='Quantum Models'),
            plt.Rectangle((0,0),1,1, facecolor=colors['classical'], label='Classical Models')
        ]
        fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.98, 0.02))
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(self.output_dir, exist_ok=True)
        plt.savefig(os.path.join(self.output_dir, 'quantum_100_performance_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Performance comparison chart saved!")
    
    def create_radar_chart(self):
        """Create radar chart comparing model performance"""
        print("🎯 Creating radar chart...")
        
        # Select key models for radar chart
        key_models = [
            'Quantum-Inspired 100-Qubit (Ensemble)',
            'Quantum-Inspired Quantum Rf',
            'Quantum-Inspired Quantum Nn',
            'Classical Random Forest',
            'Classical Svm',
            'Quantum Qsvm',
            'Quantum Vqc'
        ]
        
        # Filter available models
        available_models = [model for model in key_models if model in self.results]
        
        # Metrics for radar chart
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
        
        # Number of variables
        N = len(metrics)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Color scheme
        colors = ['#FF6B6B', '#FF8E8E', '#FFB1B1', '#4ECDC4', '#45B7D1', '#FFD93D', '#FF6B9D']
        
        # Plot each model
        for i, model in enumerate(available_models):
            values = [self.results[model][metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
        
        # Customize the plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        # Add title and legend
        plt.title('🔬 Quantum-Inspired 100-Qubit vs All Models - Radar Chart', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'quantum_100_radar_chart.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Radar chart saved!")
    
    def create_accuracy_evolution_chart(self):
        """Create chart showing accuracy evolution from quantum to quantum-inspired"""
        print("📈 Creating accuracy evolution chart...")
        
        # Define model categories and their accuracies
        categories = {
            'Original Quantum (2-10 qubits)': 0.52,  # Average of original quantum models
            'Classical Models': 0.98,               # Average of classical models
            'Quantum-Inspired 100-Qubit': 0.9794    # Our new result
        }
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create bar plot
        bars = ax.bar(categories.keys(), categories.values(), 
                     color=['#4ECDC4', '#45B7D1', '#FF6B6B'], alpha=0.8)
        
        # Highlight quantum-inspired
        bars[2].set_edgecolor('red')
        bars[2].set_linewidth(3)
        
        # Add value labels
        for i, (category, value) in enumerate(categories.items()):
            ax.text(i, value + 0.005, f'{value:.3f}', ha='center', va='bottom', 
                   fontsize=12, fontweight='bold')
        
        # Customize plot
        ax.set_title('🚀 Accuracy Evolution: From Quantum to Quantum-Inspired 100-Qubit', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        ax.annotate('Original quantum models\nstruggled with limited qubits', 
                   xy=(0, 0.52), xytext=(0.5, 0.3),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   fontsize=10, ha='center')
        
        ax.annotate('Quantum-inspired 100-qubit\ncloses the gap with classical!', 
                   xy=(2, 0.9794), xytext=(1.5, 0.8),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2),
                   fontsize=10, ha='center')
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'quantum_100_accuracy_evolution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Accuracy evolution chart saved!")
    
    def create_feature_complexity_chart(self):
        """Create chart showing feature complexity progression"""
        print("🔬 Creating feature complexity chart...")
        
        # Define complexity levels
        complexities = {
            'Original Quantum (2-10 qubits)': {
                'features': 29,
                'qubits': 10,
                'accuracy': 0.52
            },
            'Classical Models': {
                'features': 29,
                'qubits': 'N/A',
                'accuracy': 0.98
            },
            'Quantum-Inspired 100-Qubit': {
                'features': 580,
                'qubits': 100,
                'accuracy': 0.9794
            }
        }
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Feature count comparison
        feature_counts = [complexities[cat]['features'] for cat in complexities.keys()]
        colors = ['#4ECDC4', '#45B7D1', '#FF6B6B']
        
        bars1 = ax1.bar(complexities.keys(), feature_counts, color=colors, alpha=0.8)
        bars1[2].set_edgecolor('red')
        bars1[2].set_linewidth(3)
        
        ax1.set_title('🔬 Feature Complexity Evolution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Features', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, count in enumerate(feature_counts):
            ax1.text(i, count + 10, str(count), ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Accuracy vs Feature Count
        accuracies = [complexities[cat]['accuracy'] for cat in complexities.keys()]
        
        ax2.scatter(feature_counts, accuracies, s=200, c=colors, alpha=0.8, edgecolors='black')
        ax2.scatter(feature_counts[2], accuracies[2], s=300, c='red', alpha=0.8, 
                   edgecolors='red', linewidth=3, zorder=5)
        
        # Add labels
        for i, cat in enumerate(complexities.keys()):
            ax2.annotate(cat, (feature_counts[i], accuracies[i]), 
                        xytext=(10, 10), textcoords='offset points', fontsize=10)
        
        ax2.set_title('📊 Accuracy vs Feature Complexity', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Number of Features', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, max(feature_counts) * 1.1)
        ax2.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'quantum_100_feature_complexity.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Feature complexity chart saved!")
    
    def create_summary_table(self):
        """Create summary table of all results"""
        print("📋 Creating summary table...")
        
        # Prepare data for table
        table_data = []
        for model, metrics in self.results.items():
            table_data.append([
                model,
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1_score']:.4f}",
                f"{metrics['roc_auc']:.4f}",
                metrics['type'].title()
            ])
        
        # Sort by accuracy
        table_data.sort(key=lambda x: float(x[1]), reverse=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, len(table_data) * 0.4 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=table_data,
                        colLabels=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC', 'Type'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.25, 0.12, 0.12, 0.12, 0.12, 0.12, 0.15])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color header
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color quantum-inspired models
        for i, row in enumerate(table_data):
            if 'Quantum-Inspired' in row[0]:
                for j in range(len(row)):
                    table[(i+1, j)].set_facecolor('#FFE6E6')
                    table[(i+1, j)].set_text_props(weight='bold')
        
        # Color quantum models
        for i, row in enumerate(table_data):
            if row[6] == 'Quantum' and 'Quantum-Inspired' not in row[0]:
                for j in range(len(row)):
                    table[(i+1, j)].set_facecolor('#E6F7F5')
        
        # Color classical models
        for i, row in enumerate(table_data):
            if row[6] == 'Classical':
                for j in range(len(row)):
                    table[(i+1, j)].set_facecolor('#E6F3F7')
        
        plt.title('🏆 Complete Model Performance Summary\nQuantum-Inspired 100-Qubit vs All Models', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'quantum_100_summary_table.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Summary table saved!")
    
    def create_all_visualizations(self):
        """Create all visualizations"""
        print("🎨 Creating comprehensive visualizations...")
        
        # Load metrics
        self.load_all_metrics()
        
        # Create all charts
        self.create_performance_comparison_chart()
        self.create_radar_chart()
        self.create_accuracy_evolution_chart()
        self.create_feature_complexity_chart()
        self.create_summary_table()
        
        print("\n🎉 All visualizations completed!")
        print(f"📁 Check the '{self.output_dir}' directory for all charts")

def main():
    """Main function to run all visualizations"""
    print("🚀 Starting Quantum-Inspired 100-Qubit Visualization Suite")
    print("="*70)
    
    visualizer = Quantum100ComparisonVisualizer()
    visualizer.create_all_visualizations()
    
    print("\n✅ Visualization suite completed!")
    print("🔬 Quantum-Inspired 100-Qubit results have been compared with all previous models!")

if __name__ == "__main__":
    main() 