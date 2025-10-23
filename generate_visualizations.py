"""
AI-Powered Email Spam Detection - Visualization Generation Module
This module generates all visualizations for the project report
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from sklearn.metrics import confusion_matrix
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set figure DPI for high quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


class VisualizationGenerator:
    """Class to generate all project visualizations"""
    
    def __init__(self, data_dir, results_path, output_dir):
        """
        Initialize the visualization generator
        
        Args:
            data_dir: Directory containing data files
            results_path: Path to model results JSON
            output_dir: Directory to save visualizations
        """
        self.data_dir = data_dir
        self.results_path = results_path
        self.output_dir = output_dir
        self.results = None
        self.df = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self):
        """Load preprocessed data and results"""
        print("Loading data and results...")
        
        # Load preprocessed data
        self.df = pd.read_csv(
            os.path.join(self.data_dir, 'preprocessed_data.csv')
        )
        
        # Load results
        with open(self.results_path, 'r') as f:
            self.results = json.load(f)
        
        print(f"Loaded {len(self.df)} messages")
        print(f"Loaded results for {len(self.results)} models")
    
    def plot_dataset_distribution(self):
        """Plot spam vs ham distribution"""
        print("\nGenerating dataset distribution plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Count plot
        label_counts = self.df['label'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        
        axes[0].bar(label_counts.index, label_counts.values, color=colors, alpha=0.7)
        axes[0].set_xlabel('Class', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[0].set_title('Dataset Distribution', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(label_counts.values):
            axes[0].text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        axes[1].pie(
            label_counts.values, 
            labels=label_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'}
        )
        axes[1].set_title('Class Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, '01_dataset_distribution.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")
    
    def plot_message_length_distribution(self):
        """Plot message length distribution"""
        print("\nGenerating message length distribution plot...")
        
        # Calculate message lengths
        self.df['message_length'] = self.df['message'].str.len()
        self.df['word_count'] = self.df['message'].str.split().str.len()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Character length distribution
        for label in ['ham', 'spam']:
            data = self.df[self.df['label'] == label]['message_length']
            axes[0].hist(data, bins=50, alpha=0.6, label=label, edgecolor='black')
        
        axes[0].set_xlabel('Message Length (characters)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('Message Length Distribution', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Word count distribution
        for label in ['ham', 'spam']:
            data = self.df[self.df['label'] == label]['word_count']
            axes[1].hist(data, bins=50, alpha=0.6, label=label, edgecolor='black')
        
        axes[1].set_xlabel('Word Count', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1].set_title('Word Count Distribution', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, '02_message_length_distribution.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        print("\nGenerating confusion matrices...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.ravel()
        
        model_names = list(self.results.keys())
        
        for idx, model_name in enumerate(model_names):
            cm = np.array(self.results[model_name]['confusion_matrix'])
            
            # Create heatmap
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                ax=axes[idx],
                cbar_kws={'label': 'Count'},
                annot_kws={'fontsize': 14, 'fontweight': 'bold'}
            )
            
            axes[idx].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontsize=12, fontweight='bold')
            axes[idx].set_title(f'{model_name}\nConfusion Matrix', 
                               fontsize=13, fontweight='bold')
            axes[idx].set_xticklabels(['Ham', 'Spam'], fontsize=11)
            axes[idx].set_yticklabels(['Ham', 'Spam'], fontsize=11)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, '03_confusion_matrices.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")
    
    def plot_model_comparison(self):
        """Plot model performance comparison"""
        print("\nGenerating model comparison plot...")
        
        # Prepare data
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        data = {metric: [] for metric in metrics}
        for model in models:
            for metric in metrics:
                data[metric].append(self.results[model][metric])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(models))
        width = 0.2
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for idx, metric in enumerate(metrics):
            offset = width * (idx - 1.5)
            bars = ax.bar(
                x + offset, 
                data[metric], 
                width, 
                label=metric.replace('_', ' ').title(),
                color=colors[idx],
                alpha=0.8
            )
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2., 
                    height,
                    f'{height:.3f}',
                    ha='center', 
                    va='bottom',
                    fontsize=8,
                    fontweight='bold'
                )
        
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.legend(fontsize=10, loc='lower right')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, '04_model_comparison.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        print("\nGenerating ROC curves...")
        
        # Load the actual predictions to compute ROC curves
        data = np.load(os.path.join(self.data_dir, 'train_test_data.npz'))
        X_test = data['X_test']
        y_test = data['y_test']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for idx, (model_name, color) in enumerate(zip(self.results.keys(), colors)):
            # Load model
            model_filename = model_name.lower().replace(' ', '_') + '.pkl'
            model_path = os.path.join(
                os.path.dirname(self.data_dir), 
                'models', 
                model_filename
            )
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Get predictions
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate ROC curve
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot
            ax.plot(
                fpr, 
                tpr, 
                color=color,
                lw=2.5,
                label=f'{model_name} (AUC = {roc_auc:.3f})'
            )
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, '05_roc_curves.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")
    
    def plot_precision_recall_curves(self):
        """Plot Precision-Recall curves for all models"""
        print("\nGenerating Precision-Recall curves...")
        
        # Load the actual predictions
        data = np.load(os.path.join(self.data_dir, 'train_test_data.npz'))
        X_test = data['X_test']
        y_test = data['y_test']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for idx, (model_name, color) in enumerate(zip(self.results.keys(), colors)):
            # Load model
            model_filename = model_name.lower().replace(' ', '_') + '.pkl'
            model_path = os.path.join(
                os.path.dirname(self.data_dir), 
                'models', 
                model_filename
            )
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Get predictions
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate PR curve
            from sklearn.metrics import precision_recall_curve, auc
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall, precision)
            
            # Plot
            ax.plot(
                recall, 
                precision, 
                color=color,
                lw=2.5,
                label=f'{model_name} (AUC = {pr_auc:.3f})'
            )
        
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curves - Model Comparison', 
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower left')
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, '06_precision_recall_curves.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")
    
    def plot_training_time_comparison(self):
        """Plot training time comparison"""
        print("\nGenerating training time comparison plot...")
        
        models = list(self.results.keys())
        times = [self.results[model]['training_time'] for model in models]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        bars = ax.bar(models, times, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Model Training Time Comparison', fontsize=14, fontweight='bold')
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., 
                height,
                f'{height:.2f}s',
                ha='center', 
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, '07_training_time_comparison.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")
    
    def plot_feature_importance(self):
        """Plot feature importance for Random Forest"""
        print("\nGenerating feature importance plot...")
        
        # Load Random Forest model
        model_path = os.path.join(
            os.path.dirname(self.data_dir), 
            'models', 
            'random_forest.pkl'
        )
        
        with open(model_path, 'rb') as f:
            rf_model = pickle.load(f)
        
        # Load vectorizer to get feature names
        vectorizer_path = os.path.join(self.data_dir, 'tfidf_vectorizer.pkl')
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Get feature importance
        importances = rf_model.feature_importances_
        feature_names = vectorizer.get_feature_names_out()
        
        # Get top 20 features
        indices = np.argsort(importances)[-20:]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        bars = ax.barh(range(len(top_features)), top_importances, color=colors, alpha=0.8)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=10)
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Features', fontsize=12, fontweight='bold')
        ax.set_title('Top 20 Feature Importance (Random Forest)', 
                     fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, '08_feature_importance.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        self.load_data()
        self.plot_dataset_distribution()
        self.plot_message_length_distribution()
        self.plot_confusion_matrices()
        self.plot_model_comparison()
        self.plot_roc_curves()
        self.plot_precision_recall_curves()
        self.plot_training_time_comparison()
        self.plot_feature_importance()
        
        print("\n" + "="*60)
        print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("="*60)


def main():
    """Main execution function"""
    data_dir = '/home/ubuntu/spam_detection/data'
    results_path = '/home/ubuntu/spam_detection/results/model_results.json'
    output_dir = '/home/ubuntu/spam_detection/results'
    
    generator = VisualizationGenerator(data_dir, results_path, output_dir)
    generator.generate_all_visualizations()


if __name__ == "__main__":
    main()

