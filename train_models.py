"""
AI-Powered Email Spam Detection - Model Training and Evaluation Module
This module trains multiple ML models and evaluates their performance
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json


class SpamClassifierTrainer:
    """Class to train and evaluate spam classification models"""
    
    def __init__(self, data_path):
        """
        Initialize the trainer
        
        Args:
            data_path: Path to the train/test data file
        """
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load the preprocessed train/test data"""
        print("Loading preprocessed data...")
        data = np.load(self.data_path)
        self.X_train = data['X_train']
        self.X_test = data['X_test']
        self.y_train = data['y_train']
        self.y_test = data['y_test']
        
        print(f"Training samples: {self.X_train.shape[0]}")
        print(f"Testing samples: {self.X_test.shape[0]}")
        print(f"Features: {self.X_train.shape[1]}")
        
    def initialize_models(self):
        """Initialize all machine learning models"""
        print("\nInitializing models...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='liblinear'
            ),
            'Naive Bayes': MultinomialNB(alpha=1.0),
            'Support Vector Machine': SVC(
                kernel='linear',
                probability=True,
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
        }
        
        print(f"Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"  - {name}")
    
    def train_models(self):
        """Train all models"""
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            start_time = datetime.now()
            
            model.fit(self.X_train, self.y_train)
            
            training_time = (datetime.now() - start_time).total_seconds()
            print(f"Training completed in {training_time:.2f} seconds")
            
            # Store training time
            if name not in self.results:
                self.results[name] = {}
            self.results[name]['training_time'] = training_time
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n" + "="*60)
        print("EVALUATING MODELS")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            
            # ROC curve
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Precision-Recall curve
            prec, rec, _ = precision_recall_curve(self.y_test, y_pred_proba)
            pr_auc = auc(rec, prec)
            
            # Store results
            self.results[name].update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'fpr': fpr,
                'tpr': tpr,
                'precision_curve': prec,
                'recall_curve': rec,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            })
            
            # Print results
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  ROC AUC:   {roc_auc:.4f}")
            print(f"  PR AUC:    {pr_auc:.4f}")
    
    def print_summary(self):
        """Print summary comparison of all models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        # Create comparison dataframe
        comparison_data = []
        for name, results in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': f"{results['accuracy']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1-Score': f"{results['f1_score']:.4f}",
                'ROC AUC': f"{results['roc_auc']:.4f}",
                'Training Time (s)': f"{results['training_time']:.2f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print("\n" + df_comparison.to_string(index=False))
        
        # Find best model
        best_model = max(
            self.results.items(), 
            key=lambda x: x[1]['f1_score']
        )
        print(f"\nBest Model (by F1-Score): {best_model[0]}")
        print(f"F1-Score: {best_model[1]['f1_score']:.4f}")
    
    def save_models(self, output_dir):
        """Save all trained models"""
        print(f"\nSaving models to {output_dir}...")
        
        for name, model in self.models.items():
            filename = name.lower().replace(' ', '_') + '.pkl'
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            
            print(f"  Saved: {filename}")
    
    def save_results(self, output_path):
        """Save evaluation results"""
        print(f"\nSaving results to {output_path}...")
        
        # Prepare results for JSON serialization
        results_json = {}
        for name, results in self.results.items():
            results_json[name] = {
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1_score']),
                'roc_auc': float(results['roc_auc']),
                'pr_auc': float(results['pr_auc']),
                'training_time': float(results['training_time']),
                'confusion_matrix': results['confusion_matrix'].tolist()
            }
        
        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=4)
        
        print("Results saved successfully")
    
    def get_results(self):
        """Return the results dictionary"""
        return self.results


def main():
    """Main execution function"""
    # Set paths
    data_path = '/home/ubuntu/spam_detection/data/train_test_data.npz'
    models_dir = '/home/ubuntu/spam_detection/models'
    results_path = '/home/ubuntu/spam_detection/results/model_results.json'
    
    # Initialize trainer
    trainer = SpamClassifierTrainer(data_path)
    
    # Load data
    trainer.load_data()
    
    # Initialize models
    trainer.initialize_models()
    
    # Train models
    trainer.train_models()
    
    # Evaluate models
    trainer.evaluate_models()
    
    # Print summary
    trainer.print_summary()
    
    # Save models and results
    trainer.save_models(models_dir)
    trainer.save_results(results_path)
    
    print("\n" + "="*60)
    print("MODEL TRAINING AND EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return trainer


if __name__ == "__main__":
    trainer = main()

